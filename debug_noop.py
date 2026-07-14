"""
debug_noop_plurality.py

Diagnoses the plurality-vs-majority gap in deterministic evaluation for a
"K real candidates + 1 noop" categorical action space.

Argmax (deterministic=True) always returns whichever single option has the
highest individual probability. If probability is split fairly evenly
across several real candidates while noop holds a narrow *plurality*
(highest individual share, but well under 50%), argmax will pick noop even
in states where sampling would pick "some real action" the large majority
of the time (since sampling reflects the true *aggregate* preference).

This script runs episodes with the actual trained policy and, at every
per-robot decision where at least one real candidate was available, records:
  - P(noop)                         — individual probability of noop
  - P(best real candidate)          — individual probability of the top
                                       real candidate
  - sum(P(real candidates))         — aggregate probability of "act at all"
  - is_plurality  = P(noop) is the single largest value (this is what
                    argmax/deterministic actually acts on)
  - is_majority   = P(noop) > 0.5   (this is what "noop is genuinely the
                    best choice" would require)

If is_plurality is frequently True while is_majority is frequently False,
that confirms the plurality-vs-majority mismatch as the cause of a
deterministic policy that looks far worse than its own stochastic sampling
or training-time rollout reward.

Usage:
    python3 debug_noop_plurality.py --seed 42 --episodes 20
    python3 debug_noop_plurality.py --run-id 20260712_143000 --seed 7
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.models.sb3_gnn_policy import RTGNNPolicy
from eval_ppo import load_config, load_data, make_env, pick_model, find_latest_run, set_seed


def run_diagnostic(model: PPO, env, episodes: int, deterministic: bool):
    K = env.get_attr("action_space")[0].nvec[0] - 1  # noop index == K_max

    plurality_flags = []   # 1 if P(noop) is the single largest value
    majority_flags = []    # 1 if P(noop) > 0.5
    p_noop_values = []
    p_real_sum_values = []
    p_best_real_values = []

    for ep in range(episodes):
        obs = env.reset()
        done = [False]

        while not done[0]:
            # obs is already batched (DummyVecEnv, num_envs=1) — pass through as-is.
            probs = model.policy.get_action_probs(obs)  # [1, R, K+1]
            probs = probs[0]  # [R, K+1]

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = env.step(action)
            done = dones

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            mask = info.get("action_mask") if isinstance(info, dict) else None

            for rid in range(probs.shape[0]):
                real_mask = mask[rid, :K].astype(bool) if mask is not None else None
                has_real_candidate = bool(real_mask.any()) if real_mask is not None else True
                if not has_real_candidate:
                    continue  # only care about decisions where noop was a genuine choice

                p_noop = float(probs[rid, K])
                p_real = probs[rid, :K]
                p_real_sum = float(p_real.sum())
                p_best_real = float(p_real.max())

                is_plurality = bool(p_noop == float(probs[rid].max()))
                # plurality = P(noop) is the single largest entry among all K+1 options
                is_majority = bool(p_noop > 0.5)

                plurality_flags.append(is_plurality)
                majority_flags.append(is_majority)
                p_noop_values.append(p_noop)
                p_real_sum_values.append(p_real_sum)
                p_best_real_values.append(p_best_real)

    n = len(plurality_flags)
    if n == 0:
        print("No decisions with a real candidate were observed — check env/episodes.")
        return

    plurality_rate = float(np.mean(plurality_flags))
    majority_rate = float(np.mean(majority_flags))

    print(f"\n{'='*70}")
    print(f"Decisions analyzed (real candidate available): {n}")
    print(f"{'='*70}")
    print(f"P(noop) is the single largest option (plurality) : {plurality_rate:.4f}")
    print(f"P(noop) > 0.5                        (majority)  : {majority_rate:.4f}")
    print(f"\nMean P(noop)                     : {np.mean(p_noop_values):.4f}")
    print(f"Mean P(best single real candidate): {np.mean(p_best_real_values):.4f}")
    print(f"Mean sum(P(real candidates))      : {np.mean(p_real_sum_values):.4f}")

    if plurality_rate > 0.5 and majority_rate < 0.3:
        print(
            "\n>>> CONFIRMED: noop wins the argmax far more often than it wins a true\n"
            "    majority of probability mass. This is the plurality-vs-majority gap —\n"
            "    deterministic=True is not a reliable readout of 'what the policy thinks\n"
            "    is best' for this action space; it's picking the single largest option\n"
            "    even when the aggregate 'act' probability is much larger than P(noop)."
        )
    else:
        print(
            "\n>>> Pattern not clearly present in this sample — noop's plurality wins are\n"
            "    reasonably close to true majority wins, or both are low. The det-vs-stoch\n"
            "    reward gap (if any) likely has a different cause."
        )
    print(f"{'='*70}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true", default=True,
                     help="Run under deterministic action selection (default: True, "
                          "since that's the mode we're diagnosing).")
    ap.add_argument("--stochastic", action="store_true",
                     help="Run under stochastic sampling instead, for comparison.")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    agents, tasks = load_data(args.data_dir)

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(args.seed, args.run_id)
    model_path = pick_model(run_dir)
    print("Using model:", model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device, custom_objects={"policy_class": RTGNNPolicy})

    deterministic = not args.stochastic
    print(f"Mode: {'deterministic' if deterministic else 'stochastic'}")

    env = make_env(agents, tasks, config, args.seed)
    run_diagnostic(model, env, args.episodes, deterministic)
    env.close()


if __name__ == "__main__":
    main()