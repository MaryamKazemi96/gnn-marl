#!/usr/bin/env python3
"""
Evaluate trained GNN-PPO model (deterministic + stochastic)
Clean + robust version
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.environment import MultiAgentTaskEnv
from src.models.sb3_gnn_policy import RTGNNPolicy


# =========================================================
# Utils
# =========================================================

def load_json(p: Path):
    return json.loads(p.read_text())


def save_json(data, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    print(f"✓ {p}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_latest_run(seed: int) -> Path:
    base = Path("runs") / f"seed_{seed}"

    if not base.exists():
        raise FileNotFoundError(f"Seed folder not found: {base}")

    runs = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime)

    if not runs:
        raise FileNotFoundError(f"No runs found in {base}")

    return runs[-1]
# =========================================================
# Data
# =========================================================

def load_data(data_dir: str):
    p = Path(data_dir)

    agents = np.load(p / "agents.npy", allow_pickle=True)

    tasks = []
    i = 0
    while (p / f"tasks_batch_{i}.npy").exists():
        tasks.append(np.load(p / f"tasks_batch_{i}.npy", allow_pickle=True))
        i += 1

    return agents, tasks


def make_env(agents, tasks, config, seed):

    def _init():
        env = MultiAgentTaskEnv(
            agents=agents,
            tasks_batches=tasks,
            K_max=config.get("K_max", 5),
            N_max=config.get("N_max", 15),
            E_max=config.get("E_max", 50),
            use_xy_pickup=config.get("use_xy_pickup", False),
            normalize_features=config.get("normalize_features", True),
            use_node_type=config.get("use_node_type", True),
            use_ego_robot=config.get("use_ego_robot", True),
            use_edge_rt=config.get("use_edge_rt", False),
            two_hop=config.get("two_hop", False),
            vicinity_m=config.get("vicinity_m", 20.0),
            max_steps=config.get("max_steps", 1000),
            max_robot_capacity=config.get("max_robot_capacity", 2),
            max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
            max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
        )
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])


# =========================================================
# Model
# =========================================================

def pick_model(run_dir: Path) -> Path:
    model_dir = run_dir 

    models = list(model_dir.glob("*.zip"))

    if not models:
        raise FileNotFoundError(f"No models found in {model_dir}")

    return max(models, key=lambda p: p.stat().st_mtime)
# =========================================================
# Evaluation core
# =========================================================

def run_eval(model, env, episodes, deterministic):

    rewards, lengths, completed, obsolete = [], [], [], []

    for ep in range(episodes):

        obs = env.reset()
        done = [False]

        ep_r, ep_l = 0.0, 0
        ep_c, ep_o = 0, 0

        while not done[0]:

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = env.step(action)

            done = dones

            ep_r += float(r[0])
            ep_l += 1

            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            if isinstance(info, dict):
                ep_c = info.get("completed_count", ep_c)
                ep_o = info.get("obsolete_count", ep_o)

        rewards.append(ep_r)
        lengths.append(ep_l)
        completed.append(ep_c)
        obsolete.append(ep_o)

    r = np.array(rewards, dtype=float)

    return {
        "rewards": rewards,
        "lengths": lengths,
        "completed": completed,
        "obsolete": obsolete,
        "stats": {
            "reward_mean": float(r.mean()),
            "reward_std": float(r.std()),
            "min": float(r.min()),
            "max": float(r.max()),
            "completed": float(np.mean(completed)),
            "obsolete": float(np.mean(obsolete)),
        }
    }


# =========================================================
# Main
# =========================================================

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--output", type=str, default=None,
                help="Optional override output dir (defaults to run_dir/eval_results)")

    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    print("\n============================")
    print(" PPO EVALUATION")
    print("============================\n")

    # -------------------------
    # Load data
    # -------------------------
    agents, tasks = load_data(args.data_dir)

    # -------------------------
    # Select run folder
    # -------------------------
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(args.seed)

    print(" Selected run:", run_dir)

    model_path = pick_model(run_dir)
    print("Using model:", model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": RTGNNPolicy},
    )

    print(f"✓ Loaded model\n")

    # =====================================================
    # Deterministic (fresh env)
    # =====================================================
    print("Running deterministic...")

    env_det = make_env(agents, tasks, config, args.seed)
    det = run_eval(model, env_det, args.episodes, True)

    env_det.close()

    # =====================================================
    # Stochastic (fresh env)
    # =====================================================
    print("Running stochastic...")

    env_sto = make_env(agents, tasks, config, args.seed + 1)
    sto = run_eval(model, env_sto, args.episodes, False)

    env_sto.close()

    # -------------------------
    # Save results
    # -------------------------
    # -------------------------
    # Save results inside run
    # -------------------------
    if args.output is not None:
        out = Path(args.output)
    else:
        out = run_dir / "eval_results"

    out.mkdir(parents=True, exist_ok=True)

    save_json(det, out / "deterministic.json")
    save_json(sto, out / "stochastic.json")

    print("\n============================")
    print("RESULTS")
    print("============================")

    print(f"Deterministic: {det['stats']['reward_mean']:.2f} ± {det['stats']['reward_std']:.2f}")
    print(f"Stochastic:    {sto['stats']['reward_mean']:.2f} ± {sto['stats']['reward_std']:.2f}")
    print(f"\nSaved → {out}\n")


if __name__ == "__main__":
    main()