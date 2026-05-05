#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.environment.environment import MultiTaskAllocationEnv
from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
from src.models.sb3_gnn_policy import RTGNNPolicy


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config, seed: int):
    env_cfg = config["environment"]
    agents = np.load(env_cfg["agents_file"], allow_pickle=True)

    batches = []
    for i in range(env_cfg["n_batches"]):
        batch_file = Path(env_cfg["data_dir"]) / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batches.append(np.load(batch_file, allow_pickle=True))

    base_env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=env_cfg["radius"],
        feature_size=env_cfg["feature_size"],
        use_true_id=env_cfg["use_true_id"],
        all_batches=True,
    )
    env = WarehouseEnvSB3Final(
        base_env,
        assignment_interval=env_cfg["assignment_interval"],
        k_max=env_cfg.get("k_max", 5),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def pick(d, keys):
    out = {}
    for k in keys:
        if isinstance(d, dict) and k in d:
            out[k] = d[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    model = PPO.load(args.model, custom_objects={"policy_class": RTGNNPolicy})
    env = make_env(config, seed=args.seed)

    obs, info = env.reset()  # same start state due to env.reset(seed=...)
    print("Initial info keys:", sorted(list(info.keys())) if isinstance(info, dict) else type(info))

    total = 0.0
    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)

        # Print the important bits (we don't know exact keys yet, so we probe common ones)
        print(f"\n--- step {t} ---")
        print("action:", action)

        if isinstance(info, dict):
            # candidate/action mask visibility
            print("info[cand_task_ids]/mask:",
                  ("cand_task_ids" in info), ("action_mask" in info))

            # reward components (common patterns)
            print("reward_parts:", pick(info, [
    "rew/pickups_this_step",
    "rew/deliveries_this_step",
    "rew/obsolete_this_step",
    "rew/step_penalty",
    "rew/sum_rewards",
]))

            # any assignment debug (common patterns)
            print("assignment_debug:", pick(info, [
                "assignments", "final_assignments", "decoded_assignments",
                "assigned_task_ids", "chosen_task_ids"
            ]))

            # episode summary keys (your other scripts read these)
            print("episode_flags:", pick(info, [
                "episode_completed", "episode_obsolete", "episode_reward"
            ]))

        done = bool(terminated) or bool(truncated)
        print("reward:", float(reward), "done:", done)

        if done:
            break

    print("\nTOTAL reward (partial):", total)


if __name__ == "__main__":
    main()