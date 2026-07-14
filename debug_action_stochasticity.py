# #!/usr/bin/env python3
# import argparse
# import random
# from pathlib import Path

# import numpy as np
# import torch
# import yaml
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor

# from src.environment.environment import MultiTaskAllocationEnv
# from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final
# from src.models.sb3_gnn_policy import RTGNNPolicy


# def load_config(config_path: str):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)


# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def make_env(config, seed: int):
#     env_cfg = config["environment"]
#     agents = np.load(env_cfg["agents_file"], allow_pickle=True)

#     batches = []
#     for i in range(env_cfg["n_batches"]):
#         batch_file = Path(env_cfg["data_dir"]) / f"tasks_batch_{i}.npy"
#         if batch_file.exists():
#             batches.append(np.load(batch_file, allow_pickle=True))

#     base_env = MultiTaskAllocationEnv(
#         agents_cont_coord_array=agents,
#         task_cont_coord_array=batches,
#         radius=env_cfg["radius"],
#         feature_size=env_cfg["feature_size"],
#         use_true_id=env_cfg["use_true_id"],
#         all_batches=True,
#     )
#     env = WarehouseEnvSB3Final(
#         base_env,
#         assignment_interval=env_cfg["assignment_interval"],
#         k_max=env_cfg.get("k_max", 5),
#     )
#     env = Monitor(env)

#     # IMPORTANT: reset with a fixed seed so both runs start identical
#     env.reset(seed=seed)
#     return env


# def run_one_episode(model: PPO, env, deterministic: bool, max_steps: int = 5000):
#     obs, info = env.reset()  # do NOT pass seed here; we already seeded above
#     actions = []
#     done = False
#     step = 0
#     ep_reward = 0.0

#     while not done and step < max_steps:
#         action, _ = model.predict(obs, deterministic=deterministic)

#         # store a copy so it doesn't get mutated
#         if isinstance(action, np.ndarray):
#             actions.append(action.copy())
#         else:
#             actions.append(int(action))

#         obs, reward, terminated, truncated, info = env.step(action)
#         ep_reward += float(reward)
#         done = bool(terminated) or bool(truncated)
#         step += 1

#     return actions, ep_reward, step


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--model", required=True, help="Path to PPO zip, e.g. checkpoints_ppo/seed_42/ppo_final.zip")
#     ap.add_argument("--seed", type=int, default=123)
#     ap.add_argument("--n", type=int, default=50, help="How many actions to print")
#     args = ap.parse_args()

#     config = load_config(args.config)

#     # Load model
#     model = PPO.load(args.model, custom_objects={"policy_class": RTGNNPolicy})

#     # Run deterministic episode
#     set_seed(args.seed)
#     env_det = make_env(config, seed=args.seed)
#     a_det, r_det, steps_det = run_one_episode(model, env_det, deterministic=True)

#     # Run stochastic episode from same initial seed
#     set_seed(args.seed)
#     env_sto = make_env(config, seed=args.seed)
#     a_sto, r_sto, steps_sto = run_one_episode(model, env_sto, deterministic=False)

#     print("\n=== SUMMARY ===")
#     print(f"det: reward={r_det:.3f}, steps={steps_det}")
#     print(f"sto: reward={r_sto:.3f}, steps={steps_sto}")

#     print("\n=== FIRST ACTIONS ===")
#     n = min(args.n, len(a_det), len(a_sto))
#     same = 0
#     for i in range(n):
#         d = a_det[i]
#         s = a_sto[i]
#         eq = np.array_equal(d, s) if isinstance(d, np.ndarray) else (d == s)
#         same += int(eq)
#         print(f"{i:03d}  det={d}   sto={s}   same={eq}")

#     print(f"\nSame among first {n}: {same}/{n}")

# if __name__ == "__main__":
#     main()