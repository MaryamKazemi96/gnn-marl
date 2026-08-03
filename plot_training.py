# #!/usr/bin/env python3
# """
# Unified plotting script for GNN-PPO runs.

# Features:
# - Auto-detect latest run if not provided
# - Loads TensorBoard + monitor.csv
# - Baselines from external folder
# - Saves all plots into run_dir/plots/
# - --multi-seed: overlays reward curves (mean +/- std band) across every
#   seed's latest run, for runs launched with train_ppo.py's multi-seed loop
# """

# import matplotlib
# matplotlib.use("Agg")  # headless: must be set before importing pyplot.
# # Without this, matplotlib can pick a Qt-based interactive backend, and if
# # opencv-python (cv2) is installed it ships its own bundled Qt plugins that
# # shadow the system ones -> "Could not load the Qt platform plugin xcb" ->
# # hard crash (SIGABRT). This script only ever writes PNG files, so headless
# # Agg is correct here regardless of what else is installed in the venv.

# import argparse
# from pathlib import Path
# import json

# import numpy as np
# import matplotlib.pyplot as plt

# from tensorboard.backend.event_processing import event_accumulator


# # =========================================================
# # Utils
# # =========================================================

# def latest_run_id(runs_root: Path) -> Path:
#     """Return the most recently modified runs/run_{id}/ folder."""
#     run_dirs = sorted(runs_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
#     if not run_dirs:
#         raise FileNotFoundError(f"No run_* directories found in {runs_root}")
#     return run_dirs[-1]


# def seed_dir_in_run(run_root: Path, seed: int) -> Path:
#     """runs/run_{id}/seed_{seed}/ — raises with a helpful listing if that
#     seed wasn't part of this particular sweep."""
#     sd = run_root / f"seed_{seed}"
#     if not sd.exists():
#         available = sorted(p.name for p in run_root.glob("seed_*"))
#         raise FileNotFoundError(
#             f"No seed_{seed} under {run_root}. Available: {available}"
#         )
#     return sd


# def moving_average(x, w):
#     if len(x) < w:
#         return x
#     return np.convolve(x, np.ones(w) / w, mode="valid")


# def extract_baseline_stats(v):
#     """Robust baseline parser (handles multiple formats)."""
#     if not isinstance(v, dict):
#         return None, None

#     if "stats" in v:
#         s = v["stats"]
#         return s.get("reward_mean"), s.get("reward_std")

#     return v.get("reward_mean"), v.get("reward_std")


# # =========================================================
# # TensorBoard
# # =========================================================

# def load_tensorboard(tb_dir: Path):
#     runs = list(tb_dir.glob("PPO_*"))
#     if not runs:
#         print("⚠️ No TensorBoard runs found")
#         return None

#     latest = max(runs, key=lambda p: p.stat().st_mtime)
#     print(f"📈 Loading TensorBoard: {latest}")

#     ea = event_accumulator.EventAccumulator(str(latest))
#     ea.Reload()

#     data = {}
#     for tag in ea.Tags()["scalars"]:
#         events = ea.Scalars(tag)
#         data[tag] = {
#             "steps": np.array([e.step for e in events]),
#             "values": np.array([e.value for e in events]),
#         }

#     return data


# def plot_tb_rewards_with_baselines(data, baselines, out_dir: Path):

#     if "rollout/ep_rew_mean" not in data:
#         print("Missing rollout/ep_rew_mean")
#         return

#     s = data["rollout/ep_rew_mean"]
#     steps, rewards = s["steps"], s["values"]

#     plt.figure(figsize=(11, 5))

#     plt.plot(steps, rewards, alpha=0.3, label="PPO raw")

#     w = max(10, len(rewards)//20)
#     ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
#     plt.plot(steps[-len(ma):], ma, linewidth=2, label="PPO MA")

#     # ---------------- baselines ----------------
#     if baselines:
#         for k, v in baselines.items():
#             mean, std = extract_baseline_stats(v)
#             if mean is None:
#                 continue

#             plt.axhline(mean, linestyle="--", label=f"{k} mean")
#             if std is not None:
#                 plt.fill_between(
#                     [steps[0], steps[-1]],
#                     mean - std,
#                     mean + std,
#                     alpha=0.1
#                 )

#     plt.xlabel("Timesteps")
#     plt.ylabel("Reward")
#     plt.title("Training Reward vs Baselines")
#     plt.grid(alpha=0.3)
#     plt.legend()

#     out = out_dir / "reward_with_baselines.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# def plot_tb_diagnostics(data, out_dir: Path):
#     """2x3 grid covering everything train_ppo.py's log_interval prints:
#     value_loss, entropy_loss, approx_kl, clip_fraction, total loss,
#     policy_gradient_loss. learning_rate/ep_len_mean get their own plots
#     below since they're on different scales."""
#     keys = [
#         "train/value_loss",
#         "train/entropy_loss",
#         "train/approx_kl",
#         "train/clip_fraction",
#         "train/loss",
#         "train/policy_gradient_loss",
#     ]

#     fig, axes = plt.subplots(2, 3, figsize=(17, 8))

#     for ax, key in zip(axes.flatten(), keys):
#         if key not in data:
#             ax.set_title(f"Missing {key}")
#             ax.axis("off")
#             continue

#         s = data[key]
#         ax.plot(s["steps"], s["values"])
#         ax.set_title(key)
#         ax.set_xlabel("Timesteps")
#         ax.grid(alpha=0.3)

#     plt.tight_layout()
#     out = out_dir / "tb_diagnostics.png"
#     plt.savefig(out, dpi=150)
#     plt.close()
#     print(f"✓ {out}")

# def plot_policy_entropy(data, out_dir: Path):
#     if "policy/entropy" not in data:
#         print("⚠️ Missing policy/entropy")
#         return

#     s = data["policy/entropy"]

#     plt.figure(figsize=(10,4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Policy Entropy")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Entropy")
#     plt.grid(alpha=0.3)

#     out = out_dir / "policy_entropy.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

# def plot_action_probabilities(data, out_dir: Path):

#     plt.figure(figsize=(10,5))

#     colors = ["C0","C1","C2","C3"]

#     for action in range(4):

#         curves = []

#         for robot in range(6):

#             key = f"policy/robot_{robot}/prob_{action}"

#             if key in data:
#                 curves.append(data[key]["values"])

#         if len(curves) == 0:
#             continue

#         mean_curve = np.mean(np.stack(curves), axis=0)

#         steps = data[f"policy/robot_0/prob_{action}"]["steps"]

#         plt.plot(
#             steps,
#             mean_curve,
#             label=f"Action {action}",
#             color=colors[action],
#         )

#     plt.legend()
#     plt.grid(alpha=.3)
#     plt.xlabel("Timesteps")
#     plt.ylabel("Probability")
#     plt.title("Mean Policy Action Probabilities")

#     out = out_dir / "policy_action_probs.png"
#     plt.savefig(out,dpi=150,bbox_inches="tight")
#     plt.close()

# def plot_logits(data, out_dir: Path):

#     plt.figure(figsize=(10,5))

#     colors=["C0","C1","C2","C3"]

#     for action in range(4):

#         curves=[]

#         for robot in range(6):

#             key=f"policy/robot_{robot}/logit_{action}"

#             if key in data:
#                 curves.append(data[key]["values"])

#         if len(curves)==0:
#             continue

#         mean=np.mean(np.stack(curves),axis=0)

#         steps=data[f"policy/robot_0/logit_{action}"]["steps"]

#         plt.plot(
#             steps,
#             mean,
#             color=colors[action],
#             label=f"Action {action}"
#         )

#     plt.legend()
#     plt.grid(alpha=.3)
#     plt.xlabel("Timesteps")
#     plt.ylabel("Logit")
#     plt.title("Mean Policy Logits")

#     out=out_dir/"policy_logits.png"
#     plt.savefig(out,dpi=150,bbox_inches="tight")
#     plt.close()

# def plot_logit_spread(data, out_dir: Path):

#     robot_logits = []

#     for robot in range(6):

#         keys = [
#             f"policy/robot_{robot}/logit_{a}"
#             for a in range(4)
#         ]

#         if not all(k in data for k in keys):
#             continue

#         vals = np.stack(
#             [data[k]["values"] for k in keys],
#             axis=1,
#         )

#         spread = vals.max(axis=1) - vals.min(axis=1)

#         robot_logits.append(spread)

#     if len(robot_logits)==0:
#         return

#     spread=np.mean(np.stack(robot_logits),axis=0)

#     steps=data["policy/robot_0/logit_0"]["steps"]

#     plt.figure(figsize=(10,4))
#     plt.plot(steps,spread)

#     plt.title("Average Logit Spread")
#     plt.xlabel("Timesteps")
#     plt.ylabel("max(logit)-min(logit)")
#     plt.grid(alpha=.3)

#     out=out_dir/"logit_spread.png"
#     plt.savefig(out,dpi=150,bbox_inches="tight")
#     plt.close()
# def plot_tb_learning_rate(data, out_dir: Path):
#     if "train/learning_rate" not in data:
#         print("⚠️ Missing train/learning_rate")
#         return

#     s = data["train/learning_rate"]
#     plt.figure(figsize=(10, 4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Learning Rate")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Learning Rate")
#     plt.grid(alpha=0.3)

#     out = out_dir / "learning_rate.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✓ {out}")


# def plot_tb_episode_length(data, out_dir: Path):
#     if "rollout/ep_len_mean" not in data:
#         print("⚠️ Missing rollout/ep_len_mean")
#         return

#     s = data["rollout/ep_len_mean"]
#     plt.figure(figsize=(10, 4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Episode Length (mean)")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Steps per episode")
#     plt.grid(alpha=0.3)

#     out = out_dir / "episode_length.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✓ {out}")


# def plot_explained_variance(data, out_dir: Path):

#     if "train/explained_variance" not in data:
#         print("⚠️ Missing explained_variance")
#         return

#     s = data["train/explained_variance"]

#     plt.figure(figsize=(10, 4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Explained Variance (Critic Quality)")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Explained Variance")
#     plt.ylim(-1, 1)
#     plt.grid(alpha=0.3)

#     out = out_dir / "explained_variance.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# # =========================================================
# # Monitor
# # =========================================================

# def plot_monitor(monitor_path: Path, out_dir: Path):
#     try:
#         data = np.genfromtxt(
#             monitor_path,
#             delimiter=",",
#             skip_header=1,
#             names=True,
#             dtype=None,
#             encoding="utf-8"
#         )

#         rewards = data["r"]
#         lengths = data["l"]

#         fig, axes = plt.subplots(2, 2, figsize=(12, 9))

#         axes[0, 0].plot(rewards)
#         axes[0, 0].set_title("Episode Reward")

#         w = max(10, len(rewards)//20)
#         axes[0, 1].plot(moving_average(rewards, w))
#         axes[0, 1].set_title("Reward MA")

#         axes[1, 0].plot(lengths)
#         axes[1, 0].set_title("Episode Length")

#         axes[1, 1].hist(rewards, bins=30)
#         axes[1, 1].set_title("Reward Distribution")

#         for ax in axes.flatten():
#             ax.grid(alpha=0.3)

#         plt.tight_layout()
#         out = out_dir / "monitor.png"
#         plt.savefig(out, dpi=150)
#         plt.close()

#         print(f"✓ {out}")

#     except Exception as e:
#         print(f"⚠️ monitor error: {e}")


# # =========================================================
# # Logits (SAFE VERSION)
# # =========================================================

# def plot_logits(data, out_dir: Path):

#     keys = ["train/max_logit", "train/mean_logit", "train/logit_gap"]

#     plt.figure(figsize=(10, 5))

#     found = False
#     for k in keys:
#         if k in data:
#             s = data[k]
#             plt.plot(s["steps"], s["values"], label=k)
#             found = True

#     if not found:
#         print("⚠️ No logit stats found")
#         return

#     plt.title("Logit Statistics")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Value")
#     plt.grid(alpha=0.3)
#     plt.legend()

#     out = out_dir / "logits.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# # =========================================================
# # Baselines
# # =========================================================

# def plot_baselines(base_dir: Path, out_dir: Path):

#     file = base_dir / "baseline_results_all.json"

#     if not file.exists():
#         print("⚠️ No baseline file found")
#         return

#     data = json.loads(file.read_text())

#     policies = ["greedy", "random", "unique"]

#     names, means, stds = [], [], []

#     for p in policies:
#         if p not in data:
#             continue

#         mean, std = extract_baseline_stats(data[p])

#         if mean is None:
#             continue

#         names.append(p)
#         means.append(mean)
#         stds.append(std or 0.0)

#     if not names:
#         return

#     x = np.arange(len(names))

#     plt.figure(figsize=(8, 5))
#     plt.bar(x, means, yerr=stds, capsize=5)
#     plt.xticks(x, names)
#     plt.title("Baseline Comparison")
#     plt.grid(axis="y", alpha=0.3)

#     out = out_dir / "baselines.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# # =========================================================
# # Multi-seed comparison
# # =========================================================

# def plot_multi_seed_rewards(run_root: Path, out_dir: Path):
#     """Overlay rollout/ep_rew_mean across every seed_* trained in this sweep
#     (runs/run_{id}/seed_*/), plus a mean +/- std band across seeds."""
#     seed_dirs = sorted(run_root.glob("seed_*"))
#     if not seed_dirs:
#         print("⚠️ No seed_* directories found under", run_root)
#         return

#     curves = {}
#     for sd in seed_dirs:
#         tb_data = load_tensorboard(sd / "tensorboard")
#         print('***********')
#         print('tb_data****',tb_data)
#         if not tb_data or "rollout/ep_rew_mean" not in tb_data:
#             continue
#         curves[sd.name] = tb_data["rollout/ep_rew_mean"]

#     if not curves:
#         print("⚠️ No reward curves found across seeds")
#         return

#     plt.figure(figsize=(11, 5))
#     for name, s in curves.items():
#         plt.plot(s["steps"], s["values"], alpha=0.4, label=name)

#     # mean +/- std band, aligned by shortest common length
#     min_len = min(len(s["values"]) for s in curves.values())
#     if min_len > 1:
#         stacked = np.stack([s["values"][:min_len] for s in curves.values()])
#         ref_steps = next(iter(curves.values()))["steps"][:min_len]
#         mean = stacked.mean(axis=0)
#         std = stacked.std(axis=0)
#         plt.plot(ref_steps, mean, color="black", linewidth=2.5, label="mean across seeds")
#         plt.fill_between(ref_steps, mean - std, mean + std, color="black", alpha=0.15)

#     plt.xlabel("Timesteps")
#     plt.ylabel("Episode reward (mean)")
#     plt.title(f"Reward across {len(curves)} seeds — {run_root.name}")
#     plt.grid(alpha=0.3)
#     plt.legend(fontsize=8)

#     out = out_dir / "multi_seed_rewards.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✓ {out}")


# # =========================================================
# # MAIN
# # =========================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run-id", type=str, default=None,
#                          help="Sweep to plot, e.g. '20260712_143000' (the runs/run_{id}/ "
#                              "folder name minus the 'run_' prefix). Defaults to the most "
#                              "recently modified run_* folder under --runs-root.")
#     parser.add_argument("--seed", type=int, default=42,
#                          help="Which seed within the run to plot single-run diagnostics for.")
#     parser.add_argument("--run-dir", type=str, default=None,
#                          help="Explicit override: full path to a seed folder, "
#                               "e.g. runs/run_20260712_143000/seed_42. "
#                               "Takes precedence over --run-id/--seed.")
#     parser.add_argument("--base-dir", type=str, default="baseline_results")
#     parser.add_argument("--runs-root", type=str, default="runs",
#                          help="Root containing run_* sweep directories.")
#     parser.add_argument("--multi-seed", action="store_true",
#                          help="Also produce a cross-seed reward comparison plot "
#                               "(runs/run_{id}/multi_seed_plots/multi_seed_rewards.png) "
#                               "by scanning every seed_* dir inside this sweep.")
#     args = parser.parse_args()

#     base_dir = Path(args.base_dir)
#     runs_root = Path(args.runs_root)

#     if args.run_dir:
#         run_dir = Path(args.run_dir)
#         run_root = run_dir.parent
#     else:
#         run_root = Path("runs") / f"run_{args.run_id}" if args.run_id else latest_run_id(runs_root)
#         run_dir = seed_dir_in_run(run_root, args.seed)

#     out_dir = run_dir / "plots"
#     out_dir.mkdir(exist_ok=True, parents=True)

#     # TensorBoard
#     tb_data = load_tensorboard(run_dir / "tensorboard")

#     baseline_file = base_dir / "baseline_results_all.json"
#     baselines = json.loads(baseline_file.read_text()) if baseline_file.exists() else None

#     if tb_data:
#         plot_tb_rewards_with_baselines(tb_data, baselines, out_dir)
#         plot_tb_diagnostics(tb_data, out_dir)
#         plot_tb_learning_rate(tb_data, out_dir)
#         plot_tb_episode_length(tb_data, out_dir)
#         plot_explained_variance(tb_data, out_dir)
#         plot_logits(tb_data, out_dir)
#         plot_explained_variance(tb_data, out_dir)
#         plot_policy_entropy(tb_data, out_dir)
#         plot_action_probabilities(tb_data, out_dir)
#         plot_logits(tb_data, out_dir)
#         plot_logit_spread(tb_data, out_dir)
#     else:
#         print("⚠️ Skipping TensorBoard-derived plots (no data)")

#     monitor_path = run_dir / "logs" / "monitor.csv"
#     if monitor_path.exists():
#         plot_monitor(monitor_path, out_dir)
#     else:
#         print(f"⚠️ No monitor.csv at {monitor_path}")

#     # Baselines
#     plot_baselines(base_dir, out_dir)

#     if args.multi_seed:
#         multi_out = run_root / "multi_seed_plots"
#         multi_out.mkdir(exist_ok=True, parents=True)
#         plot_multi_seed_rewards(run_root, multi_out)

#     print("\n All plots saved to:", out_dir)


# if __name__ == "__main__":
#     main()



















# #!/usr/bin/env python3
# """
# Unified plotting script for GNN-PPO runs.

# Features:
# - Auto-detect latest run if not provided
# - Loads TensorBoard + monitor.csv
# - Baselines from external folder
# - Saves all plots into run_dir/plots/
# - --multi-seed: overlays reward curves (mean +/- std band) across every
#   seed's latest run, for runs launched with train_ppo.py's multi-seed loop
# """

# import matplotlib
# matplotlib.use("Agg")  # headless: must be set before importing pyplot.
# # Without this, matplotlib can pick a Qt-based interactive backend, and if
# # opencv-python (cv2) is installed it ships its own bundled Qt plugins that
# # shadow the system ones -> "Could not load the Qt platform plugin xcb" ->
# # hard crash (SIGABRT). This script only ever writes PNG files, so headless
# # Agg is correct here regardless of what else is installed in the venv.

# import argparse
# from pathlib import Path
# import json

# import numpy as np
# import matplotlib.pyplot as plt

# from tensorboard.backend.event_processing import event_accumulator


# # =========================================================
# # Utils
# # =========================================================

# def latest_run_id(runs_root: Path) -> Path:
#     """Return the most recently modified runs/run_{id}/ folder."""
#     run_dirs = sorted(runs_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
#     if not run_dirs:
#         raise FileNotFoundError(f"No run_* directories found in {runs_root}")
#     return run_dirs[-1]


# def seed_dir_in_run(run_root: Path, seed: int) -> Path:
#     """runs/run_{id}/seed_{seed}/ — raises with a helpful listing if that
#     seed wasn't part of this particular sweep."""
#     sd = run_root / f"seed_{seed}"
#     if not sd.exists():
#         available = sorted(p.name for p in run_root.glob("seed_*"))
#         raise FileNotFoundError(
#             f"No seed_{seed} under {run_root}. Available: {available}"
#         )
#     return sd


# def moving_average(x, w):
#     if len(x) < w:
#         return x
#     return np.convolve(x, np.ones(w) / w, mode="valid")


# def extract_baseline_stats(v):
#     """Robust baseline parser (handles multiple formats)."""
#     if not isinstance(v, dict):
#         return None, None

#     if "stats" in v:
#         s = v["stats"]
#         return s.get("reward_mean"), s.get("reward_std")

#     return v.get("reward_mean"), v.get("reward_std")


# # =========================================================
# # TensorBoard
# # =========================================================

# def load_tensorboard(tb_dir: Path):
#     runs = list(tb_dir.glob("PPO_*"))
#     if not runs:
#         print("⚠️ No TensorBoard runs found")
#         return None

#     latest = max(runs, key=lambda p: p.stat().st_mtime)
#     print(f"📈 Loading TensorBoard: {latest}")

#     ea = event_accumulator.EventAccumulator(str(latest))
#     ea.Reload()

#     data = {}
#     for tag in ea.Tags()["scalars"]:
#         events = ea.Scalars(tag)
#         data[tag] = {
#             "steps": np.array([e.step for e in events]),
#             "values": np.array([e.value for e in events]),
#         }

#     return data


# def plot_tb_rewards_with_baselines(data, baselines, out_dir: Path):

#     if "rollout/ep_rew_mean" not in data:
#         print("Missing rollout/ep_rew_mean")
#         return

#     s = data["rollout/ep_rew_mean"]
#     steps, rewards = s["steps"], s["values"]

#     plt.figure(figsize=(11, 5))

#     plt.plot(steps, rewards, alpha=0.3, label="PPO raw")

#     w = max(10, len(rewards)//20)
#     ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
#     plt.plot(steps[-len(ma):], ma, linewidth=2, label="PPO MA")

#     # ---------------- baselines ----------------
#     if baselines:
#         for k, v in baselines.items():
#             mean, std = extract_baseline_stats(v)
#             if mean is None:
#                 continue

#             plt.axhline(mean, linestyle="--", label=f"{k} mean")
#             if std is not None:
#                 plt.fill_between(
#                     [steps[0], steps[-1]],
#                     mean - std,
#                     mean + std,
#                     alpha=0.1
#                 )

#     plt.xlabel("Timesteps")
#     plt.ylabel("Reward")
#     plt.title("Training Reward vs Baselines")
#     plt.grid(alpha=0.3)
#     plt.legend()

#     out = out_dir / "reward_with_baselines.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# def plot_tb_diagnostics(data, out_dir: Path):
#     """2x3 grid covering everything train_ppo.py's log_interval prints:
#     value_loss, entropy_loss, approx_kl, clip_fraction, total loss,
#     policy_gradient_loss. learning_rate/ep_len_mean get their own plots
#     below since they're on different scales."""
#     keys = [
#         "train/value_loss",
#         "train/entropy_loss",
#         "train/approx_kl",
#         "train/clip_fraction",
#         "train/loss",
#         "train/policy_gradient_loss",
#     ]

#     fig, axes = plt.subplots(2, 3, figsize=(17, 8))

#     for ax, key in zip(axes.flatten(), keys):
#         if key not in data:
#             ax.set_title(f"Missing {key}")
#             ax.axis("off")
#             continue

#         s = data[key]
#         ax.plot(s["steps"], s["values"])
#         ax.set_title(key)
#         ax.set_xlabel("Timesteps")
#         ax.grid(alpha=0.3)

#     plt.tight_layout()
#     out = out_dir / "tb_diagnostics.png"
#     plt.savefig(out, dpi=150)
#     plt.close()
#     print(f"✓ {out}")

# def plot_policy_entropy(data, out_dir: Path):
#     if "policy/entropy" not in data:
#         print("⚠️ Missing policy/entropy")
#         return

#     s = data["policy/entropy"]

#     plt.figure(figsize=(10,4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Policy Entropy")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Entropy")
#     plt.grid(alpha=0.3)

#     out = out_dir / "policy_entropy.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

# def plot_action_probabilities(data, out_dir: Path):

#     plt.figure(figsize=(10,5))

#     colors = ["C0","C1","C2","C3"]

#     for action in range(4):

#         curves = []

#         for robot in range(6):

#             key = f"policy/robot_{robot}/prob_{action}"

#             if key in data:
#                 curves.append(data[key]["values"])

#         if len(curves) == 0:
#             continue

#         mean_curve = np.mean(np.stack(curves), axis=0)

#         steps = data[f"policy/robot_0/prob_{action}"]["steps"]

#         plt.plot(
#             steps,
#             mean_curve,
#             label=f"Action {action}",
#             color=colors[action],
#         )

#     plt.legend()
#     plt.grid(alpha=.3)
#     plt.xlabel("Timesteps")
#     plt.ylabel("Probability")
#     plt.title("Mean Policy Action Probabilities")

#     out = out_dir / "policy_action_probs.png"
#     plt.savefig(out,dpi=150,bbox_inches="tight")
#     plt.close()

# def _dead_plot_logits_removed(data, out_dir: Path):
#     # Removed: consumed policy/robot_{r}/logit_{a} TB tags from a buggy,
#     # per-robot-per-action logging block in train_ppo.py that has since been
#     # replaced with the reference-schema (train/mean_best_cand_logit etc.)
#     # and per-rank (train/rank_{i}_logit) tags — see plot_logits() and
#     # plot_logit_ranks() below.
#     pass

# def plot_logit_spread(data, out_dir: Path):

#     robot_logits = []

#     for robot in range(6):

#         keys = [
#             f"policy/robot_{robot}/logit_{a}"
#             for a in range(4)
#         ]

#         if not all(k in data for k in keys):
#             continue

#         vals = np.stack(
#             [data[k]["values"] for k in keys],
#             axis=1,
#         )

#         spread = vals.max(axis=1) - vals.min(axis=1)

#         robot_logits.append(spread)

#     if len(robot_logits)==0:
#         return

#     spread=np.mean(np.stack(robot_logits),axis=0)

#     steps=data["policy/robot_0/logit_0"]["steps"]

#     plt.figure(figsize=(10,4))
#     plt.plot(steps,spread)

#     plt.title("Average Logit Spread")
#     plt.xlabel("Timesteps")
#     plt.ylabel("max(logit)-min(logit)")
#     plt.grid(alpha=.3)

#     out=out_dir/"logit_spread.png"
#     plt.savefig(out,dpi=150,bbox_inches="tight")
#     plt.close()
# def plot_tb_learning_rate(data, out_dir: Path):
#     if "train/learning_rate" not in data:
#         print("⚠️ Missing train/learning_rate")
#         return

#     s = data["train/learning_rate"]
#     plt.figure(figsize=(10, 4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Learning Rate")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Learning Rate")
#     plt.grid(alpha=0.3)

#     out = out_dir / "learning_rate.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✓ {out}")


# def plot_tb_episode_length(data, out_dir: Path):
#     if "rollout/ep_len_mean" not in data:
#         print("⚠️ Missing rollout/ep_len_mean")
#         return

#     s = data["rollout/ep_len_mean"]
#     plt.figure(figsize=(10, 4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Episode Length (mean)")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Steps per episode")
#     plt.grid(alpha=0.3)

#     out = out_dir / "episode_length.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✓ {out}")


# def plot_explained_variance(data, out_dir: Path):

#     if "train/explained_variance" not in data:
#         print("⚠️ Missing explained_variance")
#         return

#     s = data["train/explained_variance"]

#     plt.figure(figsize=(10, 4))
#     plt.plot(s["steps"], s["values"])
#     plt.title("Explained Variance (Critic Quality)")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Explained Variance")
#     plt.ylim(-1, 1)
#     plt.grid(alpha=0.3)

#     out = out_dir / "explained_variance.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# # =========================================================
# # Monitor
# # =========================================================

# def plot_monitor(monitor_path: Path, out_dir: Path):
#     try:
#         data = np.genfromtxt(
#             monitor_path,
#             delimiter=",",
#             skip_header=1,
#             names=True,
#             dtype=None,
#             encoding="utf-8"
#         )

#         rewards = data["r"]
#         lengths = data["l"]

#         fig, axes = plt.subplots(2, 2, figsize=(12, 9))

#         axes[0, 0].plot(rewards)
#         axes[0, 0].set_title("Episode Reward")

#         w = max(10, len(rewards)//20)
#         axes[0, 1].plot(moving_average(rewards, w))
#         axes[0, 1].set_title("Reward MA")

#         axes[1, 0].plot(lengths)
#         axes[1, 0].set_title("Episode Length")

#         axes[1, 1].hist(rewards, bins=30)
#         axes[1, 1].set_title("Reward Distribution")

#         for ax in axes.flatten():
#             ax.grid(alpha=0.3)

#         plt.tight_layout()
#         out = out_dir / "monitor.png"
#         plt.savefig(out, dpi=150)
#         plt.close()

#         print(f"✓ {out}")

#     except Exception as e:
#         print(f"⚠️ monitor error: {e}")


# # =========================================================
# # Logits (SAFE VERSION)
# # =========================================================

# def plot_logits(data, out_dir: Path):

#     keys = ["train/max_logit", "train/mean_logit", "train/logit_gap"]

#     plt.figure(figsize=(10, 5))

#     found = False
#     for k in keys:
#         if k in data:
#             s = data[k]
#             plt.plot(s["steps"], s["values"], label=k)
#             found = True

#     if not found:
#         print("⚠️ No logit stats found")
#         return

#     plt.title("Logit Statistics")
#     plt.xlabel("Timesteps")
#     plt.ylabel("Value")
#     plt.grid(alpha=0.3)
#     plt.legend()

#     out = out_dir / "logits.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# # =========================================================
# # Baselines
# # =========================================================

# def plot_baselines(base_dir: Path, out_dir: Path):

#     file = base_dir / "baseline_results_all.json"

#     if not file.exists():
#         print("⚠️ No baseline file found")
#         return

#     data = json.loads(file.read_text())

#     policies = ["greedy", "random", "unique"]

#     names, means, stds = [], [], []

#     for p in policies:
#         if p not in data:
#             continue

#         mean, std = extract_baseline_stats(data[p])

#         if mean is None:
#             continue

#         names.append(p)
#         means.append(mean)
#         stds.append(std or 0.0)

#     if not names:
#         return

#     x = np.arange(len(names))

#     plt.figure(figsize=(8, 5))
#     plt.bar(x, means, yerr=stds, capsize=5)
#     plt.xticks(x, names)
#     plt.title("Baseline Comparison")
#     plt.grid(axis="y", alpha=0.3)

#     out = out_dir / "baselines.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"✓ {out}")


# # =========================================================
# # Multi-seed comparison
# # =========================================================

# def plot_multi_seed_rewards(run_root: Path, out_dir: Path):
#     """Overlay rollout/ep_rew_mean across every seed_* trained in this sweep
#     (runs/run_{id}/seed_*/), plus a mean +/- std band across seeds."""
#     seed_dirs = sorted(run_root.glob("seed_*"))
#     if not seed_dirs:
#         print("⚠️ No seed_* directories found under", run_root)
#         return

#     curves = {}
#     for sd in seed_dirs:
#         tb_data = load_tensorboard(sd / "tensorboard")
#         print('***********')
#         print('tb_data****',tb_data)
#         if not tb_data or "rollout/ep_rew_mean" not in tb_data:
#             continue
#         curves[sd.name] = tb_data["rollout/ep_rew_mean"]

#     if not curves:
#         print("⚠️ No reward curves found across seeds")
#         return

#     plt.figure(figsize=(11, 5))
#     for name, s in curves.items():
#         plt.plot(s["steps"], s["values"], alpha=0.4, label=name)

#     # mean +/- std band, aligned by shortest common length
#     min_len = min(len(s["values"]) for s in curves.values())
#     if min_len > 1:
#         stacked = np.stack([s["values"][:min_len] for s in curves.values()])
#         ref_steps = next(iter(curves.values()))["steps"][:min_len]
#         mean = stacked.mean(axis=0)
#         std = stacked.std(axis=0)
#         plt.plot(ref_steps, mean, color="black", linewidth=2.5, label="mean across seeds")
#         plt.fill_between(ref_steps, mean - std, mean + std, color="black", alpha=0.15)

#     plt.xlabel("Timesteps")
#     plt.ylabel("Episode reward (mean)")
#     plt.title(f"Reward across {len(curves)} seeds — {run_root.name}")
#     plt.grid(alpha=0.3)
#     plt.legend(fontsize=8)

#     out = out_dir / "multi_seed_rewards.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"✓ {out}")


# # =========================================================
# # MAIN
# # =========================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--run-id", type=str, default=None,
#                          help="Sweep to plot, e.g. '20260712_143000' (the runs/run_{id}/ "
#                              "folder name minus the 'run_' prefix). Defaults to the most "
#                              "recently modified run_* folder under --runs-root.")
#     parser.add_argument("--seed", type=int, default=42,
#                          help="Which seed within the run to plot single-run diagnostics for.")
#     parser.add_argument("--run-dir", type=str, default=None,
#                          help="Explicit override: full path to a seed folder, "
#                               "e.g. runs/run_20260712_143000/seed_42. "
#                               "Takes precedence over --run-id/--seed.")
#     parser.add_argument("--base-dir", type=str, default="baseline_results")
#     parser.add_argument("--runs-root", type=str, default="runs",
#                          help="Root containing run_* sweep directories.")
#     parser.add_argument("--multi-seed", action="store_true",
#                          help="Also produce a cross-seed reward comparison plot "
#                               "(runs/run_{id}/multi_seed_plots/multi_seed_rewards.png) "
#                               "by scanning every seed_* dir inside this sweep.")
#     args = parser.parse_args()

#     base_dir = Path(args.base_dir)
#     runs_root = Path(args.runs_root)

#     if args.run_dir:
#         run_dir = Path(args.run_dir)
#         run_root = run_dir.parent
#     else:
#         run_root = Path("runs") / f"run_{args.run_id}" if args.run_id else latest_run_id(runs_root)
#         run_dir = seed_dir_in_run(run_root, args.seed)

#     out_dir = run_dir / "plots"
#     out_dir.mkdir(exist_ok=True, parents=True)

#     # TensorBoard
#     tb_data = load_tensorboard(run_dir / "tensorboard")

#     baseline_file = base_dir / "baseline_results_all.json"
#     baselines = json.loads(baseline_file.read_text()) if baseline_file.exists() else None

#     if tb_data:
#         plot_tb_rewards_with_baselines(tb_data, baselines, out_dir)
#         plot_tb_diagnostics(tb_data, out_dir)
#         plot_tb_learning_rate(tb_data, out_dir)
#         plot_tb_episode_length(tb_data, out_dir)
#         plot_explained_variance(tb_data, out_dir)
#         plot_logits(tb_data, out_dir)
#         plot_explained_variance(tb_data, out_dir)
#         plot_policy_entropy(tb_data, out_dir)
#         plot_action_probabilities(tb_data, out_dir)
#         plot_logits(tb_data, out_dir)
#         plot_logit_spread(tb_data, out_dir)
#     else:
#         print("⚠️ Skipping TensorBoard-derived plots (no data)")

#     monitor_path = run_dir / "logs" / "monitor.csv"
#     if monitor_path.exists():
#         plot_monitor(monitor_path, out_dir)
#     else:
#         print(f"⚠️ No monitor.csv at {monitor_path}")

#     # Baselines
#     plot_baselines(base_dir, out_dir)

#     if args.multi_seed:
#         multi_out = run_root / "multi_seed_plots"
#         multi_out.mkdir(exist_ok=True, parents=True)
#         plot_multi_seed_rewards(run_root, multi_out)

#     print("\n All plots saved to:", out_dir)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Unified plotting script for GNN-PPO runs.

Features:
- Auto-detect latest run if not provided
- Loads TensorBoard + monitor.csv
- Baselines from external folder
- Saves all plots into run_dir/plots/
- --multi-seed: overlays reward curves (mean +/- std band) across every
  seed's latest run, for runs launched with train_ppo.py's multi-seed loop
"""

import matplotlib
matplotlib.use("Agg")  # headless: must be set before importing pyplot.
# Without this, matplotlib can pick a Qt-based interactive backend, and if
# opencv-python (cv2) is installed it ships its own bundled Qt plugins that
# shadow the system ones -> "Could not load the Qt platform plugin xcb" ->
# hard crash (SIGABRT). This script only ever writes PNG files, so headless
# Agg is correct here regardless of what else is installed in the venv.

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


# =========================================================
# Utils
# =========================================================

def latest_run_id(runs_root: Path) -> Path:
    """Return the most recently modified runs/run_{id}/ folder."""
    run_dirs = sorted(runs_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {runs_root}")
    return run_dirs[-1]


def seed_dir_in_run(run_root: Path, seed: int) -> Path:
    """runs/run_{id}/seed_{seed}/ — raises with a helpful listing if that
    seed wasn't part of this particular sweep."""
    sd = run_root / f"seed_{seed}"
    if not sd.exists():
        available = sorted(p.name for p in run_root.glob("seed_*"))
        raise FileNotFoundError(
            f"No seed_{seed} under {run_root}. Available: {available}"
        )
    return sd


def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def extract_baseline_stats(v):
    """Robust baseline parser (handles multiple formats)."""
    if not isinstance(v, dict):
        return None, None

    if "stats" in v:
        s = v["stats"]
        return s.get("reward_mean"), s.get("reward_std")

    return v.get("reward_mean"), v.get("reward_std")


# =========================================================
# TensorBoard
# =========================================================

def load_tensorboard(tb_dir: Path):
    runs = list(tb_dir.glob("PPO_*"))
    if not runs:
        print("⚠️ No TensorBoard runs found")
        return None

    latest = max(runs, key=lambda p: p.stat().st_mtime)
    print(f"Loading TensorBoard: {latest}")

    ea = event_accumulator.EventAccumulator(str(latest))
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": np.array([e.step for e in events]),
            "values": np.array([e.value for e in events]),
        }

    return data


def plot_tb_rewards_with_baselines(data, baselines, out_dir: Path):

    if "rollout/ep_rew_mean" not in data:
        print("Missing rollout/ep_rew_mean")
        return

    s = data["rollout/ep_rew_mean"]
    steps, rewards = s["steps"], s["values"]

    plt.figure(figsize=(11, 5))

    plt.plot(steps, rewards, alpha=0.3, label="PPO raw")

    w = max(10, len(rewards)//20)
    ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
    plt.plot(steps[-len(ma):], ma, linewidth=2, label="PPO MA")

    # ---------------- baselines ----------------
    if baselines:
        for k, v in baselines.items():
            mean, std = extract_baseline_stats(v)
            if mean is None:
                continue

            plt.axhline(mean, linestyle="--", label=f"{k} mean")
            if std is not None:
                plt.fill_between(
                    [steps[0], steps[-1]],
                    mean - std,
                    mean + std,
                    alpha=0.1
                )

    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Reward vs Baselines")
    plt.grid(alpha=0.3)
    plt.legend()

    out = out_dir / "reward_with_baselines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ {out}")


def plot_tb_diagnostics(data, out_dir: Path):
    """2x3 grid covering everything train_ppo.py's log_interval prints:
    value_loss, entropy_loss, approx_kl, clip_fraction, total loss,
    policy_gradient_loss. learning_rate/ep_len_mean get their own plots
    below since they're on different scales."""
    keys = [
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/loss",
        "train/policy_gradient_loss",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))

    for ax, key in zip(axes.flatten(), keys):
        if key not in data:
            ax.set_title(f"Missing {key}")
            ax.axis("off")
            continue

        s = data[key]
        ax.plot(s["steps"], s["values"])
        ax.set_title(key)
        ax.set_xlabel("Timesteps")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "tb_diagnostics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✓ {out}")

def plot_policy_entropy(data, out_dir: Path):
    if "policy/entropy" not in data:
        print("⚠️ Missing policy/entropy")
        return

    s = data["policy/entropy"]

    plt.figure(figsize=(10,4))
    plt.plot(s["steps"], s["values"])
    plt.title("Policy Entropy")
    plt.xlabel("Timesteps")
    plt.ylabel("Entropy")
    plt.grid(alpha=0.3)

    out = out_dir / "policy_entropy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

def plot_action_probabilities(data, out_dir: Path):

    plt.figure(figsize=(10,5))

    colors = ["C0","C1","C2","C3"]

    for action in range(4):

        curves = []

        for robot in range(6):

            key = f"policy/robot_{robot}/prob_{action}"

            if key in data:
                curves.append(data[key]["values"])

        if len(curves) == 0:
            continue

        mean_curve = np.mean(np.stack(curves), axis=0)

        steps = data[f"policy/robot_0/prob_{action}"]["steps"]

        plt.plot(
            steps,
            mean_curve,
            label=f"Action {action}",
            color=colors[action],
        )

    plt.legend()
    plt.grid(alpha=.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Probability")
    plt.title("Mean Policy Action Probabilities")

    out = out_dir / "policy_action_probs.png"
    plt.savefig(out,dpi=150,bbox_inches="tight")
    plt.close()

def _dead_plot_logits_removed(data, out_dir: Path):
    # Removed: consumed policy/robot_{r}/logit_{a} TB tags from a buggy,
    # per-robot-per-action logging block in train_ppo.py that has since been
    # replaced with the reference-schema (train/mean_best_cand_logit etc.)
    # and per-rank (train/rank_{i}_logit) tags — see plot_logits() and
    # plot_logit_ranks() below.
    pass

def plot_logit_spread(data, out_dir: Path):

    robot_logits = []

    for robot in range(6):

        keys = [
            f"policy/robot_{robot}/logit_{a}"
            for a in range(4)
        ]

        if not all(k in data for k in keys):
            continue

        vals = np.stack(
            [data[k]["values"] for k in keys],
            axis=1,
        )

        spread = vals.max(axis=1) - vals.min(axis=1)

        robot_logits.append(spread)

    if len(robot_logits)==0:
        return

    spread=np.mean(np.stack(robot_logits),axis=0)

    steps=data["policy/robot_0/logit_0"]["steps"]

    plt.figure(figsize=(10,4))
    plt.plot(steps,spread)

    plt.title("Average Logit Spread")
    plt.xlabel("Timesteps")
    plt.ylabel("max(logit)-min(logit)")
    plt.grid(alpha=.3)

    out=out_dir/"logit_spread.png"
    plt.savefig(out,dpi=150,bbox_inches="tight")
    plt.close()
def plot_tb_learning_rate(data, out_dir: Path):
    if "train/learning_rate" not in data:
        print("⚠️ Missing train/learning_rate")
        return

    s = data["train/learning_rate"]
    plt.figure(figsize=(10, 4))
    plt.plot(s["steps"], s["values"])
    plt.title("Learning Rate")
    plt.xlabel("Timesteps")
    plt.ylabel("Learning Rate")
    plt.grid(alpha=0.3)

    out = out_dir / "learning_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


def plot_tb_episode_length(data, out_dir: Path):
    if "rollout/ep_len_mean" not in data:
        print("⚠️ Missing rollout/ep_len_mean")
        return

    s = data["rollout/ep_len_mean"]
    plt.figure(figsize=(10, 4))
    plt.plot(s["steps"], s["values"])
    plt.title("Episode Length (mean)")
    plt.xlabel("Timesteps")
    plt.ylabel("Steps per episode")
    plt.grid(alpha=0.3)

    out = out_dir / "episode_length.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


def plot_explained_variance(data, out_dir: Path):

    if "train/explained_variance" not in data:
        print("⚠️ Missing explained_variance")
        return

    s = data["train/explained_variance"]

    plt.figure(figsize=(10, 4))
    plt.plot(s["steps"], s["values"])
    plt.title("Explained Variance (Critic Quality)")
    plt.xlabel("Timesteps")
    plt.ylabel("Explained Variance")
    plt.ylim(-1, 1)
    plt.grid(alpha=0.3)

    out = out_dir / "explained_variance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ {out}")


# =========================================================
# Monitor
# =========================================================

def plot_monitor(monitor_path: Path, out_dir: Path):
    try:
        data = np.genfromtxt(
            monitor_path,
            delimiter=",",
            skip_header=1,
            names=True,
            dtype=None,
            encoding="utf-8"
        )

        rewards = data["r"]
        lengths = data["l"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(rewards)
        axes[0, 0].set_title("Episode Reward")

        w = max(10, len(rewards)//20)
        axes[0, 1].plot(moving_average(rewards, w))
        axes[0, 1].set_title("Reward MA")

        axes[1, 0].plot(lengths)
        axes[1, 0].set_title("Episode Length")

        axes[1, 1].hist(rewards, bins=30)
        axes[1, 1].set_title("Reward Distribution")

        for ax in axes.flatten():
            ax.grid(alpha=0.3)

        plt.tight_layout()
        out = out_dir / "monitor.png"
        plt.savefig(out, dpi=150)
        plt.close()

        print(f"✓ {out}")

    except Exception as e:
        print(f"⚠️ monitor error: {e}")


# =========================================================
# Logits (SAFE VERSION)
# =========================================================

def plot_logits(data, out_dir: Path):
    """Best-real-candidate vs noop, from TensorBoard (reference-schema tags
    logged once per episode by TrainingLogCallback)."""
    keys = ["train/mean_best_cand_logit", "train/mean_noop_logit", "train/logit_gap", "train/top1_top2_margin"]

    plt.figure(figsize=(10, 5))

    found = False
    for k in keys:
        if k in data:
            s = data[k]
            plt.plot(s["steps"], s["values"], label=k.replace("train/", ""))
            found = True

    if not found:
        print("⚠️ No logit stats found (train/mean_best_cand_logit etc.) — "
              "retrain with the updated TrainingLogCallback to populate these.")
        return

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Best Real Candidate vs Noop Logit")
    plt.xlabel("Timesteps")
    plt.ylabel("Logit")
    plt.grid(alpha=0.3)
    plt.legend()

    out = out_dir / "logits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ {out}")


def plot_logit_ranks(run_dir: Path, out_dir: Path):
    """'Logits of all actions' over training — every candidate rank
    (0=best, 1=2nd-best, ...) plus noop, read from the authoritative CSV
    (src/utils/logit_metrics_logger.py's training_logit_ranks.csv) rather
    than TensorBoard, since a rank only gets a TB scalar on episodes where
    at least one decision had that many candidates, while the CSV has one
    row per episode regardless with None for ranks that had no data that
    episode (cleanly droppable rather than sparsely missing from TB).

    This is the plot to use for picking noop_init: it shows where the real
    per-rank candidate logits actually sit over the course of training, so
    you can choose a noop_init that starts competitive with (not always
    below, or always dominating) where the best few real candidates
    naturally land — rather than guessing a value with no visibility into
    the rest of the distribution.
    """
    csv_path = run_dir / "logs" / "training_logit_ranks.csv"
    if not csv_path.exists():
        print(f"⚠️ No {csv_path} — retrain with the updated TrainingLogCallback to populate this.")
        return

    import csv as _csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"⚠️ {csv_path} is empty")
        return

    K_max = int(rows[0]["K_max"])
    episodes = np.array([int(r["episode"]) for r in rows])

    def parse_col(col):
        return np.array([float(r[col]) if r[col] not in ("", "None") else np.nan for r in rows])

    noop_vals = parse_col("noop_logit_mean")

    plt.figure(figsize=(11, 6))
    plt.plot(episodes, noop_vals, label="noop", color="black", linewidth=2.5, linestyle="--")

    cmap = plt.get_cmap("viridis", K_max)
    for i in range(K_max):
        vals = parse_col(f"rank_{i}_logit_mean")
        if np.all(np.isnan(vals)):
            continue
        label = "best real candidate" if i == 0 else f"rank {i} candidate"
        plt.plot(episodes, vals, label=label, color=cmap(i), alpha=0.85)

    plt.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    plt.title("Logits of All Actions (per candidate rank + noop)")
    plt.xlabel("Episode")
    plt.ylabel("Mean logit")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)

    out = out_dir / "logit_ranks_all_actions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")

    # Print a short summary at the point closest to the start of training —
    # this is the practical "what noop_init should I use" readout: it shows
    # where the real-candidate ranks actually sit early on, so noop_init can
    # be chosen to start in a sensible position relative to them instead of
    # guessing (e.g. roughly at or slightly below the early best-candidate
    # logit, so noop is competitive but not an automatic winner at init).
    n_head = max(1, len(rows) // 20)  # first ~5% of episodes
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        early_noop = np.nanmean(noop_vals[:n_head])
        early_ranks = [np.nanmean(parse_col(f"rank_{i}_logit_mean")[:n_head]) for i in range(K_max)]
    print(f"\n--- Early-training logit levels (first {n_head} episodes) — for choosing noop_init ---")
    print(f"noop_logit (current init)     : {early_noop:.4f}")
    for i, v in enumerate(early_ranks):
        label = "best real candidate" if i == 0 else f"rank {i} candidate"
        print(f"{label:<24}: {v:.4f}" if not np.isnan(v) else f"{label:<24}: (no data)")
    print(
        "A reasonable noop_init sits somewhere in this range rather than far below every\n"
        "rank (which starves noop of ever being chosen, risking the opposite collapse) or\n"
        "far above the best candidate (which recreates the always-noop plurality problem\n"
        "diagnosed earlier). If freeze_noop_logit=True, this value never moves, so getting\n"
        "it right here matters more than when it's trainable.\n"
    )


# =========================================================
# Baselines
# =========================================================

def plot_baselines(base_dir: Path, out_dir: Path):

    file = base_dir / "baseline_results_all.json"

    if not file.exists():
        print("⚠️ No baseline file found")
        return

    data = json.loads(file.read_text())

    policies = ["greedy", "random", "unique"]

    names, means, stds = [], [], []

    for p in policies:
        if p not in data:
            continue

        mean, std = extract_baseline_stats(data[p])

        if mean is None:
            continue

        names.append(p)
        means.append(mean)
        stds.append(std or 0.0)

    if not names:
        return

    x = np.arange(len(names))

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, names)
    plt.title("Baseline Comparison")
    plt.grid(axis="y", alpha=0.3)

    out = out_dir / "baselines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ {out}")


# =========================================================
# Multi-seed comparison
# =========================================================

def plot_multi_seed_rewards(run_root: Path, out_dir: Path):
    """Overlay rollout/ep_rew_mean across every seed_* trained in this sweep
    (runs/run_{id}/seed_*/), plus a mean +/- std band across seeds."""
    seed_dirs = sorted(run_root.glob("seed_*"))
    if not seed_dirs:
        print("⚠️ No seed_* directories found under", run_root)
        return

    curves = {}
    for sd in seed_dirs:
        tb_data = load_tensorboard(sd / "tensorboard")
        print('***********')
        print('tb_data****',tb_data)
        if not tb_data or "rollout/ep_rew_mean" not in tb_data:
            continue
        curves[sd.name] = tb_data["rollout/ep_rew_mean"]

    if not curves:
        print("⚠️ No reward curves found across seeds")
        return

    plt.figure(figsize=(11, 5))
    for name, s in curves.items():
        plt.plot(s["steps"], s["values"], alpha=0.4, label=name)

    # mean +/- std band, aligned by shortest common length
    min_len = min(len(s["values"]) for s in curves.values())
    if min_len > 1:
        stacked = np.stack([s["values"][:min_len] for s in curves.values()])
        ref_steps = next(iter(curves.values()))["steps"][:min_len]
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        plt.plot(ref_steps, mean, color="black", linewidth=2.5, label="mean across seeds")
        plt.fill_between(ref_steps, mean - std, mean + std, color="black", alpha=0.15)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (mean)")
    plt.title(f"Reward across {len(curves)} seeds — {run_root.name}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    out = out_dir / "multi_seed_rewards.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                         help="Sweep to plot, e.g. '20260712_143000' (the runs/run_{id}/ "
                             "folder name minus the 'run_' prefix). Defaults to the most "
                             "recently modified run_* folder under --runs-root.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Which seed within the run to plot single-run diagnostics for.")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Explicit override: full path to a seed folder, "
                              "e.g. runs/run_20260712_143000/seed_42. "
                              "Takes precedence over --run-id/--seed.")
    parser.add_argument("--base-dir", type=str, default="baseline_results")
    parser.add_argument("--runs-root", type=str, default="runs",
                         help="Root containing run_* sweep directories.")
    parser.add_argument("--multi-seed", action="store_true",
                         help="Also produce a cross-seed reward comparison plot "
                              "(runs/run_{id}/multi_seed_plots/multi_seed_rewards.png) "
                              "by scanning every seed_* dir inside this sweep.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    runs_root = Path(args.runs_root)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_root = run_dir.parent
    else:
        run_root = Path("runs") / f"run_{args.run_id}" if args.run_id else latest_run_id(runs_root)
        run_dir = seed_dir_in_run(run_root, args.seed)

    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    # TensorBoard
    tb_data = load_tensorboard(run_dir / "tensorboard")

    baseline_file = base_dir / "baseline_results_all.json"
    baselines = json.loads(baseline_file.read_text()) if baseline_file.exists() else None

    if tb_data:
        plot_tb_rewards_with_baselines(tb_data, baselines, out_dir)
        plot_tb_diagnostics(tb_data, out_dir)
        plot_tb_learning_rate(tb_data, out_dir)
        plot_tb_episode_length(tb_data, out_dir)
        plot_explained_variance(tb_data, out_dir)
        plot_logits(tb_data, out_dir)
        plot_policy_entropy(tb_data, out_dir)
        plot_action_probabilities(tb_data, out_dir)
        plot_logit_spread(tb_data, out_dir)
    else:
        print("⚠️ Skipping TensorBoard-derived plots (no data)")

    # "Logits of all actions" — reads the CSV directly (not TensorBoard),
    # see plot_logit_ranks() docstring for why, and prints the early-training
    # summary that's useful for picking noop_init.
    plot_logit_ranks(run_dir, out_dir)

    monitor_path = run_dir / "logs" / "monitor.csv"
    if monitor_path.exists():
        plot_monitor(monitor_path, out_dir)
    else:
        print(f"⚠️ No monitor.csv at {monitor_path}")

    # Baselines
    plot_baselines(base_dir, out_dir)

    if args.multi_seed:
        multi_out = run_root / "multi_seed_plots"
        multi_out.mkdir(exist_ok=True, parents=True)
        plot_multi_seed_rewards(run_root, multi_out)

    print("\n All plots saved to:", out_dir)


if __name__ == "__main__":
    main()