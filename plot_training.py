#!/usr/bin/env python3
"""
Unified plotting script for GNN-PPO runs.

Features:
- Auto-detect latest run if not provided
- Loads TensorBoard + monitor.csv
- Baselines from external folder
- Saves all plots into run_dir/plots/
"""

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


# =========================================================
# Utils
# =========================================================

def latest_run(seed_dir: Path) -> Path:
    runs = sorted(seed_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs found in {seed_dir}")
    return runs[-1]


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
    print(f"📈 Loading TensorBoard: {latest}")

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
    keys = [
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, key in zip(axes.flatten(), keys):
        if key not in data:
            ax.set_title(f"Missing {key}")
            ax.axis("off")
            continue

        s = data[key]
        ax.plot(s["steps"], s["values"])
        ax.set_title(key)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "tb_diagnostics.png"
    plt.savefig(out, dpi=150)
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

    keys = ["train/max_logit", "train/mean_logit", "train/logit_gap"]

    plt.figure(figsize=(10, 5))

    found = False
    for k in keys:
        if k in data:
            s = data[k]
            plt.plot(s["steps"], s["values"], label=k)
            found = True

    if not found:
        print("⚠️ No logit stats found")
        return

    plt.title("Logit Statistics")
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.legend()

    out = out_dir / "logits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ {out}")


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
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default="baseline_results")
    args = parser.parse_args()

    base = Path("runs") / f"seed_{args.seed}"
    base_dir = Path(args.base_dir)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run(base)

    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # TensorBoard
    tb_data = load_tensorboard(run_dir / "tensorboard")

    baseline_file = base_dir / "baseline_results_all.json"
    baselines = json.loads(baseline_file.read_text()) if baseline_file.exists() else None

    plot_tb_rewards_with_baselines(tb_data, baselines, out_dir)
    plot_tb_diagnostics(tb_data, out_dir)
    plot_explained_variance(tb_data, out_dir)
    plot_logits(tb_data, out_dir)

    # Baselines
    plot_baselines(base_dir, out_dir)

    print("\n All plots saved to:", out_dir)


if __name__ == "__main__":
    main()