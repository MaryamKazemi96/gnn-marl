"""
Comprehensive evaluation and training plots.

Reads:
  - eval_results_deterministic.json (from eval_ppo.py)
  - eval_results_stochastic.json (from eval_ppo.py)
  - baseline_results_all.json (from eval_baseline.py)
  - logs/monitor.csv (from training)

Generates plots:
  - Evaluation plots (rewards, completion, boxplots)
  - Training curves (rewards, losses, entropy)
  - Logit diagnostics (per-action logits, gaps)
  - Policy behavior (NOOP fraction, assignments)

Usage:
  python plot_evaluation.py
  python plot_evaluation.py --output-dir custom_plots --ma-window 10
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Helpers
# ============================================================================

def _load_json(p: Path) -> Any:
    """Load JSON file."""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_fig(fig, out_path: Path) -> None:
    """Save figure to file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f" {out_path.name}")


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average."""
    data = np.asarray(data, dtype=float)
    if data.size == 0 or window <= 1 or data.size < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad_len = data.size - ma.size
    return np.concatenate([data[:pad_len], ma])


def _load_monitor_csv(csv_file: Path) -> Dict[str, List[float]]:
    """Load monitor.csv from stable_baselines3."""
    data = {
        "timestep": [],
        "reward": [],
        "length": [],
    }
    
    if not csv_file.exists():
        return data
    
    try:
        with csv_file.open("r") as f:
            next(f) 
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data["timestep"].append(float(row.get("t", 0)))
                    data["reward"].append(float(row.get("r", 0)))
                    data["length"].append(float(row.get("l", 0)))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f" Error reading monitor.csv: {e}")
    
    return data


# ============================================================================
# Load Baseline Results
# ============================================================================

def load_baselines_all(baseline_file: Path) -> Dict[str, Dict]:
    """Load baseline_results_all.json."""
    if not baseline_file.exists():
        return {}
    
    try:
        data = _load_json(baseline_file)
    except Exception as e:
        print(f" Could not load baseline results: {e}")
        return {}
    
    out: Dict[str, Dict] = {}
    for pol in ["random", "greedy", "unique"]:
        if pol in data:
            pol_data = data[pol]
            out[pol.upper()] = {
                "rewards": pol_data.get("rewards", []),
                "completed": pol_data.get("completed", []),
                "obsolete": pol_data.get("obsolete", []),
            }
    
    return out


# ============================================================================
# Evaluation Plots
# ============================================================================

def plot_eval_rewards_per_episode(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
    ma_window: int = 5,
) -> None:
    """Plot per-episode rewards with moving average."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    # PPO results
    for label, color, data in [
        ("PPO Deterministic", "#2980b9", det_data),
        ("PPO Stochastic", "#27ae60", stoch_data),
    ]:
        if not data:
            continue
        
        rewards = np.asarray(data.get("rewards", []), dtype=float)
        if rewards.size == 0:
            continue
        
        x = np.arange(1, rewards.size + 1)
        ax.plot(x, rewards, alpha=0.15, color=color, linewidth=0.8)
        ax.plot(x, _moving_average(rewards, ma_window), lw=2.6, color=color,
                label=f"{label} MA({ma_window}) – mean {rewards.mean():.2f}")
        ax.axhline(rewards.mean(), color=color, lw=1.4, ls="--", alpha=0.55)

    # Baseline results
    baseline_order = ["RANDOM", "GREEDY", "UNIQUE"]
    baseline_colors = {"RANDOM": "#e74c3c", "GREEDY": "#f39c12", "UNIQUE": "#8c564b"}
    
    for name in baseline_order:
        if name not in baselines:
            continue
        
        rewards = np.asarray(baselines[name].get("rewards", []), dtype=float)
        if rewards.size == 0:
            continue
        
        x = np.arange(1, rewards.size + 1)
        c = baseline_colors.get(name, "#7f8c8d")
        ax.plot(x, rewards, alpha=0.10, color=c, linewidth=0.8)
        ax.plot(x, _moving_average(rewards, ma_window), lw=2.2, color=c,
                label=f"{name} MA({ma_window}) – mean {rewards.mean():.2f}")
        ax.axhline(rewards.mean(), color=c, lw=1.2, ls="--", alpha=0.45)

    ax.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Per-Episode Rewards (PPO Det/Stoch + Baselines)", 
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_eval_rewards_boxplot(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    """Plot reward distribution as boxplot."""
    ordered: Dict[str, np.ndarray] = {}
    
    if det_data:
        r = np.asarray(det_data.get("rewards", []), dtype=float)
        if r.size:
            ordered["PPO Det"] = r
    
    if stoch_data:
        r = np.asarray(stoch_data.get("rewards", []), dtype=float)
        if r.size:
            ordered["PPO Stoch"] = r
    
    for name, d in baselines.items():
        r = np.asarray(d.get("rewards", []), dtype=float)
        if r.size:
            ordered[name] = r

    if not ordered:
        return

    labels = list(ordered.keys())
    series = list(ordered.values())

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    
    bp = ax.boxplot(series, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2))
    
    colors_list = ["#2980b9", "#27ae60", "#e74c3c", "#f39c12", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors_list[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Reward Distribution", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_eval_completion_obsolete(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    """Plot completion and obsolescence rates."""
    methods: Dict[str, Dict] = {}
    
    if det_data:
        methods["PPO Det"] = det_data
    if stoch_data:
        methods["PPO Stoch"] = stoch_data
    
    methods.update(baselines)

    if not methods:
        return

    labels: List[str] = []
    comp_means: List[float] = []
    obs_means: List[float] = []

    for name, d in methods.items():
        c = np.asarray(d.get("completed", []), dtype=float)
        o = np.asarray(d.get("obsolete", []), dtype=float)
        labels.append(name)
        comp_means.append(float(c.mean()) if c.size else 0.0)
        obs_means.append(float(o.mean()) if o.size else 0.0)

    x = np.arange(len(labels))
    w = 0.35
    
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    
    ax.bar(x - w / 2, comp_means, width=w, label="Completed", color="#27ae60", alpha=0.8)
    ax.bar(x + w / 2, obs_means, width=w, label="Obsolete", color="#c0392b", alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count per Episode", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Task Completion & Obsolescence", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_eval_stats_summary(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    """Plot summary statistics table."""
    methods: Dict[str, Dict[str, float]] = {}
    
    if det_data:
        st = det_data.get("stats", {})
        methods["PPO Det"] = {
            "Reward": float(st.get("reward_mean", 0.0)),
            "Std": float(st.get("reward_std", 0.0)),
            "Completed": float(st.get("completed", 0.0)),
            "Obsolete": float(st.get("obsolete", 0.0)),
        }
    
    if stoch_data:
        st = stoch_data.get("stats", {})
        methods["PPO Stoch"] = {
            "Reward": float(st.get("reward_mean", 0.0)),
            "Std": float(st.get("reward_std", 0.0)),
            "Completed": float(st.get("completed", 0.0)),
            "Obsolete": float(st.get("obsolete", 0.0)),
        }
    
    for name, b_data in baselines.items():
        rewards = np.asarray(b_data.get("rewards", []), dtype=float)
        completed = np.asarray(b_data.get("completed", []), dtype=float)
        obsolete = np.asarray(b_data.get("obsolete", []), dtype=float)
        
        methods[name] = {
            "Reward": float(rewards.mean()) if rewards.size else 0.0,
            "Std": float(rewards.std()) if rewards.size else 0.0,
            "Completed": float(completed.mean()) if completed.size else 0.0,
            "Obsolete": float(obsolete.mean()) if obsolete.size else 0.0,
        }

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.axis("tight")
    ax.axis("off")

    cols = ["Method", "Reward", "Std", "Completed", "Obsolete"]
    rows = []
    
    for method, stats in methods.items():
        rows.append([
            method,
            f"{stats['Reward']:.2f}",
            f"{stats['Std']:.2f}",
            f"{stats['Completed']:.2f}",
            f"{stats['Obsolete']:.2f}",
        ])

    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(cols)):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")

    ax.set_title("Evaluation: Summary Statistics", fontsize=14, fontweight="bold", pad=20)
    
    _save_fig(fig, out_png)


# ============================================================================
# Training Plots
# ============================================================================

def plot_training_rewards(
    monitor_data: Dict[str, List[float]],
    baselines: Dict[str, Dict],
    out_png: Path,
    ma_window: int = 5,
) -> None:
    """Plot training rewards over time."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    if monitor_data["reward"]:
        rewards = np.asarray(monitor_data["reward"], dtype=float)
        timesteps = np.asarray(monitor_data["timestep"], dtype=float)
        
        ax.plot(timesteps, rewards, "o", markersize=3, alpha=0.25, 
               color="#3498db", label="PPO raw")
        ax.plot(timesteps, _moving_average(rewards, ma_window), lw=2.8, 
               color="#2980b9", label=f"PPO MA({ma_window})")
        ax.axhline(float(np.mean(rewards)), color="#2c3e50", lw=2, ls="--", 
                  alpha=0.65, label=f"PPO mean: {float(np.mean(rewards)):.2f}")

    # Add baseline lines
    baseline_colors = {"RANDOM": "#e74c3c", "GREEDY": "#f39c12", "UNIQUE": "#8c564b"}
    for name, b_data in baselines.items():
        rewards = np.asarray(b_data.get("rewards", []), dtype=float)
        if rewards.size:
            mean = float(rewards.mean())
            std = float(rewards.std())
            c = baseline_colors.get(name, "#7f8c8d")
            ax.axhline(mean, color=c, ls="--", lw=2, alpha=0.8,
                      label=f"{name} baseline: {mean:.2f}")
            ax.fill_between(ax.get_xlim(), mean - std, mean + std, 
                          color=c, alpha=0.1)

    ax.set_xlabel("Training Timesteps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward Mean", fontsize=11, fontweight="bold")
    ax.set_title("Training: Rewards Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_training_episode_length(
    monitor_data: Dict[str, List[float]],
    out_png: Path,
    ma_window: int = 5,
) -> None:
    """Plot episode length over training."""
    if not monitor_data["length"]:
        return

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    lengths = np.asarray(monitor_data["length"], dtype=float)
    timesteps = np.asarray(monitor_data["timestep"], dtype=float)
    ax.plot(timesteps, lengths, "o", markersize=3, alpha=0.25,
           color="#9b59b6", label="Raw")
    ax.plot(timesteps, _moving_average(lengths, ma_window), lw=2.8,
           color="#8e44ad", label=f"MA({ma_window})")
    ax.axhline(float(np.mean(lengths)), color="#2c3e50", lw=2, ls="--",
              alpha=0.65, label=f"Mean: {float(np.mean(lengths)):.2f}")

    ax.set_xlabel("Training Timesteps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Length (steps)", fontsize=11, fontweight="bold")
    ax.set_title("Training: Episode Length Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_reward_distribution(
    monitor_data: Dict[str, List[float]],
    out_png: Path,
) -> None:
    """Plot reward distribution histogram."""
    if not monitor_data["reward"]:
        return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    rewards = np.asarray(monitor_data["reward"], dtype=float)
    
    ax.hist(rewards, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(rewards), color="red", linestyle="--", linewidth=2,
              label=f"Mean: {np.mean(rewards):.2f}")
    ax.axvline(np.median(rewards), color="green", linestyle="--", linewidth=2,
              label=f"Median: {np.median(rewards):.2f}")
    
    ax.set_xlabel("Episode Reward", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Training: Reward Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_cumulative_reward(
    monitor_data: Dict[str, List[float]],
    out_png: Path,
) -> None:
    """Plot cumulative rewards over training."""
    if not monitor_data["reward"]:
        return

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    rewards = np.asarray(monitor_data["reward"], dtype=float)
    timesteps = np.asarray(monitor_data["timestep"], dtype=float)
    cum_rewards = np.cumsum(rewards)
    
    ax.plot(timesteps, cum_rewards, linewidth=2.5, color="#27ae60", label="Cumulative")
    ax.fill_between(timesteps, cum_rewards, alpha=0.3, color="#27ae60")
    
    ax.set_xlabel("Training Timesteps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=11, fontweight="bold")
    ax.set_title("Training: Cumulative Reward", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


# ============================================================================
# Logit Plots
# ============================================================================

def plot_eval_length_distribution(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    out_png: Path,
) -> None:
    """Plot episode length distribution for eval."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_facecolor("#fafafa")

    if det_data:
        lengths = np.asarray(det_data.get("lengths", []), dtype=float)
        if lengths.size > 0:
            ax.hist(lengths, bins=20, alpha=0.6, label="Deterministic", color="#2980b9")

    if stoch_data:
        lengths = np.asarray(stoch_data.get("lengths", []), dtype=float)
        if lengths.size > 0:
            ax.hist(lengths, bins=20, alpha=0.6, label="Stochastic", color="#27ae60")

    ax.set_xlabel("Episode Length (steps)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Episode Length Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_reward_comparison_bar(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    """Bar chart comparing all methods."""
    methods: Dict[str, Tuple[float, float]] = {}  # mean, std
    
    if det_data:
        st = det_data.get("stats", {})
        methods["PPO Det"] = (
            float(st.get("reward_mean", 0.0)),
            float(st.get("reward_std", 0.0)),
        )
    
    if stoch_data:
        st = stoch_data.get("stats", {})
        methods["PPO Stoch"] = (
            float(st.get("reward_mean", 0.0)),
            float(st.get("reward_std", 0.0)),
        )
    
    for name, b_data in baselines.items():
        rewards = np.asarray(b_data.get("rewards", []), dtype=float)
        methods[name] = (
            float(rewards.mean()) if rewards.size else 0.0,
            float(rewards.std()) if rewards.size else 0.0,
        )

    if not methods:
        return

    labels = list(methods.keys())
    means = [m[0] for m in methods.values()]
    stds = [m[1] for m in methods.values()]
    
    colors_map = {
        "PPO Det": "#2980b9",
        "PPO Stoch": "#27ae60",
        "RANDOM": "#e74c3c",
        "GREEDY": "#f39c12",
        "UNIQUE": "#8c564b",
    }
    colors = [colors_map.get(l, "#7f8c8d") for l in labels]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 6), facecolor="white")
    ax.set_facecolor("#fafafa")
    
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.8, 
          edgecolor="black", linewidth=1.5)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 1, f"{mean:.2f}", ha="center", va="bottom", 
               fontweight="bold", fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Reward Mean ± Std", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Reward Comparison", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)


def plot_training_vs_eval_reward(
    monitor_data: Dict[str, List[float]],
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    out_png: Path,
) -> None:
    """Compare training rewards with eval rewards."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    # Training
    if monitor_data["reward"]:
        rewards = np.asarray(monitor_data["reward"], dtype=float)
        episodes = np.arange(1, len(rewards) + 1)
        ax.scatter(episodes, rewards, alpha=0.3, s=30, color="#3498db", label="Training (raw)")

    # Evaluation
    if det_data:
        det_rewards = np.asarray(det_data.get("rewards", []), dtype=float)
        det_mean = float(det_data.get("stats", {}).get("reward_mean", 0))
        det_std = float(det_data.get("stats", {}).get("reward_std", 0))
        ax.axhline(det_mean, color="#2980b9", lw=2.5, ls="-", alpha=0.9,
                  label=f"PPO Deterministic: {det_mean:.2f} ± {det_std:.2f}")
        ax.fill_between(ax.get_xlim(), det_mean - det_std, det_mean + det_std,
                       color="#2980b9", alpha=0.15)

    if stoch_data:
        stoch_rewards = np.asarray(stoch_data.get("rewards", []), dtype=float)
        stoch_mean = float(stoch_data.get("stats", {}).get("reward_mean", 0))
        stoch_std = float(stoch_data.get("stats", {}).get("reward_std", 0))
        ax.axhline(stoch_mean, color="#27ae60", lw=2.5, ls="-", alpha=0.9,
                  label=f"PPO Stochastic: {stoch_mean:.2f} ± {stoch_std:.2f}")
        ax.fill_between(ax.get_xlim(), stoch_mean - stoch_std, stoch_mean + stoch_std,
                       color="#27ae60", alpha=0.15)

    ax.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
    ax.set_title("Training vs Evaluation: Reward Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    _save_fig(fig, out_png)

def find_latest_run(seed: int) -> Path:
    base = Path("runs") / f"seed_{seed}"

    if not base.exists():
        raise FileNotFoundError(f"Seed folder not found: {base}")

    runs = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime)

    if not runs:
        raise FileNotFoundError(f"No runs found in {base}")

    return runs[-1]
def plot_noop_and_action_diagnostics(det_data, stoch_data, out_png):
    """NOOP fraction per episode + aggregate action-index histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # --- Left: NOOP fraction over episodes ---
    ax = axes[0]
    for label, color, data in [("PPO Det", "#2980b9", det_data), ("PPO Stoch", "#27ae60", stoch_data)]:
        if not data or "noop_fractions" not in data:
            continue
        nf = np.asarray(data["noop_fractions"], dtype=float)
        if nf.size == 0:
            continue
        ax.plot(np.arange(1, nf.size + 1), nf, color=color, lw=1.8,
                label=f"{label} – mean {nf.mean():.2%}")
    ax.set_xlabel("Episode", fontweight="bold"); ax.set_ylabel("NOOP fraction", fontweight="bold")
    ax.set_title("Fraction of decisions that were NOOP", fontweight="bold")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.25)

    # --- Right: aggregate action-index histogram (which candidate slot / noop gets picked) ---
    ax = axes[1]
    for label, color, data in [("PPO Det", "#2980b9", det_data), ("PPO Stoch", "#27ae60", stoch_data)]:
        if not data or "action_hists" not in data or not data["action_hists"]:
            continue
        hists = np.asarray(data["action_hists"], dtype=float)
        agg = hists.sum(axis=0)
        agg = agg / agg.sum()
        idx = np.arange(agg.size)
        ax.bar(idx + (0.0 if label == "PPO Det" else 0.35), agg, width=0.35, label=label, color=color, alpha=0.8)
    ax.set_xlabel("Action index (last = NOOP)", fontweight="bold")
    ax.set_ylabel("Fraction of decisions", fontweight="bold")
    ax.set_title("Chosen action-slot distribution", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.25, axis="y")

    _save_fig(fig, out_png)

def _safe_series(d: Optional[Dict], key: str) -> np.ndarray:
    if not d:
        return np.array([], dtype=float)
    arr = d.get(key, [])
    return np.asarray(arr, dtype=float) if arr is not None else np.array([], dtype=float)

def _safe_rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    if num.size == 0 or den.size == 0:
        return np.array([], dtype=float)
    n = min(num.size, den.size)
    num = num[:n]
    den = den[:n]
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def plot_reward_components_debug(det_data, stoch_data, out_png, ma_window=5):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="white")

    for ax, data, title, color_base in [
        (axes[0], det_data, "Deterministic", "#2980b9"),
        (axes[1], stoch_data, "Stochastic", "#27ae60"),
    ]:
        if not data:
            ax.set_title(f"{title}: no data")
            continue

        comp = _safe_series(data, "ep_r_comp")
        wait = _safe_series(data, "ep_r_wait")
        dead = _safe_series(data, "ep_r_deadline")
        obs  = _safe_series(data, "ep_r_obsolete")
        n = max(comp.size, wait.size, dead.size, obs.size)
        if n == 0:
            ax.set_title(f"{title}: reward component logs not found")
            continue

        x = np.arange(1, n + 1)

        def pad(a):
            if a.size == n:
                return a
            b = np.zeros(n, dtype=float)
            b[:min(n, a.size)] = a[:min(n, a.size)]
            return b

        comp = _moving_average(pad(comp), ma_window)
        wait = _moving_average(pad(wait), ma_window)
        dead = _moving_average(pad(dead), ma_window)
        obs  = _moving_average(pad(obs),  ma_window)

        ax.plot(x, comp, lw=2.2, label="r_comp")
        ax.plot(x, wait, lw=2.2, label="r_wait")
        ax.plot(x, dead, lw=2.2, label="r_deadline")
        ax.plot(x, obs,  lw=2.2, label="r_obsolete")
        ax.axhline(0.0, color="black", lw=1, alpha=0.4)
        ax.set_title(f"{title}: Reward Components (MA)", fontweight="bold")
        ax.set_ylabel("Episode component sum")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    axes[1].set_xlabel("Episode", fontweight="bold")
    _save_fig(fig, out_png)


def plot_action_quality_rates_debug(det_data, stoch_data, out_png, ma_window=5):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="white")

    for ax, data, title, color in [
        (axes[0], det_data, "Deterministic", "#2980b9"),
        (axes[1], stoch_data, "Stochastic", "#27ae60"),
    ]:
        if not data:
            ax.set_title(f"{title}: no data")
            continue

        inv = _safe_series(data, "ep_invalid_action_count")
        tot = _safe_series(data, "ep_total_action_count")
        cap = _safe_series(data, "ep_capacity_rejected_count")
        cfl = _safe_series(data, "ep_conflict_dropped_count")

        inv_r = _safe_rate(inv, tot)
        cap_r = _safe_rate(cap, tot)
        cfl_r = _safe_rate(cfl, tot)

        n = max(inv_r.size, cap_r.size, cfl_r.size)
        if n == 0:
            ax.set_title(f"{title}: action-quality logs not found")
            continue

        x = np.arange(1, n + 1)

        def pad(a):
            if a.size == n:
                return a
            b = np.zeros(n, dtype=float)
            b[:min(n, a.size)] = a[:min(n, a.size)]
            return b

        ax.plot(x, _moving_average(pad(inv_r), ma_window), lw=2.2, label="invalid_rate")
        ax.plot(x, _moving_average(pad(cap_r), ma_window), lw=2.2, label="capacity_reject_rate")
        ax.plot(x, _moving_average(pad(cfl_r), ma_window), lw=2.2, label="conflict_drop_rate")

        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{title}: Action Quality Rates (MA)", fontweight="bold")
        ax.set_ylabel("Rate")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    axes[1].set_xlabel("Episode", fontweight="bold")
    _save_fig(fig, out_png)


def plot_mask_pressure_noop_debug(det_data, stoch_data, out_png, ma_window=5):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="white")

    for ax, data, title, color in [
        (axes[0], det_data, "Deterministic", "#2980b9"),
        (axes[1], stoch_data, "Stochastic", "#27ae60"),
    ]:
        if not data:
            ax.set_title(f"{title}: no data")
            continue

        maskz = _safe_series(data, "ep_mask_zero_count")
        noop = _safe_series(data, "noop_fractions")
        n = max(maskz.size, noop.size)
        if n == 0:
            ax.set_title(f"{title}: mask/noop logs not found")
            continue

        x = np.arange(1, n + 1)

        def pad(a):
            if a.size == n:
                return a
            b = np.zeros(n, dtype=float)
            b[:min(n, a.size)] = a[:min(n, a.size)]
            return b

        m = _moving_average(pad(maskz), ma_window)
        npf = _moving_average(pad(noop), ma_window)

        ax2 = ax.twinx()
        l1 = ax.plot(x, m, lw=2.2, color="#8e44ad", label="mask_zero_count")
        l2 = ax2.plot(x, npf, lw=2.2, color="#16a085", label="noop_fraction")

        ax.set_title(f"{title}: Mask Pressure & NOOP", fontweight="bold")
        ax.set_ylabel("Mask zero count", color="#8e44ad")
        ax2.set_ylabel("NOOP fraction", color="#16a085")
        ax.grid(alpha=0.25)

        lines = l1 + l2
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, fontsize=9, loc="best")

    axes[1].set_xlabel("Episode", fontweight="bold")
    _save_fig(fig, out_png)


def plot_debug_summary_table(det_data, stoch_data, out_png):
    def extract(d):
        st = (d or {}).get("stats", {})
        return {
            "invalid_rate": float(st.get("invalid_action_rate", 0.0)),
            "capacity_reject_rate": float(st.get("capacity_reject_rate", 0.0)),
            "conflict_drop_rate": float(st.get("conflict_drop_rate", 0.0)),
            "noop_frac_mean": float(st.get("noop_frac_mean", 0.0)),
            # noop breakdown: forced = no candidates offered (structural),
            # chosen = policy picked noop despite candidates being available.
            # chosen_noop_rate_when_available is the one that matters most —
            # it's the noop rate restricted to decisions the policy could
            # actually have acted on.
            "noop_frac_forced": float(st.get("noop_frac_forced", 0.0)),
            "noop_frac_chosen": float(st.get("noop_frac_chosen", 0.0)),
            "chosen_noop_rate_when_available": float(st.get("chosen_noop_rate_when_available", 0.0)),
            "mask_zero_mean": float(st.get("mask_zero_mean", 0.0)),
            "r_comp_mean": float(st.get("r_comp_mean", 0.0)),
            "r_wait_mean": float(st.get("r_wait_mean", 0.0)),
            "r_deadline_mean": float(st.get("r_deadline_mean", 0.0)),
            "r_obsolete_mean": float(st.get("r_obsolete_mean", 0.0)),
        }
 
    det = extract(det_data)
    sto = extract(stoch_data)
 
    cols = ["Metric", "Deterministic", "Stochastic"]
    rows = []
    for k in det.keys():
        rows.append([k, f"{det[k]:.4f}", f"{sto[k]:.4f}"])
 
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
 
    for i in range(len(cols)):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")
 
    ax.set_title("Debug Summary Metrics", fontsize=14, fontweight="bold", pad=16)
    _save_fig(fig, out_png)

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate comprehensive evaluation and training plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-dir", type=str, default=None,
                help="Run directory (overrides eval-dir/baseline-dir)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed used to locate latest run if run-dir not provided")
    ap.add_argument("--eval-dir", type=str, default="eval_results",
                   help="Directory with eval_results_*.json")
    ap.add_argument("--baseline-dir", type=str, default="baseline_results",
                   help="Directory with baseline_results_*.json")
    ap.add_argument("--monitor", type=str, default="logs/monitor.csv",
                   help="Monitor CSV from training")
    ap.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for plots")
    ap.add_argument("--ma-window", type=int, default=5,
                   help="Moving average window")
    
    ap.add_argument("--base-dir", type=str, default="baseline_results")

     
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(args.seed)

    eval_dir = run_dir / "eval_results"
    base_dir = Path(args.base_dir)

    baseline_dir = run_dir / "baseline_results"
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(output_dir,'output_dir')
    print("="*80)
    print("Generating Comprehensive Evaluation & Training Plots")
    print("="*80 + "\n")

    # Load evaluation results
    print("Loading evaluation results...")
    det_file = eval_dir / "deterministic.json"
    stoch_file = eval_dir / "stochastic.json"

    det_data = _load_json(det_file) if det_file.exists() else None
    stoch_data = _load_json(stoch_file) if stoch_file.exists() else None

    if det_data:
        print(f" Deterministic ({len(det_data.get('rewards', []))} episodes)")
    if stoch_data:
        print(f" Stochastic ({len(stoch_data.get('rewards', []))} episodes)")

    # Load baselines
    print("Loading baseline results...")
    baseline_file = base_dir / "baseline_results_all.json"
    baselines = load_baselines_all(baseline_file)
    
    if baselines:
        print(f"Loaded {len(baselines)} baseline policies")

    # Load training data
    print(" Loading training data...")
    monitor_data = _load_monitor_csv(Path(run_dir) / args.monitor)
    if monitor_data["reward"]:
        print(f" Monitor CSV ({len(monitor_data['reward'])} episodes)")

    # Generate plots
    print(f" Generating plots {output_dir}\n")

    # Evaluation plots
    print("Evaluation Plots:")
    plot_eval_rewards_per_episode(det_data, stoch_data, baselines,
                                 output_dir / "01_eval_rewards_per_episode.png",
                                 ma_window=args.ma_window)
    plot_eval_rewards_boxplot(det_data, stoch_data, baselines,
                             output_dir / "02_eval_rewards_boxplot.png")
    plot_eval_completion_obsolete(det_data, stoch_data, baselines,
                                 output_dir / "03_eval_completion_obsolete.png")
    plot_eval_stats_summary(det_data, stoch_data, baselines,
                           output_dir / "04_eval_stats_summary.png")
    plot_eval_length_distribution(det_data, stoch_data,
                                 output_dir / "05_eval_length_distribution.png")
    plot_reward_comparison_bar(det_data, stoch_data, baselines,
                              output_dir / "06_eval_reward_comparison.png")

    # Training plots
    print("Training Plots:")
    plot_training_rewards(monitor_data, baselines,
                         output_dir / "07_training_rewards.png",
                         ma_window=args.ma_window)
    plot_training_episode_length(monitor_data,
                                output_dir / "08_training_episode_length.png",
                                ma_window=args.ma_window)
    plot_reward_distribution(monitor_data,
                            output_dir / "09_training_reward_distribution.png")
    plot_cumulative_reward(monitor_data,
                          output_dir / "10_training_cumulative_reward.png")
    plot_training_vs_eval_reward(monitor_data, det_data, stoch_data,
                                output_dir / "11_training_vs_eval.png")
    plot_noop_and_action_diagnostics(det_data, stoch_data, output_dir / "12_noop_and_action_diagnostic.png")    # Debug plots from eval_ppo extended fields
    print("Debug Plots:")
    plot_reward_components_debug(
        det_data, stoch_data, output_dir / "13_reward_components.png", ma_window=args.ma_window
    )
    plot_action_quality_rates_debug(
        det_data, stoch_data, output_dir / "14_action_quality_rates.png", ma_window=args.ma_window
    )
    plot_mask_pressure_noop_debug(
        det_data, stoch_data, output_dir / "15_mask_pressure_noop.png", ma_window=args.ma_window
    )
    plot_debug_summary_table(
        det_data, stoch_data, output_dir / "16_debug_summary_table.png"
    )
        # Debug plots from eval_ppo extended fields
    print("Debug Plots:")
    plot_reward_components_debug(
        det_data, stoch_data, output_dir / "13_reward_components.png", ma_window=args.ma_window
    )
    plot_action_quality_rates_debug(
        det_data, stoch_data, output_dir / "14_action_quality_rates.png", ma_window=args.ma_window
    )
    plot_mask_pressure_noop_debug(
        det_data, stoch_data, output_dir / "15_mask_pressure_noop.png", ma_window=args.ma_window
    )
    plot_debug_summary_table(
        det_data, stoch_data, output_dir / "16_debug_summary_table.png"
    )
    print(f"generated 11 plots in {output_dir}\n")


if __name__ == "__main__":
    main()