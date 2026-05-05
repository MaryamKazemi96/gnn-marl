#!/usr/bin/env python3
"""
Comprehensive evaluation and plotting script.

Runs PPO evaluation in BOTH deterministic and stochastic modes for every
seed found under --checkpoint-dir, saves the results as JSON, then
produces a full set of plots:

  Per-seed (checkpoints_ppo/seed_N/eval_plots/):
    eval_rewards_per_episode.png  - per-episode reward line, det vs stoch
    eval_rewards_boxplot.png      - boxplot, det + stoch + baselines
    eval_completion_obsolete.png  - bar chart completions & obsolete
    eval_reward_components.png    - bar chart of reward sub-components
    training_rewards.png          - TensorBoard reward curve + baseline lines
    training_entropy.png          - entropy / entropy-loss over training
    training_value_overview.png   - value_loss, explained_var, KL, clip-frac
    training_value_loss.png       - critic loss (log scale)

  Aggregate (checkpoints_ppo/eval_plots/):
    agg_rewards_boxplot.png       - boxplot across all seeds, det+stoch+baselines
    agg_completion_obsolete.png   - bar chart across all seeds

Usage (runs eval + plots for all seeds):
    python plot_evaluation.py

Usage (skip re-running eval if JSONs already exist):
    python plot_evaluation.py --skip-eval

Usage (single seed, custom episodes):
    python plot_evaluation.py --seeds 42 --episodes 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorboard.backend.event_processing import event_accumulator

# These legacy classes live in environment2.py and sb3_env_wrapper.py.
# MultiTaskAllocationEnv is the OLD env used by make_env() below.
# WarehouseEnvSB3Final and RTGNNPolicy are optional; if missing the eval
# runner is disabled but all plotting functions still work.
try:
    from src.environment.environment2 import MultiTaskAllocationEnv
except ImportError:
    MultiTaskAllocationEnv = None  # type: ignore[assignment,misc]
try:
    from src.environment.sb3_env_wrapper import WarehouseEnvSB3Final  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    WarehouseEnvSB3Final = None  # type: ignore[assignment,misc]
try:
    from src.models.sb3_gnn_policy import RTGNNPolicy  # type: ignore[import]
except ImportError:
    RTGNNPolicy = None  # type: ignore[assignment,misc]


# ============================================================
# Helpers
# ============================================================

def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"✓ Saved JSON: {p}")


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.size == 0 or window <= 1 or data.size < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad_len = data.size - ma.size
    return np.concatenate([data[:pad_len], ma])


# ============================================================
# Environment / config helpers (mirrors eval_ppo.py)
# ============================================================

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(config: Dict, seed: int):
    """Build the evaluation environment.

    Supports two env configs:
    - New style (MultiAgentTaskEnv): config["environment"] has "data_dir"
    - Legacy style (MultiTaskAllocationEnv): config["environment"] has "agents_file"

    Falls back gracefully if legacy classes are unavailable.
    """
    from src.environment.environment import MultiAgentTaskEnv

    env_cfg = config.get("environment", config)  # tolerate flat configs

    # ---- New-style env (MultiAgentTaskEnv) ----
    if "data_dir" in env_cfg:
        data_dir = Path(env_cfg["data_dir"])
        agents = np.load(data_dir / "agents.npy", allow_pickle=True)
        batches: list = []
        i = 0
        while (data_dir / f"tasks_batch_{i}.npy").exists():
            batches.append(np.load(data_dir / f"tasks_batch_{i}.npy", allow_pickle=True))
            i += 1
        env = MultiAgentTaskEnv(
            agents=agents,
            tasks_batches=batches,
            K_max=env_cfg.get("K_max", 5),
            N_max=env_cfg.get("N_max", 15),
            E_max=env_cfg.get("E_max", 50),
            use_xy_pickup=env_cfg.get("use_xy_pickup", False),
            normalize_features=env_cfg.get("normalize_features", True),
            use_node_type=env_cfg.get("use_node_type", True),
            use_ego_robot=env_cfg.get("use_ego_robot", True),
            use_edge_rt=env_cfg.get("use_edge_rt", False),
            two_hop=env_cfg.get("two_hop", False),
            vicinity_m=env_cfg.get("vicinity_m", 20.0),
            max_steps=env_cfg.get("max_steps", 1000),
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    # ---- Legacy env (MultiTaskAllocationEnv) ----
    if MultiTaskAllocationEnv is None:
        raise ImportError(
            "Legacy MultiTaskAllocationEnv not available. "
            "Add 'data_dir' to your config to use MultiAgentTaskEnv instead."
        )
    agents = np.load(env_cfg["agents_file"], allow_pickle=True)
    batches = []
    for i in range(env_cfg.get("n_batches", 0)):
        batch_file = Path(env_cfg["data_dir"]) / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batches.append(np.load(batch_file, allow_pickle=True))

    base_env = MultiTaskAllocationEnv(  # type: ignore[call-arg]
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=env_cfg["radius"],
        feature_size=env_cfg["feature_size"],
        use_true_id=env_cfg["use_true_id"],
        all_batches=True,
    )
    if WarehouseEnvSB3Final is None:
        raise ImportError("WarehouseEnvSB3Final not available in sb3_env_wrapper.py")
    env = WarehouseEnvSB3Final(  # type: ignore[call-arg]
        base_env,
        assignment_interval=env_cfg["assignment_interval"],
        k_max=env_cfg.get("k_max", 5),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ============================================================
# Evaluation runner
# ============================================================

COMP_KEYS: List[str] = [
  "rew/pickups_this_step",
  "rew/deliveries_this_step",
  "rew/obsolete_this_step",
  "rew/step_penalty",
]
# ["rew/pickup", "rew/delivery", "rew/obsolete", "rew/step_penalty"]

def run_evaluation(
    model_path: Path,
    config: Dict,
    seed: int,
    n_episodes: int,
    deterministic: bool,
) -> Dict[str, Any]:
    """Run n_episodes with a loaded PPO checkpoint; return results dict.

    Requires RTGNNPolicy to be importable from src.models.sb3_gnn_policy.
    """
    if RTGNNPolicy is None:
        raise ImportError(
            "RTGNNPolicy could not be imported (src/models/sb3_gnn_policy.py missing). "
            "Cannot run evaluation."
        )
    set_seed(seed)

    env = make_env(config, seed=seed)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env, custom_objects={"policy_class": RTGNNPolicy})

    rewards: List[float] = []
    completions: List[int] = []
    obsolete: List[int] = []
    lengths: List[int] = []
    comp_sums: Dict[str, List[float]] = {k: [] for k in COMP_KEYS}

    mode_label = "deterministic" if deterministic else "stochastic"

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_comp: Dict[str, float] = {k: 0.0 for k in COMP_KEYS}
        ep_completed = 0
        ep_obsolete = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            ep_reward += float(r[0])
            ep_len += 1

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info, dict):
                for k in COMP_KEYS:
                    v = info.get(k, None)
                    if isinstance(v, (int, float, np.number)):
                        ep_comp[k] += float(v)
                if "episode_completed" in info:
                    ep_completed = int(info["episode_completed"])
                if "episode_obsolete" in info:
                    ep_obsolete = int(info["episode_obsolete"])

        rewards.append(ep_reward)
        lengths.append(ep_len)
        completions.append(ep_completed)
        obsolete.append(ep_obsolete)
        for k in COMP_KEYS:
            comp_sums[k].append(ep_comp[k])

        if (ep + 1) % 10 == 0:
            print(
                f"  [{mode_label}] seed {seed} ep {ep+1}/{n_episodes}: "
                f"reward={ep_reward:.2f} completed={ep_completed} obsolete={ep_obsolete}"
            )

    vec_env.close()

    return {
        "seed": int(seed),
        "n_episodes": int(n_episodes),
        "deterministic": bool(deterministic),
        "rewards": rewards,
        "lengths": lengths,
        "completions": completions,
        "obsolete": obsolete,
        "reward_components": comp_sums,
        "stats": {
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "completion_mean": float(np.mean(completions)) if completions else 0.0,
            "obsolete_mean": float(np.mean(obsolete)) if obsolete else 0.0,
        },
    }


# ============================================================
# Baseline loading
# ============================================================

def _get_baseline_block(data: Dict) -> Dict:
    if "results" in data and isinstance(data["results"], dict):
        return data["results"]
    return data


def load_baselines_all(checkpoint_dir: Path) -> Dict[str, Dict]:
    """Load baseline_results_all.json → {RANDOM: {...}, GREEDY: {...}, UNIQUE: {...}}"""
    p = checkpoint_dir / "baseline_results_all.json"
    if not p.exists():
        return {}
    data = _load_json(p)
    block = _get_baseline_block(data)
    out: Dict[str, Dict] = {}
    for pol in ["random", "greedy", "unique"]:
        b = block.get(pol, None)
        if isinstance(b, dict):
            out[pol.upper()] = {
                "rewards": b.get("rewards", []),
                "completions": b.get("completions", []),
                "obsolete": b.get("obsolete", []),
            }
    return out


def load_baseline_stats_for_seed(root_dir: Path, seed: int) -> Optional[Dict[str, Dict[str, float]]]:
    """Load per-seed baseline JSON files → {random: {mean, std}, ...}"""
    out: Dict[str, Dict[str, float]] = {}
    for pol in ["random", "greedy", "unique"]:
        p = root_dir / f"baseline_{pol}_seed_{seed}.json"
        if not p.exists():
            continue
        try:
            data = _load_json(p)
        except Exception as exc:
            print(f"[WARN] Could not read {p}: {exc}")
            continue
        st = (data or {}).get("stats", {})
        if isinstance(st, dict):
            out[pol] = {
                "mean": float(st.get("reward_mean", 0.0)),
                "std": float(st.get("reward_std", 0.0)),
            }
    return out if out else None


# ============================================================
# TensorBoard loading
# ============================================================

def load_tensorboard_data(tb_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    run_dirs = list(tb_dir.glob("PPO_*"))
    if not run_dirs:
        print(f"[WARN] No TensorBoard runs found under: {tb_dir}")
        return {}
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    print(f"  Loading TensorBoard run: {latest_run}")
    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()
    data: Dict[str, Dict[str, List[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [float(e.step) for e in events],
            "values": [float(e.value) for e in events],
        }
    return data


# ============================================================
# Evaluation plots
# ============================================================

# def plot_eval_rewards_per_episode(det_data: Dict, stoch_data: Dict, out_png: Path, ma_window: int = 10) -> None:
#     """Line plot of per-episode rewards for deterministic and stochastic modes."""
#     fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
#     ax.set_facecolor("#fafafa")

#     for label, color, data in [
#         ("PPO Deterministic", "#2980b9", det_data),
#         ("PPO Stochastic", "#e67e22", stoch_data),
#     ]:
#         if data is None:
#             continue
#         rewards = np.asarray(data.get("rewards", []), dtype=float)
#         if rewards.size == 0:
#             continue
#         episodes = np.arange(1, rewards.size + 1)
#         ax.plot(episodes, rewards, alpha=0.2, color=color, linewidth=0.8)
#         ax.plot(episodes, _moving_average(rewards, ma_window), lw=2.4, color=color,
#                 label=f"{label} MA({ma_window}) – mean {rewards.mean():.2f}")
#         ax.axhline(rewards.mean(), color=color, lw=1.5, ls="--", alpha=0.6)

#     ax.set_xlabel("Episode", fontsize=11, fontweight="bold")
#     ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
#     ax.set_title("Evaluation: Per-Episode Rewards (Deterministic vs Stochastic)", fontsize=14, fontweight="bold")
#     ax.legend(fontsize=10)
#     ax.grid(alpha=0.25)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     _save_fig(fig, out_png)
def plot_eval_rewards_per_episode(
    det_data: Dict,
    stoch_data: Dict,
    baselines: Dict[str, Dict],
    out_png: Path,
    ma_window: int = 10,
) -> None:
    """
    Line plot of per-episode rewards:
      - PPO deterministic
      - PPO stochastic
      - Baselines (RANDOM/GREEDY/UNIQUE) if available

    baselines format (from load_baselines_all):
      {
        "RANDOM": {"rewards": [...], "completions": [...], "obsolete": [...]},
        "GREEDY": {...},
        "UNIQUE": {...},
      }
    """
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    # --- PPO curves ---
    for label, color, data in [
        ("PPO Deterministic", "#2980b9", det_data),
        ("PPO Stochastic", "#3CB371", stoch_data),
    ]:
        if data is None:
            continue
        rewards = np.asarray(data.get("rewards", []), dtype=float)
        if rewards.size == 0:
            continue
        episodes = np.arange(1, rewards.size + 1)
        ax.plot(episodes, rewards, alpha=0.18, color=color, linewidth=0.8)
        ax.plot(
            episodes,
            _moving_average(rewards, ma_window),
            lw=2.6,
            color=color,
            label=f"{label} MA({ma_window}) – mean {rewards.mean():.2f}",
        )
        ax.axhline(rewards.mean(), color=color, lw=1.4, ls="--", alpha=0.55)

    # --- Baselines ---
    # Keep a stable order and consistent colors.
    baseline_order = ["RANDOM", "GREEDY", "UNIQUE"]
    baseline_colors = {
        "RANDOM": "#e74c3c",
        "GREEDY": "#f39c12",
        "UNIQUE": "#8c564b",
    }

    if isinstance(baselines, dict):
        for name in baseline_order:
            if name not in baselines:
                continue
            b = baselines.get(name, {})
            rewards = np.asarray(b.get("rewards", []), dtype=float)
            if rewards.size == 0:
                continue
            episodes = np.arange(1, rewards.size + 1)
            c = baseline_colors.get(name, "#7f8c8d")

            ax.plot(episodes, rewards, alpha=0.12, color=c, linewidth=0.8)
            ax.plot(
                episodes,
                _moving_average(rewards, ma_window),
                lw=2.2,
                color=c,
                label=f"{name} MA({ma_window}) – mean {rewards.mean():.2f}",
            )
            ax.axhline(rewards.mean(), color=c, lw=1.2, ls="--", alpha=0.45)

    ax.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward", fontsize=11, fontweight="bold")
    ax.set_title(
        "Evaluation: Per-Episode Rewards (PPO Det/Stoch + Baselines)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_training_noop_fraction(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    """
    Plot NOOP fraction during training (TensorBoard scalar: policy/noop_fraction).

    Logged by FinalTaskAllocationCallback if:
      - env info contains action_mask with shape [R, K+1]
      - actions are MultiDiscrete with shape (R,)
    """
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    s = tb_data.get("policy/noop_fraction")
    if s is None:
        ax.axis("off")
        ax.set_title(
            "Missing TensorBoard tag: policy/noop_fraction\n"
            "Fix: ensure wrapper puts action_mask into info and callback is used during training.",
            fontsize=14,
            fontweight="bold",
        )
        _save_fig(fig, out_png)
        return

    steps = np.asarray(s["steps"], dtype=float)
    vals = np.asarray(s["values"], dtype=float)

    ax.plot(steps, vals, "o", markersize=4, alpha=0.25, color="#34495e", label="Raw")
    ax.plot(
        steps,
        _moving_average(vals, ma_window),
        lw=2.8,
        color="#2c3e50",
        label=f"MA({ma_window})",
    )
    ax.axhline(float(np.mean(vals)), color="#7f8c8d", lw=1.8, ls="--", alpha=0.65,
               label=f"Mean: {float(np.mean(vals)):.3f}")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("NOOP fraction (over robots)", fontsize=11, fontweight="bold")
    ax.set_title("Training: NOOP Action Fraction", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, out_png)

def plot_eval_rewards_boxplot(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    """Boxplot: PPO-Det, PPO-Stoch, then baseline policies."""
    ordered: Dict[str, np.ndarray] = {}
    for label, data in [("PPO Det", det_data), ("PPO Stoch", stoch_data)]:
        if data is not None:
            arr = np.asarray(data.get("rewards", []), dtype=float)
            if arr.size:
                ordered[label] = arr
    for name, d in baselines.items():
        arr = np.asarray(d.get("rewards", []), dtype=float)
        if arr.size:
            ordered[name] = arr

    if not ordered:
        print("[WARN] No reward data for boxplot.")
        return

    labels = list(ordered.keys())
    series = list(ordered.values())

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    bp = ax.boxplot(series, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2))
    colors_list = ["#2980b9", "#e67e22", "#e74c3c", "#f39c12", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors_list[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Episode reward", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Reward Distribution (Det vs Stoch vs Baselines)", fontsize=13, fontweight="bold")
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
    """Grouped bar chart: mean completions & obsolete for each method."""
    methods: Dict[str, Dict] = {}
    for label, data in [("PPO Det", det_data), ("PPO Stoch", stoch_data)]:
        if data is not None:
            methods[label] = data
    methods.update(baselines)

    if not methods:
        return

    labels: List[str] = []
    comp_means: List[float] = []
    obs_means: List[float] = []

    for name, d in methods.items():
        c = np.asarray(d.get("completions", []), dtype=float)
        o = np.asarray(d.get("obsolete", []), dtype=float)
        labels.append(name)
        comp_means.append(float(c.mean()) if c.size else 0.0)
        obs_means.append(float(o.mean()) if o.size else 0.0)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.bar(x - w / 2, comp_means, width=w, label="Completed (mean)", color="#27ae60", alpha=0.8)
    ax.bar(x + w / 2, obs_means, width=w, label="Obsolete (mean)", color="#c0392b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count per episode", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Completions & Obsolete Tasks (Det vs Stoch vs Baselines)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_eval_reward_components(
    det_data: Optional[Dict],
    stoch_data: Optional[Dict],
    out_png: Path,
) -> None:
    """Bar chart of reward component means for both modes side by side."""
    keys = COMP_KEYS

    def _extract(data: Optional[Dict]) -> Optional[Dict[str, np.ndarray]]:
        if data is None:
            return None
        rc = data.get("reward_components", None)
        if not isinstance(rc, dict):
            return None
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            arr = np.asarray(rc.get(k, []), dtype=float)
            if arr.size:
                out[k] = arr
        return out if out else None

    det_rc = _extract(det_data)
    stoch_rc = _extract(stoch_data)

    if det_rc is None and stoch_rc is None:
        print("[INFO] No reward_components data; skipping component plot.")
        return

    # Use the union of keys present in either
    present_keys = []
    for k in keys:
        if (det_rc and k in det_rc) or (stoch_rc and k in stoch_rc):
            present_keys.append(k)

    if not present_keys:
        return

    x = np.arange(len(present_keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(present_keys) * 2.2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")

    for offset, label, color, rc in [
        (-w / 2, "PPO Det", "#2980b9", det_rc),
        (+w / 2, "PPO Stoch", "#e67e22", stoch_rc),
    ]:
        if rc is None:
            continue
        means = [float(rc[k].mean()) if k in rc else 0.0 for k in present_keys]
        stds = [float(rc[k].std()) if k in rc else 0.0 for k in present_keys]
        ax.bar(x + offset, means, width=w, yerr=stds, capsize=5,
               label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(present_keys, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Reward per episode (mean ± std)", fontsize=11, fontweight="bold")
    ax.set_title("Evaluation: Reward Components (Det vs Stoch)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


# ============================================================
# Training plots (mirrors plot_training.py)
# ============================================================

def _plot_series(
    ax,
    series: Dict[str, List[float]],
    title: str,
    color: str,
    ma_window: int,
    yscale: Optional[str] = None,
) -> None:
    ax.set_facecolor("#fafafa")
    steps = np.asarray(series["steps"], dtype=float)
    vals = np.asarray(series["values"], dtype=float)
    ax.plot(steps, vals, "o", markersize=3, alpha=0.25, color=color, label="Raw")
    ax.plot(steps, _moving_average(vals, ma_window), lw=2.4, color=color, label=f"MA({ma_window})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)
    if yscale:
        ax.set_yscale(yscale)
    ax.legend(fontsize=9, loc="best")


def _add_baseline_lines(ax, baselines: Optional[Dict[str, Dict[str, float]]], show_std: bool) -> None:
    if not baselines:
        return
    colors = {"random": "#d62728", "greedy": "#ff7f0e", "unique": "#8c564b"}
    display = {"random": "Random", "greedy": "Greedy", "unique": "Greedy-Unique"}
    for pol in ["random", "greedy", "unique"]:
        if pol not in baselines:
            continue
        mean = float(baselines[pol]["mean"])
        std = float(baselines[pol].get("std", 0.0))
        c = colors[pol]
        ax.axhline(mean, color=c, ls="--", lw=2.2, alpha=0.9,
                   label=f"{display[pol]} baseline ({mean:.2f})")
        if show_std and std > 0:
            ax.axhline(mean + std, color=c, ls=":", lw=1.2, alpha=0.75)
            ax.axhline(mean - std, color=c, ls=":", lw=1.2, alpha=0.75)


def plot_training_rewards(
    tb_data: Dict,
    seed: int,
    baselines: Optional[Dict],
    out_png: Path,
    ma_window: int,
    baseline_std: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    ax.set_facecolor("#fafafa")
    s = tb_data.get("rollout/ep_rew_mean")
    if s is None:
        ax.axis("off")
        ax.set_title("Missing rollout/ep_rew_mean in TensorBoard logs", fontsize=14, fontweight="bold")
        _save_fig(fig, out_png)
        return
    steps = np.asarray(s["steps"], dtype=float)
    rewards = np.asarray(s["values"], dtype=float)
    ax.plot(steps, rewards, "o", markersize=4, alpha=0.25, color="#3498db", label="PPO raw")
    ax.plot(steps, _moving_average(rewards, ma_window), lw=2.8, color="#2980b9", label=f"PPO MA({ma_window})")
    ax.axhline(float(np.mean(rewards)), color="#2c3e50", lw=2, ls="--", alpha=0.65,
               label=f"PPO mean: {float(np.mean(rewards)):.2f}")
    _add_baseline_lines(ax, baselines, baseline_std)
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Episode Reward Mean", fontsize=11, fontweight="bold")
    ax.set_title(f"Training Rewards (Seed {seed}) – PPO vs Baselines", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_training_entropy(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    s = tb_data.get("train/entropy_loss") or tb_data.get("train/entropy")
    if s is None:
        ax.axis("off")
        ax.set_title("No entropy tag found (train/entropy_loss or train/entropy).", fontsize=14, fontweight="bold")
    else:
        tag = "train/entropy_loss" if "train/entropy_loss" in tb_data else "train/entropy"
        _plot_series(ax, s, f"Exploration: Entropy ({tag})", "#9b59b6", ma_window)
    _save_fig(fig, out_png)


def plot_training_value_overview(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    tags = [
        ("train/value_loss", "Critic loss (train/value_loss)", "#2980b9", "log"),
        ("train/explained_variance", "Explained variance (train/explained_variance)", "#16a085", None),
        ("train/approx_kl", "Approx KL (train/approx_kl)", "#c0392b", None),
        ("train/clip_fraction", "Clip fraction (train/clip_fraction)", "#d35400", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    any_plotted = False
    for i, (tag, title, color, yscale) in enumerate(tags):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        s = tb_data.get(tag)
        if s is None:
            ax.axis("off")
            ax.set_title(f"Missing {tag}", fontsize=12, fontweight="bold")
            continue
        any_plotted = True
        _plot_series(ax, s, title, color, ma_window, yscale=yscale)
    if not any_plotted:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
        ax.axis("off")
        ax.set_title("No value-related tags found.", fontsize=14, fontweight="bold")
    _save_fig(fig, out_png)


def plot_training_value_loss(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    s = tb_data.get("train/value_loss")
    if s is None:
        ax.axis("off")
        ax.set_title("Missing train/value_loss in TensorBoard logs", fontsize=14, fontweight="bold")
    else:
        _plot_series(ax, s, "Critic Loss (train/value_loss) [log scale]", "#2980b9", ma_window, yscale="log")
    _save_fig(fig, out_png)


# ============================================================
# Aggregate plots across all seeds
# ============================================================

def plot_agg_rewards_boxplot(
    all_det: List[Dict],
    all_stoch: List[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    ordered: Dict[str, np.ndarray] = {}
    det_rewards = [r for d in all_det for r in d.get("rewards", [])]
    stoch_rewards = [r for d in all_stoch for r in d.get("rewards", [])]
    if det_rewards:
        ordered["PPO Det"] = np.asarray(det_rewards, dtype=float)
    if stoch_rewards:
        ordered["PPO Stoch"] = np.asarray(stoch_rewards, dtype=float)
    for name, d in baselines.items():
        arr = np.asarray(d.get("rewards", []), dtype=float)
        if arr.size:
            ordered[name] = arr

    if not ordered:
        return

    labels = list(ordered.keys())
    series = list(ordered.values())
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    bp = ax.boxplot(series, labels=labels, showmeans=True, patch_artist=True,
                    medianprops=dict(color="red", linewidth=2))
    colors_list = ["#2980b9", "#e67e22", "#e74c3c", "#f39c12", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors_list[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Episode reward", fontsize=11, fontweight="bold")
    ax.set_title("Aggregate Evaluation: Reward Distribution (all seeds)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)


def plot_agg_completion_obsolete(
    all_det: List[Dict],
    all_stoch: List[Dict],
    baselines: Dict[str, Dict],
    out_png: Path,
) -> None:
    methods: Dict[str, Dict] = {}
    det_comp = [c for d in all_det for c in d.get("completions", [])]
    det_obs = [o for d in all_det for o in d.get("obsolete", [])]
    stoch_comp = [c for d in all_stoch for c in d.get("completions", [])]
    stoch_obs = [o for d in all_stoch for o in d.get("obsolete", [])]
    if det_comp:
        methods["PPO Det"] = {"completions": det_comp, "obsolete": det_obs}
    if stoch_comp:
        methods["PPO Stoch"] = {"completions": stoch_comp, "obsolete": stoch_obs}
    methods.update(baselines)

    if not methods:
        return

    labels = list(methods.keys())
    comp_means = [float(np.mean(d.get("completions", [0]))) for d in methods.values()]
    obs_means = [float(np.mean(d.get("obsolete", [0]))) for d in methods.values()]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2), 5), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.bar(x - w / 2, comp_means, width=w, label="Completed (mean)", color="#27ae60", alpha=0.8)
    ax.bar(x + w / 2, obs_means, width=w, label="Obsolete (mean)", color="#c0392b", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count per episode", fontsize=11, fontweight="bold")
    ax.set_title("Aggregate Evaluation: Completions & Obsolete (all seeds)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, out_png)

def plot_training_noop_fraction(tb_data: Dict, out_png: Path, ma_window: int) -> None:
    """
    Plot the fraction of robot-actions that are NOOP (from TensorBoard tag policy/noop_fraction).

    This metric is logged by FinalTaskAllocationCallback if info["action_mask"] exists and
    actions are MultiDiscrete with NOOP assumed to be the last index (Kp1-1).
    """
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    ax.set_facecolor("#fafafa")

    s = tb_data.get("policy/noop_fraction")
    if s is None:
        ax.axis("off")
        ax.set_title(
            "Missing TensorBoard tag: policy/noop_fraction\n"
            "Make sure FinalTaskAllocationCallback is enabled and env info contains action_mask.",
            fontsize=14,
            fontweight="bold",
        )
        _save_fig(fig, out_png)
        return

    steps = np.asarray(s["steps"], dtype=float)
    vals = np.asarray(s["values"], dtype=float)

    ax.plot(steps, vals, "o", markersize=4, alpha=0.25, color="#34495e", label="Raw")
    ax.plot(steps, _moving_average(vals, ma_window), lw=2.8, color="#2c3e50",
            label=f"MA({ma_window})")
    ax.axhline(float(np.mean(vals)), color="#7f8c8d", lw=1.8, ls="--", alpha=0.65,
               label=f"Mean: {float(np.mean(vals)):.3f}")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("NOOP fraction (over robots)", fontsize=11, fontweight="bold")
    ax.set_title("Policy Behavior: NOOP Action Fraction", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, out_png)
# ============================================================
# Per-seed processing
# ============================================================

def process_seed(
    seed: int,
    seed_dir: Path,
    root_dir: Path,
    config: Dict,
    n_episodes: int,
    model_name: str,
    skip_eval: bool,
    ma_window: int,
    baseline_std: bool,
) -> tuple:
    """Run/load eval for one seed and produce all per-seed plots. Returns (det_result, stoch_result)."""
    model_path = seed_dir / f"{model_name}.zip"
    if not model_path.exists():
        print(f"[WARN] Model not found: {model_path} – skipping seed {seed}")
        return None, None

    plots_dir = seed_dir / "eval_plots_NOOPFixed"

    det_json = seed_dir / "eval_results_deterministic.json"
    stoch_json = seed_dir / "eval_results_stochastic.json"

    # --- Deterministic eval ---
    if skip_eval and det_json.exists():
        print(f"  [seed {seed}] Loading existing deterministic results from {det_json}")
        det_result = _load_json(det_json)
    else:
        print(f"\n{'=' * 70}")
        print(f"  [seed {seed}] Running deterministic evaluation ({n_episodes} episodes)")
        print(f"{'=' * 70}")
        det_result = run_evaluation(model_path, config, seed, n_episodes, deterministic=True)
        _save_json(det_result, det_json)

    # --- Stochastic eval ---
    if skip_eval and stoch_json.exists():
        print(f"  [seed {seed}] Loading existing stochastic results from {stoch_json}")
        stoch_result = _load_json(stoch_json)
    else:
        print(f"\n{'=' * 70}")
        print(f"  [seed {seed}] Running stochastic evaluation ({n_episodes} episodes)")
        print(f"{'=' * 70}")
        stoch_result = run_evaluation(model_path, config, seed, n_episodes, deterministic=False)
        _save_json(stoch_result, stoch_json)

    # --- Load baselines for this seed ---
    baselines_for_plot = load_baselines_all(root_dir)
    baselines_stats = load_baseline_stats_for_seed(root_dir, seed)

    # --- Evaluation plots ---
    print(f"\n  [seed {seed}] Generating evaluation plots → {plots_dir}")

    # plot_eval_rewards_per_episode(
    #     det_result, stoch_result,
    #     out_png=plots_dir / "eval_rewards_per_episode.png",
    #     ma_window=ma_window,
    # )
    plot_eval_rewards_per_episode(
    det_result,
    stoch_result,
    baselines_for_plot,
    out_png=plots_dir / "eval_rewards_per_episode.png",
    ma_window=ma_window,
)
    plot_eval_rewards_boxplot(
        det_result, stoch_result, baselines_for_plot,
        out_png=plots_dir / "eval_rewards_boxplot.png",
    )
    plot_eval_completion_obsolete(
        det_result, stoch_result, baselines_for_plot,
        out_png=plots_dir / "eval_completion_obsolete.png",
    )
    plot_eval_reward_components(
        det_result, stoch_result,
        out_png=plots_dir / "eval_reward_components.png",
    )
    
    

    # --- Training plots ---
    tb_dir = seed_dir / "tensorboard"
    if tb_dir.exists():
        print(f"  [seed {seed}] Generating training plots → {plots_dir}")
        tb_data = load_tensorboard_data(tb_dir)
        if tb_data:
            plot_training_rewards(
                tb_data, seed, baselines_stats,
                out_png=plots_dir / "training_rewards.png",
                ma_window=ma_window,
                baseline_std=baseline_std,
            )
            plot_training_entropy(
                tb_data,
                out_png=plots_dir / "training_entropy.png",
                ma_window=ma_window,
            )
            plot_training_value_overview(
                tb_data,
                out_png=plots_dir / "training_value_overview.png",
                ma_window=ma_window,
            )
            plot_training_value_loss(
                tb_data,
                out_png=plots_dir / "training_value_loss.png",
                ma_window=ma_window,
            )
            plot_training_noop_fraction(
            tb_data,
            out_png=plots_dir / "training_noop_fraction.png",
            ma_window=ma_window,
        )
    else:
        print(f"  [seed {seed}] No TensorBoard directory found at {tb_dir}; skipping training plots.")

    return det_result, stoch_result


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate PPO (det + stoch) and generate all training/evaluation plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", type=str, default="configs/training_config.yaml")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo",
                    help="Root directory containing seed_*/ sub-directories.")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Seeds to process. If omitted, all seed_*/ dirs are used.")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Number of evaluation episodes per seed per mode.")
    ap.add_argument("--model-name", type=str, default="ppo_final",
                    help="Model filename inside seed dir (without .zip).")
    ap.add_argument("--skip-eval", action="store_true",
                    help="If per-seed JSON results already exist, skip re-running eval.")
    ap.add_argument("--ma-window", type=int, default=20,
                    help="Moving-average window for training and per-episode eval plots.")
    ap.add_argument("--no-baseline-std", action="store_true",
                    help="Omit ±std dotted lines for baselines on training reward plots.")
    args = ap.parse_args()

    config = load_config(args.config)
    root_dir = Path(args.checkpoint_dir)

    # Discover seed directories
    if args.seeds:
        seed_dirs = [(s, root_dir / f"seed_{s}") for s in args.seeds]
    else:
        seed_dirs = [
            (int(d.name.replace("seed_", "")), d)
            for d in sorted(root_dir.glob("seed_*"))
        ]

    if not seed_dirs:
        print(f"[ERROR] No seed directories found in {root_dir}. Did you run train_ppo.py?")
        return

    print(f"Seeds to process: {[s for s, _ in seed_dirs]}")

    all_det_results: List[Dict] = []
    all_stoch_results: List[Dict] = []

    for seed, seed_dir in seed_dirs:
        det_res, stoch_res = process_seed(
            seed=seed,
            seed_dir=seed_dir,
            root_dir=root_dir,
            config=config,
            n_episodes=args.episodes,
            model_name=args.model_name,
            skip_eval=args.skip_eval,
            ma_window=args.ma_window,
            baseline_std=not args.no_baseline_std,
        )
        if det_res is not None:
            all_det_results.append(det_res)
        if stoch_res is not None:
            all_stoch_results.append(stoch_res)

    if not all_det_results and not all_stoch_results:
        print("[ERROR] No results collected – no models were found.")
        return

    # Save aggregated JSONs
    def _agg(results: List[Dict], deterministic: bool) -> Dict:
        agg_rewards = [r for d in results for r in d.get("rewards", [])]
        agg_comp = [c for d in results for c in d.get("completions", [])]
        agg_obs = [o for d in results for o in d.get("obsolete", [])]
        return {
            "n_seeds": len(results),
            "seeds": [d["seed"] for d in results],
            "episodes_per_seed": args.episodes,
            "deterministic": deterministic,
            "rewards": agg_rewards,
            "completions": agg_comp,
            "obsolete": agg_obs,
            "stats": {
                "reward_mean": float(np.mean(agg_rewards)) if agg_rewards else 0.0,
                "reward_std": float(np.std(agg_rewards)) if agg_rewards else 0.0,
                "completion_mean": float(np.mean(agg_comp)) if agg_comp else 0.0,
                "obsolete_mean": float(np.mean(agg_obs)) if agg_obs else 0.0,
            },
            "per_seed": results,
        }

    if all_det_results:
        _save_json(_agg(all_det_results, True), root_dir / "eval_results_all_seeds_deterministic.json")
    if all_stoch_results:
        _save_json(_agg(all_stoch_results, False), root_dir / "eval_results_all_seeds_stochastic.json")

    # Aggregate plots
    agg_plots_dir = root_dir / "eval_plotsNOOPFixed"
    baselines = load_baselines_all(root_dir)

    print(f"\nGenerating aggregate plots → {agg_plots_dir}")
    plot_agg_rewards_boxplot(
        all_det_results, all_stoch_results, baselines,
        out_png=agg_plots_dir / "agg_rewards_boxplot.png",
    )
    plot_agg_completion_obsolete(
        all_det_results, all_stoch_results, baselines,
        out_png=agg_plots_dir / "agg_completion_obsolete.png",
    )

    print(f"\n{'=' * 70}")
    print("✓ All evaluation and training plots generated.")
    print(f"{'=' * 70}")
    print(f"  Per-seed plots: {root_dir}/seed_N/eval_plots/")
    print(f"  Aggregate plots: {agg_plots_dir}/")
    print()


if __name__ == "__main__":
    main()