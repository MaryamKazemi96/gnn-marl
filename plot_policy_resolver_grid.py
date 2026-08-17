#!/usr/bin/env python3
"""
plot_policy_resolver_grid.py

Reproduces the "Policy/resolver comparison" grouped bar chart from the
reference repo's plot_metrics_wide.py (plot_resolver_cmp_grouped /
_apply_grouped_bar_plot) — x-axis = baseline policy, grouped bars =
conflict resolver, error bars = std across seeds, colorblind-safe palette
matching the reference's RESOLVER_COLOR_MAP.

Reads baseline results the same way compare_conflict_resolvers.py does:
each resolver's own runs/run_{id}/seed_*/eval_results/baseline_results_all.json,
aggregated across seeds.

Usage:
    python3 plot_policy_resolver_grid.py \\
        --resolver capacity:20260803_140000/capacity \\
        --resolver closest_than_capacity:20260803_140000/closest_than_capacity \\
        --resolver random:20260803_140000/random \\
        --resolver hungarian:20260803_140000/hungarian \\
        --resolver predicted_reward:20260803_140000/predicted_reward \\
        --resolver predicted_reward_joint:20260803_140000/predicted_reward_joint \\
        --output policy_resolver_comparison.png
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_eval import load_baselines_all

# Matches reference repo's RESOLVER_ORDER_BASE / RESOLVER_COLOR_MAP exactly
# (colorblind-safe palette) where names overlap; extra resolvers (e.g.
# hungarian_bids, which the reference doesn't have since it's PPO-only)
# get appended with additional colors from the same palette family.
RESOLVER_ORDER_BASE = ["capacity", "closest_than_capacity", "greedy", "random",
                        "predicted_reward", "predicted_reward_joint", "hungarian", "hungarian_bids"]
RESOLVER_COLOR_MAP = {
    "capacity": "#0072B2",
    "closest_then_capacity": "#56B4E9",  # kept for exact-name compatibility with the reference's spelling
    "closest_than_capacity": "#56B4E9",  # this repo's actual spelling
    "greedy": "#56B4E9",       # this repo's "greedy" (nearest-first) plays
                                # the same structural role as the
                                # reference's "closest_than_capacity"
    "random": "#F0E442",       # not one of the reference's 5 charted
                                # resolvers at all — given its own color
                                # rather than colliding with 'capacity'
    "predicted_reward": "#009E73",
    "predicted_reward_joint": "#E69F00",
    "hungarian": "#D55E00",
    "hungarian_bids": "#CC79A7",
}
GROUPED_BAR_WIDTH = 0.11


def parse_resolver_arg(s: str):
    if ":" not in s:
        raise ValueError(f"--resolver must be 'name:run_id', got {s!r}")
    name, run_id = s.split(":", 1)
    return name, run_id


def gather_baselines_for_resolver(run_id: str, runs_root: Path) -> dict:
    """{policy_name_upper: {"reward_mean": float, "reward_std": float}},
    aggregated across every seed_dir in this resolver's sweep."""
    run_root = runs_root / f"run_{run_id}"
    seed_dirs = sorted(run_root.glob("seed_*"))
    if not seed_dirs:
        print(f"⚠️  No seed_* directories found under {run_root}")
        return {}

    combined = {}
    for seed_dir in seed_dirs:
        baseline_file = seed_dir / "eval_results" / "baseline_results_all.json"
        seed_baselines = load_baselines_all(baseline_file)
        for pol, pdata in seed_baselines.items():
            dest = combined.setdefault(pol, {"rewards": []})
            dest["rewards"].extend(pdata.get("rewards", []))

    out = {}
    for pol, data in combined.items():
        rewards = data["rewards"]
        if not rewards:
            continue
        out[pol] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }
    return out


def ordered_resolvers(present: list) -> list:
    ordered = [r for r in RESOLVER_ORDER_BASE if r in present]
    ordered += [r for r in present if r not in ordered]
    return ordered


def resolver_color(name: str, fallback_palette: list, used: dict) -> str:
    if name in RESOLVER_COLOR_MAP:
        return RESOLVER_COLOR_MAP[name]
    for c in fallback_palette:
        if c not in used.values():
            used[name] = c
            return c
    return fallback_palette[len(used) % len(fallback_palette)]


def plot_grouped_bars(policy_data: dict, out_path: Path, title: str = "Reward by policy and resolver"):
    """policy_data: {resolver_name: {policy_name: {"reward_mean":, "reward_std":}}}"""
    all_policies = sorted({p for rdata in policy_data.values() for p in rdata})
    resolvers = ordered_resolvers(list(policy_data.keys()))

    # Fallback palette only used for resolver names NOT in RESOLVER_COLOR_MAP
    # (i.e. genuinely new/unknown resolvers) — every resolver this repo
    # currently supports has a fixed, semantically-consistent color from
    # the map above, so this only kicks in for future additions.
    colorblind_palette = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#F0E442", "#D55E00"]
    used_colors = {}
    colors = {r: resolver_color(r, colorblind_palette, used_colors) for r in resolvers}

    print("Resolver colors:")
    for resolver in resolvers:
        print(f"  {resolver}: {colors[resolver]}")

    positions = list(range(len(all_policies)))
    offsets = [(i - (len(resolvers) - 1) / 2.0) * GROUPED_BAR_WIDTH for i in range(len(resolvers))]

    fig_width = max(9.0, 0.9 * len(all_policies) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))

    for i, resolver in enumerate(resolvers):
        rdata = policy_data.get(resolver, {})
        means = [rdata.get(p, {}).get("reward_mean", math.nan) for p in all_policies]
        stds = [rdata.get(p, {}).get("reward_std", 0.0) for p in all_policies]
        shifted = [pos + offsets[i] for pos in positions]
        ax.bar(
            shifted, means, width=GROUPED_BAR_WIDTH, yerr=stds, capsize=2.5,
            label=resolver, color=colors[resolver], edgecolor="#2F3E4E",
            alpha=0.9, linewidth=0.6, error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(all_policies, rotation=43, ha="right")
    ax.set_ylabel("Reward", fontsize=11)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=14)
    ax.legend(title="resolver", fontsize=10, title_fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 0.93))

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolver", action="append", required=True,
                     help="name:run_id, repeatable. e.g. --resolver hungarian:20260803_140000/hungarian")
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--output", type=str, default="policy_resolver_comparison.png")
    ap.add_argument("--title", type=str, default="Reward by policy and resolver")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    pairs = [parse_resolver_arg(s) for s in args.resolver]

    policy_data = {}
    for name, run_id in pairs:
        print(f"Loading resolver={name} (run_id={run_id})...")
        baselines = gather_baselines_for_resolver(run_id, runs_root)
        policy_data[name] = baselines
        print(f"  policies found: {sorted(baselines.keys())}")

    plot_grouped_bars(policy_data, Path(args.output), title=args.title)


if __name__ == "__main__":
    main()