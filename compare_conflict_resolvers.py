#!/usr/bin/env python3
"""
compare_conflict_resolvers.py

Compare PPO (deterministic + stochastic) and baselines across several
conflict-resolution sweeps (greedy / random / hungarian / hungarian_bids),
each trained as its own runs/run_{id}/ sweep with everything else held
identical.

Produces:
  - conflict_resolver_comparison/reward_comparison.png
      grouped bars: PPO Det, PPO Stoch, + each baseline, one group per
      resolver. PPO error bars are std ACROSS SEEDS within that resolver's
      sweep; baseline error bars are std across that baseline's own
      episodes (baselines aren't seed-dependent the way PPO training is,
      but they ARE resolver-dependent, since they go through the same
      environment's conflict resolver as PPO).
  - conflict_resolver_comparison/metrics_table.png
      per-resolver table: reward (det/sto), completed, conflict_drop_rate,
      noop_frac, chosen_noop_rate_when_available — the numbers that
      actually explain WHY one resolver wins, not just that it does.
  - conflict_resolver_comparison/summary.json
      raw numbers behind both plots, for anything else you want to compute.

Usage:
    python3 compare_conflict_resolvers.py \\
        --resolver greedy:20260803_100000 \\
        --resolver random:20260803_110000 \\
        --resolver hungarian:20260803_120000 \\
        --resolver hungarian_bids:20260803_130000

Each resolver's baselines are read from every seed_dir's own
runs/run_{id}/seed_{n}/eval_results/baseline_results_all.json and
aggregated across seeds — matching where eval_baseline.py actually writes
them, and mirroring how PPO's own cross-seed stats are aggregated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_eval import load_all_seed_eval_stats, load_baselines_all, _save_fig


def parse_resolver_arg(s: str):
    if ":" not in s:
        raise ValueError(f"--resolver must be 'name:run_id', got {s!r}")
    name, run_id = s.split(":", 1)
    return name, run_id


def gather_resolver_data(name: str, run_id: str, runs_root: Path, baseline_dir_template: str = None):
    """baseline_dir_template is accepted for backward compatibility but
    unused by default — eval_baseline.py actually writes
    baseline_results_all.json into EACH seed's own
    runs/run_{id}/seed_{n}/eval_results/, not a separate top-level
    directory, so baselines are aggregated across the same seed_dirs PPO
    stats come from, exactly like PPO's own cross-seed aggregation."""
    run_root = runs_root / f"run_{run_id}"
    seed_stats = load_all_seed_eval_stats(run_root)
    if not seed_stats:
        print(f"⚠️  No per-seed eval results found for resolver={name} (run_id={run_id}) "
              f"under {run_root} — run eval_ppo.py --all-seeds --run-id {run_id} first.")

    combined_baselines = {}  # {policy_name_upper: {"rewards": [...], "completed": [...], "obsolete": [...]}}
    seed_dirs = sorted(run_root.glob("seed_*"))
    found_any = False
    for seed_dir in seed_dirs:
        baseline_file = seed_dir / "eval_results" / "baseline_results_all.json"
        seed_baselines = load_baselines_all(baseline_file)
        if seed_baselines:
            found_any = True
        for pol, pdata in seed_baselines.items():
            dest = combined_baselines.setdefault(pol, {"rewards": [], "completed": [], "obsolete": []})
            dest["rewards"].extend(pdata.get("rewards", []))
            dest["completed"].extend(pdata.get("completed", []))
            dest["obsolete"].extend(pdata.get("obsolete", []))

    if not found_any:
        print(f"⚠️  No baseline results found for resolver={name} under any seed_dir in {run_root} — "
              f"run eval_baseline.py --conflict-resolution {name} --all-seeds --run-id {run_id} first.")

    return {"run_root": run_root, "seed_stats": seed_stats, "baselines": combined_baselines}


def summarize_resolver(name: str, data: dict) -> dict:
    """Collapse one resolver's per-seed stats + baselines into the flat
    numbers used by both the bar chart and the table."""
    seed_stats = data["seed_stats"]
    baselines = data["baselines"]

    det_means = [s["det"]["stats"]["reward_mean"] for s in seed_stats.values() if s.get("det", {}).get("stats")]
    sto_means = [s["sto"]["stats"]["reward_mean"] for s in seed_stats.values() if s.get("sto", {}).get("stats")]

    def stat_mean(key, side):
        vals = [s[side]["stats"].get(key) for s in seed_stats.values()
                 if s.get(side, {}).get("stats", {}).get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "resolver": name,
        "n_seeds": len(seed_stats),
        "ppo_det_reward_mean": float(np.mean(det_means)) if det_means else None,
        "ppo_det_reward_std_across_seeds": float(np.std(det_means)) if det_means else None,
        "ppo_sto_reward_mean": float(np.mean(sto_means)) if sto_means else None,
        "ppo_sto_reward_std_across_seeds": float(np.std(sto_means)) if sto_means else None,
        "ppo_det_completed_mean": stat_mean("completed", "det"),
        "ppo_det_conflict_drop_rate_mean": stat_mean("conflict_drop_rate", "det"),
        "ppo_det_noop_frac_mean": stat_mean("noop_frac_mean", "det"),
        "ppo_det_chosen_noop_rate_when_available_mean": stat_mean("chosen_noop_rate_when_available", "det"),
        "baselines": {},
    }

    for bname, bdata in baselines.items():
        rewards = bdata.get("rewards", [])
        if not rewards:
            continue
        summary["baselines"][bname] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }

    return summary


def plot_reward_comparison(summaries: list, out_png: Path):
    resolver_names = [s["resolver"] for s in summaries]
    baseline_names = sorted({b for s in summaries for b in s["baselines"].keys()})

    series_names = ["PPO Det", "PPO Stoch"] + baseline_names
    colors = ["#1f77b4", "#2ca02c"] + ["#888888", "#aaaaaa", "#cccccc"][:len(baseline_names)]

    x = np.arange(len(resolver_names))
    width = 0.8 / len(series_names)

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(resolver_names)), 6), facecolor="white")

    for i, series in enumerate(series_names):
        means, stds = [], []
        for s in summaries:
            if series == "PPO Det":
                means.append(s["ppo_det_reward_mean"] or 0.0)
                stds.append(s["ppo_det_reward_std_across_seeds"] or 0.0)
            elif series == "PPO Stoch":
                means.append(s["ppo_sto_reward_mean"] or 0.0)
                stds.append(s["ppo_sto_reward_std_across_seeds"] or 0.0)
            else:
                b = s["baselines"].get(series)
                means.append(b["reward_mean"] if b else 0.0)
                stds.append(b["reward_std"] if b else 0.0)

        offset = (i - (len(series_names) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=series, color=colors[i], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(resolver_names, fontsize=10)
    ax.set_ylabel("Episode reward", fontsize=11, fontweight="bold")
    ax.set_title("Reward by Conflict Resolution Mechanism\n"
                  "(PPO error bars = std across seeds, baseline error bars = std across episodes)",
                  fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, out_png)


def plot_metrics_table(summaries: list, out_png: Path):
    cols = ["Resolver", "PPO Det", "PPO Stoch", "Completed", "Conflict Drop", "Noop Frac", "Chosen-Noop|Avail"]
    rows = []
    for s in summaries:
        rows.append([
            s["resolver"],
            f"{s['ppo_det_reward_mean']:.2f}" if s["ppo_det_reward_mean"] is not None else "—",
            f"{s['ppo_sto_reward_mean']:.2f}" if s["ppo_sto_reward_mean"] is not None else "—",
            f"{s['ppo_det_completed_mean']:.2f}" if s["ppo_det_completed_mean"] is not None else "—",
            f"{s['ppo_det_conflict_drop_rate_mean']:.4f}" if s["ppo_det_conflict_drop_rate_mean"] is not None else "—",
            f"{s['ppo_det_noop_frac_mean']:.4f}" if s["ppo_det_noop_frac_mean"] is not None else "—",
            f"{s['ppo_det_chosen_noop_rate_when_available_mean']:.4f}" if s["ppo_det_chosen_noop_rate_when_available_mean"] is not None else "—",
        ])

    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.5 * len(rows)), facecolor="white")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.5)

    for i in range(len(cols)):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("Conflict Resolution — Metrics by Mechanism", fontsize=13, fontweight="bold", pad=14)
    _save_fig(fig, out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolver", action="append", required=True,
                     help="name:run_id, repeatable. e.g. --resolver greedy:20260803_100000")
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--output-dir", type=str, default="conflict_resolver_comparison")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [parse_resolver_arg(s) for s in args.resolver]

    summaries = []
    for name, run_id in pairs:
        print(f"Loading resolver={name} (run_id={run_id})...")
        data = gather_resolver_data(name, run_id, runs_root)
        summary = summarize_resolver(name, data)
        summaries.append(summary)
        print(f"  n_seeds={summary['n_seeds']}  det={summary['ppo_det_reward_mean']}  "
              f"sto={summary['ppo_sto_reward_mean']}")

    plot_reward_comparison(summaries, out_dir / "reward_comparison.png")
    plot_metrics_table(summaries, out_dir / "metrics_table.png")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    print(f"\n✓ Comparison saved to {out_dir}/")


if __name__ == "__main__":
    main()