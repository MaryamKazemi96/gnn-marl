"""
src/utils/logit_metrics_logger.py

Per-episode logit/probability diagnostics logging.

Core schema (LogitStepMetrics / LogitEpisodeMetrics / compute_logit_step_metrics
/ aggregate_episode_logit_metrics / ensure_logit_metrics_log / append_logit_metrics_log
/ get_logit_metrics_header / logit_metrics_to_string) intentionally mirrors the
reference repo's utils/logit_metrics_logger.py field-for-field, so
training_logit_metrics.log from this repo and the reference repo are directly
comparable line-for-line.

Extension beyond the reference schema: compute_all_action_logit_stats /
RankLogitEpisodeMetrics track the FULL per-candidate-RANK logit distribution
(not just the single best candidate vs noop), for "logits of all actions"
plotting — see plot_training.py's plot_logit_ranks().
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# =========================================================================
# Core schema — mirrors the reference repo exactly
# =========================================================================

@dataclass
class LogitStepMetrics:
    """Metrics for a single decision step's logits."""
    step: int = 0
    best_cand_logit: float = 0.0
    noop_logit: float = 0.0
    margin: float = 0.0            # best_cand - noop
    top1_top2_margin: float = 0.0  # top1 - top2 (among candidates)
    num_valid_candidates: int = 0


@dataclass
class LogitEpisodeMetrics:
    """Aggregated logit metrics for an entire episode."""
    policy: str = ""   # "train" / "deterministic" / "stochastic"
    seed: int = 0
    ts: int = 0         # training timestep

    mean_best_cand_logit: float = 0.0
    mean_noop_logit: float = 0.0
    mean_margin: float = 0.0
    mean_top1_top2_margin: float = 0.0

    std_best_cand_logit: float = 0.0
    std_noop_logit: float = 0.0
    std_margin: float = 0.0
    std_top1_top2_margin: float = 0.0

    mean_num_valid_candidates: float = 0.0
    num_decision_steps: int = 0

    step_metrics: List[LogitStepMetrics] = field(default_factory=list)  # not written to CSV


def compute_logit_step_metrics(logits: np.ndarray, mask: np.ndarray, noop_logit_value: float) -> LogitStepMetrics:
    """
    logits: [R, K_max+1] (single env, batch dim already squeezed), last col = noop.
    mask:   [R, K_max+1] boolean, valid-slot mask.
    noop_logit_value: the scalar noop_logit parameter's current value.
    """
    metrics = LogitStepMetrics()

    cand_logits = logits[:, :-1]  # [R, K_max]
    cand_mask = mask[:, :-1]      # [R, K_max]

    valid_cand_logits = cand_logits[cand_mask.astype(bool)]
    if len(valid_cand_logits) > 0:
        metrics.best_cand_logit = float(np.max(valid_cand_logits))
        metrics.num_valid_candidates = len(valid_cand_logits)

        sorted_logits = np.sort(valid_cand_logits)[::-1]
        metrics.top1_top2_margin = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) >= 2 else 0.0
    else:
        metrics.best_cand_logit = float("-inf")
        metrics.num_valid_candidates = 0
        metrics.top1_top2_margin = 0.0

    metrics.noop_logit = noop_logit_value
    metrics.margin = metrics.best_cand_logit - metrics.noop_logit
    return metrics


def aggregate_episode_logit_metrics(step_metrics: List[LogitStepMetrics], policy: str, seed: int, ts: int) -> LogitEpisodeMetrics:
    ep_metrics = LogitEpisodeMetrics(policy=policy, seed=seed, ts=ts)
    ep_metrics.step_metrics = step_metrics
    ep_metrics.num_decision_steps = len(step_metrics)

    if len(step_metrics) == 0:
        return ep_metrics

    best_cand_logits = np.array([m.best_cand_logit for m in step_metrics if np.isfinite(m.best_cand_logit)])
    noop_logits = np.array([m.noop_logit for m in step_metrics])
    margins = np.array([m.margin for m in step_metrics if np.isfinite(m.margin)])
    top1_top2_margins = np.array([m.top1_top2_margin for m in step_metrics])
    num_valid_cands = np.array([m.num_valid_candidates for m in step_metrics])

    ep_metrics.mean_best_cand_logit = float(np.mean(best_cand_logits)) if len(best_cand_logits) > 0 else 0.0
    ep_metrics.mean_noop_logit = float(np.mean(noop_logits)) if len(noop_logits) > 0 else 0.0
    ep_metrics.mean_margin = float(np.mean(margins)) if len(margins) > 0 else 0.0
    ep_metrics.mean_top1_top2_margin = float(np.mean(top1_top2_margins)) if len(top1_top2_margins) > 0 else 0.0
    ep_metrics.mean_num_valid_candidates = float(np.mean(num_valid_cands)) if len(num_valid_cands) > 0 else 0.0

    ep_metrics.std_best_cand_logit = float(np.std(best_cand_logits)) if len(best_cand_logits) > 0 else 0.0
    ep_metrics.std_noop_logit = float(np.std(noop_logits)) if len(noop_logits) > 0 else 0.0
    ep_metrics.std_margin = float(np.std(margins)) if len(margins) > 0 else 0.0
    ep_metrics.std_top1_top2_margin = float(np.std(top1_top2_margins)) if len(top1_top2_margins) > 0 else 0.0

    return ep_metrics


def get_logit_metrics_header() -> str:
    return (
        "pol        seed      ts | "
        "best_cand  noop    margin  top1-2  | "
        "std_bcand  std_noop  std_marg  std_t12 | "
        "ncands  nsteps"
    )


def logit_metrics_to_string(metrics: LogitEpisodeMetrics) -> str:
    return (
        f"{metrics.policy:<10} {metrics.seed:>4} {metrics.ts:>8} | "
        f"{metrics.mean_best_cand_logit:>9.4f} {metrics.mean_noop_logit:>6.4f} "
        f"{metrics.mean_margin:>9.4f} {metrics.mean_top1_top2_margin:>7.4f} | "
        f"{metrics.std_best_cand_logit:>9.4f} {metrics.std_noop_logit:>9.4f} "
        f"{metrics.std_margin:>9.4f} {metrics.std_top1_top2_margin:>8.4f} | "
        f"{metrics.mean_num_valid_candidates:>6.2f} {metrics.num_decision_steps:>7}"
    )


def ensure_logit_metrics_log(log_path: str, overwrite: bool = False) -> None:
    if overwrite or not os.path.exists(log_path):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            f.write(get_logit_metrics_header() + "\n")


def append_logit_metrics_log(log_path: str, metrics: LogitEpisodeMetrics) -> None:
    ensure_logit_metrics_log(log_path)
    with open(log_path, "a") as f:
        f.write(logit_metrics_to_string(metrics) + "\n")


# =========================================================================
# Extension: full per-candidate-RANK logit distribution ("all actions")
# =========================================================================

@dataclass
class RankLogitEpisodeMetrics:
    """Per-episode mean logit for each candidate rank (0=best, 1=2nd-best,
    ...) plus noop, for plotting the whole action-logit distribution over
    training rather than just the best-candidate-vs-noop margin."""
    policy: str = "train"
    seed: int = 0
    ts: int = 0
    episode: int = 0
    K_max: int = 0
    noop_logit_mean: Optional[float] = None
    rank_logit_means: List[Optional[float]] = field(default_factory=list)  # length K_max
    rank_counts: List[int] = field(default_factory=list)                  # length K_max


def aggregate_episode_rank_logit_metrics(
    step_stats: List[dict], policy: str, seed: int, ts: int, episode: int, K_max: int,
) -> RankLogitEpisodeMetrics:
    """step_stats: list of dicts as returned by
    compute_all_action_logit_stats() (one per training step this episode)."""
    out = RankLogitEpisodeMetrics(policy=policy, seed=seed, ts=ts, episode=episode, K_max=K_max)

    valid = [s for s in step_stats if s is not None]
    if not valid:
        out.rank_logit_means = [None] * K_max
        out.rank_counts = [0] * K_max
        return out

    noop_vals = [s["noop_logit_mean"] for s in valid if s["noop_logit_mean"] is not None]
    out.noop_logit_mean = float(np.mean(noop_vals)) if noop_vals else None

    rank_means = []
    rank_counts = []
    for rank in range(K_max):
        vals, weights = [], []
        for s in valid:
            v = s["rank_logit_means"][rank]
            c = s["rank_counts"][rank]
            if v is not None and c > 0:
                vals.append(v)
                weights.append(c)
        if vals:
            rank_means.append(float(np.average(vals, weights=weights)))
            rank_counts.append(int(sum(weights)))
        else:
            rank_means.append(None)
            rank_counts.append(0)

    out.rank_logit_means = rank_means
    out.rank_counts = rank_counts
    return out


def _rank_fields(K_max: int) -> List[str]:
    fields = ["policy", "seed", "ts", "episode", "K_max", "noop_logit_mean"]
    fields += [f"rank_{i}_logit_mean" for i in range(K_max)]
    fields += [f"rank_{i}_count" for i in range(K_max)]
    return fields


def ensure_rank_logit_metrics_log(path: str, K_max: int, overwrite: bool = True) -> None:
    if not overwrite and os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_rank_fields(K_max))
        writer.writeheader()


def append_rank_logit_metrics_log(path: str, metrics: RankLogitEpisodeMetrics) -> None:
    fields = _rank_fields(metrics.K_max)
    row = {
        "policy": metrics.policy, "seed": metrics.seed, "ts": metrics.ts,
        "episode": metrics.episode, "K_max": metrics.K_max,
        "noop_logit_mean": metrics.noop_logit_mean,
    }
    for i in range(metrics.K_max):
        row[f"rank_{i}_logit_mean"] = metrics.rank_logit_means[i]
        row[f"rank_{i}_count"] = metrics.rank_counts[i]

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)