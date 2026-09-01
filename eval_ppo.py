
# """
# Evaluate trained GNN-PPO model (deterministic + stochastic)
# Clean + robust + debug metrics
# """
 
# import argparse
# import json
# import random
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple 
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
 
# import numpy as np
# import torch
# import yaml
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
 
# from src.environment.environment import MultiAgentTaskEnv
# from src.models.sb3_gnn_policy import RTGNNPolicy
# from train_ppo import get_eval_seeds
 
 
# # =========================================================
# # Utils
# # =========================================================
 
# def load_json(p: Path):
#     return json.loads(p.read_text())
 
 
# def save_json(data, p: Path):
#     p.parent.mkdir(parents=True, exist_ok=True)
#     p.write_text(json.dumps(data, indent=2))
#     print(f"✓ {p}")
 
 
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
 
 
# def load_config(path: str):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)
 
 
# def latest_run_id(runs_root: Path) -> Path:
#     """Most recently modified runs/run_{id}/ sweep folder."""
#     run_dirs = sorted(runs_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
#     if not run_dirs:
#         raise FileNotFoundError(f"No run_* directories found in {runs_root}")
#     return run_dirs[-1]
 
 
# def find_latest_run(seed: int, run_id: str = None, runs_root: str = "runs") -> Path:
#     """Resolve runs/run_{run_id}/seed_{seed}/ — layout as of the sweep-grouped
#     restructure (previously runs/seed_{seed}/run_{id}/, flipped so every seed
#     trained in one sweep lives together under a single run_id)."""
#     root = Path(runs_root)
#     run_root = (root / f"run_{run_id}") if run_id else latest_run_id(root)
 
#     seed_dir = run_root / f"seed_{seed}"
#     if not seed_dir.exists():
#         available = sorted(p.name for p in run_root.glob("seed_*"))
#         raise FileNotFoundError(f"No seed_{seed} under {run_root}. Available: {available}")
#     return seed_dir
 
 
# def all_seed_dirs_in_run(run_id: str = None, runs_root: str = "runs") -> List[Path]:
#     """Every seed_* directory trained in one sweep — used for --all-seeds eval."""
#     root = Path(runs_root)
#     run_root = (root / f"run_{run_id}") if run_id else latest_run_id(root)
#     seed_dirs = sorted(run_root.glob("seed_*"))
#     if not seed_dirs:
#         raise FileNotFoundError(f"No seed_* directories found in {run_root}")
#     return seed_dirs
 
 
# # =========================================================
# # Data
# # =========================================================
 
# def load_data(data_dir: str):
#     p = Path(data_dir)
#     agents = np.load(p / "agents.npy", allow_pickle=True)
 
#     tasks = []
#     i = 0
#     while (p / f"tasks_batch_{i}.npy").exists():
#         tasks.append(np.load(p / f"tasks_batch_{i}.npy", allow_pickle=True))
#         i += 1
 
#     if len(tasks) == 0:
#         raise FileNotFoundError(f"No task batches found in {p}")
 
#     return agents, tasks
 
 
# def make_env(agents, tasks, config, seed):
#     def _init():
#         env = MultiAgentTaskEnv(
#             agents=agents,
#             tasks_batches=tasks,
#             K_max=config["K_max"],
#             N_max=config["N_max"],
#             E_max=config["E_max"],
#             use_xy_pickup=config.get("use_xy_pickup", False),
#             normalize_features=config.get("normalize_features", True),
#             use_node_type=config.get("use_node_type", True),
#             use_ego_robot=config.get("use_ego_robot", True),
#             use_edge_rt=config.get("use_edge_rt", False),
#             edge_features=config.get("edge_features"),
#             two_hop=config.get("two_hop", False),
#             two_hop_directed=config.get("two_hop_directed", False),
#             vicinity_m=config.get("vicinity_m", 20.0),
#             max_steps=config.get("max_steps", 1000),
#             max_robot_capacity=config.get("max_robot_capacity", 2),
#             max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
#             max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
#             decision_interval=config.get("decision_interval", 8),
#             movement_speed=config.get("movement_speed", 1.0),
#             capacity_method=config.get("capacity_method", "assigned"),
#             W_COMP=config.get("W_COMP", 2.0),
#             W_WAIT=config.get("W_WAIT", 1.0),
#             W_DEADLINE=config.get("W_DEADLINE", 10.0),
#             W_OBS=config.get("W_OBS", 1.0),
#             conflict_resolution=config.get("conflict_resolution", "greedy"),
#             candidates_sorting=config.get("candidates_sorting", "distance"),
#         )
#         env.reset(seed=seed)
#         return env
 
#     return DummyVecEnv([_init])
 
 
# # =========================================================
# # Model
# # =========================================================
 
# def pick_model(run_dir: Path, prefer_best: bool = True) -> Path:
#     # Prefer best_model.zip — written by train_ppo.py's EvalCallback,
#     # tracking whichever checkpoint scored highest on periodic
#     # deterministic evaluation DURING training, not just whatever the
#     # policy happened to be at the very last timestep. ppo_final.zip can
#     # be worse than an earlier checkpoint if the policy dipped mid-training
#     # and only partially recovered (this is exactly what happened to one
#     # MLP seed — see the noop-collapse diagnosis this same evaluation
#     # pipeline surfaced). Set prefer_best=False to force the old
#     # final-model-only behavior, e.g. for reproducing/debugging older runs.
#     if prefer_best:
#         best = run_dir / "models" / "best_model.zip"
#         if best.exists():
#             print("Using best checkpoint (from EvalCallback):", best)
#             return best
#         print(f"⚠️  No best_model.zip found at {best} — this run may predate "
#               f"the best-checkpoint-tracking EvalCallback, or it was disabled. "
#               f"Falling back to ppo_final.zip.")

#     # prefer ppo_final
#     final = run_dir / "ppo_final.zip"
#     print("Looking for final model:", final)
#     if final.exists():
#         return final
 
#     # then checkpoints
#     ckpt_dir = run_dir / "models"
#     if ckpt_dir.exists():
#         ckpts = list(ckpt_dir.glob("model_episode*_ts*.zip"))
#         if ckpts:
#             import re
#             def ts(p):
#                 m = re.search(r"_ts(\d+)\.zip$", p.name)
#                 return int(m.group(1)) if m else -1
#             return max(ckpts, key=ts)
 
#     # fallback
#     models = list(run_dir.glob("*.zip"))
#     if not models:
#         raise FileNotFoundError(f"No models found in {run_dir}")
#     return max(models, key=lambda p: p.stat().st_mtime)
 
 
# # =========================================================
# # Evaluation core
# # =========================================================
 
# def _stats_from_raw(raw: dict) -> dict:
#     """Compute the 'stats' summary dict from raw per-episode arrays — shared
#     by run_eval() (single eval seed) and run_eval_across_seeds() (pooled
#     across multiple eval seeds), so both produce the exact same stats
#     shape/semantics regardless of how many seeds' episodes went into it."""
#     rewards = raw["rewards"]
#     r = np.array(rewards, dtype=float)
#     completed = raw["completed"]
#     obsolete = raw["obsolete"]
#     noop_fractions = raw["noop_fractions"]
#     ticks = raw["ticks"]
#     ep_mask_zeros = raw["ep_mask_zero_count"]
#     ep_r_comp = raw["ep_r_comp"]
#     ep_r_wait = raw["ep_r_wait"]
#     ep_r_deadline = raw["ep_r_deadline"]
#     ep_r_obsolete = raw["ep_r_obsolete"]
#     ep_invalids = raw["ep_invalid_action_count"]
#     ep_totals = raw["ep_total_action_count"]
#     ep_conflicts = raw["ep_conflict_dropped_count"]
#     ep_capacity_rej = raw["ep_capacity_rejected_count"]
#     ep_noop_forced = raw["ep_noop_forced_count"]
#     ep_noop_chosen = raw["ep_noop_chosen_count"]
#     ep_had_candidates = raw["ep_had_candidates_count"]
#     ep_decisions = raw["ep_decisions_total"]

#     total_actions_all = int(np.sum(ep_totals)) if ep_totals else 0
#     invalid_all = int(np.sum(ep_invalids)) if ep_invalids else 0
#     conflict_all = int(np.sum(ep_conflicts)) if ep_conflicts else 0
#     caprej_all = int(np.sum(ep_capacity_rej)) if ep_capacity_rej else 0

#     invalid_rate = (invalid_all / total_actions_all) if total_actions_all > 0 else 0.0
#     conflict_rate = (conflict_all / total_actions_all) if total_actions_all > 0 else 0.0
#     caprej_rate = (caprej_all / total_actions_all) if total_actions_all > 0 else 0.0

#     decisions_all      = int(np.sum(ep_decisions)) if ep_decisions else 0
#     noop_forced_all    = int(np.sum(ep_noop_forced)) if ep_noop_forced else 0
#     noop_chosen_all    = int(np.sum(ep_noop_chosen)) if ep_noop_chosen else 0
#     had_candidates_all = int(np.sum(ep_had_candidates)) if ep_had_candidates else 0

#     noop_frac_forced_rate = (noop_forced_all / decisions_all) if decisions_all > 0 else 0.0
#     noop_frac_chosen_rate = (noop_chosen_all / decisions_all) if decisions_all > 0 else 0.0
#     chosen_noop_rate_when_available = (
#         noop_chosen_all / had_candidates_all if had_candidates_all > 0 else 0.0
#     )

#     return {
#         "reward_mean": float(r.mean()) if len(r) else 0.0,
#         "reward_std": float(r.std()) if len(r) else 0.0,
#         "min": float(r.min()) if len(r) else 0.0,
#         "max": float(r.max()) if len(r) else 0.0,
#         "completed": float(np.mean(completed)) if completed else 0.0,
#         "obsolete": float(np.mean(obsolete)) if obsolete else 0.0,
#         "noop_frac_mean": float(np.mean(noop_fractions)) if noop_fractions else 0.0,
#         "ticks_mean": float(np.mean(ticks)) if ticks else 0.0,

#         "invalid_action_total": invalid_all,
#         "total_action_count": total_actions_all,
#         "invalid_action_rate": float(invalid_rate),
#         "conflict_drop_rate": float(conflict_rate),
#         "capacity_reject_rate": float(caprej_rate),
#         "mask_zero_mean": float(np.mean(ep_mask_zeros)) if ep_mask_zeros else 0.0,

#         "r_comp_mean": float(np.mean(ep_r_comp)) if ep_r_comp else 0.0,
#         "r_wait_mean": float(np.mean(ep_r_wait)) if ep_r_wait else 0.0,
#         "r_deadline_mean": float(np.mean(ep_r_deadline)) if ep_r_deadline else 0.0,
#         "r_obsolete_mean": float(np.mean(ep_r_obsolete)) if ep_r_obsolete else 0.0,

#         "noop_frac_forced": float(noop_frac_forced_rate),
#         "noop_frac_chosen": float(noop_frac_chosen_rate),
#         "chosen_noop_rate_when_available": float(chosen_noop_rate_when_available),
#         "decisions_total": decisions_all,
#         "had_candidates_total": had_candidates_all,
#     }


# def _merge_raw(raw_list: List[dict]) -> dict:
#     """Concatenate every list-valued field across multiple run_eval() raw
#     outputs (one per eval seed) — used to pool episodes from every eval
#     seed together before computing stats, so reward_mean/std reflect the
#     FULL eval_seeds pool, not one seed's episodes averaged with another's
#     already-averaged summary."""
#     if not raw_list:
#         raise ValueError("_merge_raw got an empty list")
#     merged = {}
#     for key in raw_list[0].keys():
#         merged[key] = []
#         for raw in raw_list:
#             merged[key].extend(raw[key])
#     return merged


# def run_eval_across_seeds(model, agents, tasks, config, eval_seeds: List[int],
#                            episodes_per_seed: int, deterministic: bool):
#     """Runs run_eval() once per eval seed (genuinely different environment
#     randomization each time — see get_eval_seeds() in train_ppo.py), pools
#     every episode from every eval seed together, and computes stats over
#     that pooled set. This is what 'reward_mean/reward_std' now actually
#     means for a single trained model's evaluation: mean/std across
#     len(eval_seeds) * episodes_per_seed episodes, not just one arbitrary
#     eval seed's episodes."""
#     raw_per_seed = []
#     for es in eval_seeds:
#         env = make_env(agents, tasks, config, es)
#         if config.get("conflict_resolution") == "hungarian_bids":
#             model.policy._bid_env = env
#         raw = run_eval(model, env, episodes_per_seed, deterministic)
#         env.close()
#         raw["eval_seed"] = [es] * episodes_per_seed  # tag each episode with which eval seed produced it
#         raw_per_seed.append(raw)

#     merged = _merge_raw(raw_per_seed)
#     merged["stats"] = _stats_from_raw(merged)
#     merged["stats"]["eval_seeds_used"] = list(eval_seeds)
#     merged["stats"]["episodes_per_eval_seed"] = episodes_per_seed
#     return merged


# def run_eval(model, env, episodes, deterministic):
#     rewards, lengths, ticks, completed, obsolete = [], [], [], [], []
#     noop_fractions, action_hists = [], []
 
#     # debug episode aggregates
#     ep_invalids, ep_totals, ep_valids = [], [], []
#     ep_conflicts, ep_capacity_rej, ep_mask_zeros = [], [], []
#     ep_r_comp, ep_r_wait, ep_r_deadline, ep_r_obsolete = [], [], [], []
#     ep_noop_forced, ep_noop_chosen, ep_had_candidates, ep_decisions = [], [], [], [] 
 
#     K_max = env.get_attr("action_space")[0].nvec[0] - 1  # noop index
 
#     for ep in range(episodes):
#         obs = env.reset()
#         done = [False]
 
#         ep_r, ep_l = 0.0, 0
#         ep_c, ep_o = 0, 0
#         ep_actions = []
#         ep_time = 0.0
 
#         # per-episode debug accumulators
#         inv_sum, total_sum, valid_sum = 0, 0, 0
#         conflict_sum, caprej_sum, maskz_sum = 0, 0, 0
#         rcomp_sum, rwait_sum, rdead_sum, robs_sum = 0.0, 0.0, 0.0, 0.0
#         noop_forced_sum, noop_chosen_sum, had_cand_sum, decisions_sum = 0, 0, 0, 0
 
#         while not done[0]:
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, r, dones, infos = env.step(action)
#             done = dones
 
#             ep_r += float(r[0])
#             ep_l += 1
#             ep_actions.extend(np.asarray(action).flatten().tolist())
 
#             info = infos[0] if isinstance(infos, (list, tuple)) else infos
#             if isinstance(info, dict):
#                 ep_c = info.get("completed_count", ep_c)
#                 ep_o = info.get("obsolete_count", ep_o)
#                 ep_time = info.get("time", ep_time)
 
#                 inv_sum += int(info.get("invalid_action_count", 0))
#                 total_sum += int(info.get("total_action_count", 0))
#                 valid_sum += int(info.get("valid_action_count", 0))
#                 conflict_sum += int(info.get("conflict_dropped_count", 0))
#                 caprej_sum += int(info.get("capacity_rejected_count", 0))
#                 maskz_sum += int(info.get("mask_zero_count", 0))
 
#                 rcomp_sum += float(info.get("r_comp", 0.0))
#                 rwait_sum += float(info.get("r_wait", 0.0))
#                 rdead_sum += float(info.get("r_deadline", 0.0))
#                 robs_sum += float(info.get("r_obsolete", 0.0))
 
#                 noop_forced_sum += int(info.get("noop_forced_count", 0))
#                 noop_chosen_sum += int(info.get("noop_chosen_count", 0))
#                 had_cand_sum    += int(info.get("had_candidates_count", 0))
#                 decisions_sum   += int(info.get("decisions_total", 0))
 
#         rewards.append(ep_r)
#         lengths.append(ep_l)
#         ticks.append(ep_time)
#         completed.append(ep_c)
#         obsolete.append(ep_o)
 
#         actions_arr = np.asarray(ep_actions)
#         noop_frac = float((actions_arr == K_max).mean()) if actions_arr.size else 0.0
#         noop_fractions.append(noop_frac)
#         hist = np.bincount(actions_arr, minlength=K_max + 1).tolist() if actions_arr.size else []
#         action_hists.append(hist)
 
#         ep_invalids.append(inv_sum)
#         ep_totals.append(total_sum)
#         ep_valids.append(valid_sum)
#         ep_conflicts.append(conflict_sum)
#         ep_capacity_rej.append(caprej_sum)
#         ep_mask_zeros.append(maskz_sum)
 
#         ep_r_comp.append(rcomp_sum)
#         ep_r_wait.append(rwait_sum)
#         ep_r_deadline.append(rdead_sum)
#         ep_r_obsolete.append(robs_sum)
 
#         ep_noop_forced.append(noop_forced_sum)
#         ep_noop_chosen.append(noop_chosen_sum)
#         ep_had_candidates.append(had_cand_sum)
#         ep_decisions.append(decisions_sum)
 
#     raw = {
#         "rewards": rewards,
#         "lengths": lengths,
#         "ticks": ticks,
#         "completed": completed,
#         "obsolete": obsolete,
#         "noop_fractions": noop_fractions,
#         "action_hists": action_hists,

#         "ep_invalid_action_count": ep_invalids,
#         "ep_total_action_count": ep_totals,
#         "ep_valid_action_count": ep_valids,
#         "ep_conflict_dropped_count": ep_conflicts,
#         "ep_capacity_rejected_count": ep_capacity_rej,
#         "ep_mask_zero_count": ep_mask_zeros,

#         "ep_r_comp": ep_r_comp,
#         "ep_r_wait": ep_r_wait,
#         "ep_r_deadline": ep_r_deadline,
#         "ep_r_obsolete": ep_r_obsolete,

#         "ep_noop_forced_count": ep_noop_forced,
#         "ep_noop_chosen_count": ep_noop_chosen,
#         "ep_had_candidates_count": ep_had_candidates,
#         "ep_decisions_total": ep_decisions,
#     }
#     raw["stats"] = _stats_from_raw(raw)
#     return raw
 
 
# # =========================================================
# # Main
# # =========================================================
 
# # =========================================================
# # Main
# # =========================================================
 
# def evaluate_one_seed(seed: int, run_dir: Path, config: Dict, agents, tasks,
#                        episodes_per_eval_seed: int, output_override: str = None,
#                        eval_seeds: Optional[List[int]] = None):
#     """seed here is the TRAIN seed (which trained model to load — see
#     run_dir). eval_seeds is the SEPARATE pool used for environment
#     randomization during evaluation (get_eval_seeds() in train_ppo.py) —
#     every configured eval seed gets episodes_per_eval_seed episodes, and
#     ALL of them get pooled together into one reward_mean/reward_std, for
#     both deterministic and stochastic readout. Falls back to [seed] if no
#     eval_seeds are given, matching the old single-seed behavior."""
#     if eval_seeds is None:
#         eval_seeds = [seed]

#     print(f"\n---- train seed {seed} | eval seeds {eval_seeds} ----")
#     print(" Selected run:", run_dir)
 
#     model_path = pick_model(run_dir)
#     print("Using model:", model_path)
 
#     device = "cuda" if torch.cuda.is_available() else "cpu"
 
#     model = PPO.load(
#         str(model_path),
#         device=device,
#         custom_objects={"policy_class": RTGNNPolicy},
#     )
#     print("✓ Loaded model\n")
 
#     print(f"Running deterministic across {len(eval_seeds)} eval seed(s)...")
#     det = run_eval_across_seeds(model, agents, tasks, config, eval_seeds, episodes_per_eval_seed, True)
 
#     print(f"Running stochastic across {len(eval_seeds)} eval seed(s)...")
#     sto = run_eval_across_seeds(model, agents, tasks, config, eval_seeds, episodes_per_eval_seed, False)
 
#     out = Path(output_override) if output_override else (run_dir / "eval_results")
#     out.mkdir(parents=True, exist_ok=True)
 
#     save_json(det, out / "deterministic.json")
#     save_json(sto, out / "stochastic.json")
 
#     debug_summary = {
#         "deterministic": det["stats"],
#         "stochastic": sto["stats"],
#     }
#     save_json(debug_summary, out / "debug_summary.json")
 
#     print("\n============================")
#     print(f"RESULTS — train seed {seed}  (pooled across eval seeds {eval_seeds})")
#     print("============================")
#     print(f"Deterministic: {det['stats']['reward_mean']:.2f} ± {det['stats']['reward_std']:.2f}  "
#           f"(n={len(det['rewards'])} episodes)")
#     print(f"Stochastic:    {sto['stats']['reward_mean']:.2f} ± {sto['stats']['reward_std']:.2f}  "
#           f"(n={len(sto['rewards'])} episodes)")
#     print(f"Det invalid rate: {det['stats']['invalid_action_rate']:.4f} | caprej: {det['stats']['capacity_reject_rate']:.4f} | conflict: {det['stats']['conflict_drop_rate']:.4f}")
#     print(f"Sto invalid rate: {sto['stats']['invalid_action_rate']:.4f} | caprej: {sto['stats']['capacity_reject_rate']:.4f} | conflict: {sto['stats']['conflict_drop_rate']:.4f}")
#     print(f"\nSaved → {out}\n")
 
#     return det, sto
 
 
# def load_run_config(run_dir: Path, fallback_config_path: str, conflict_resolution_override: str = None):
#     """Load the config a given run was ACTUALLY trained/built under, from
#     its own run_metadata.json, instead of blindly re-reading the live
#     configs/training_config.yaml — which may have been edited (or, in a
#     multi-resolver sweep, may simply have a different conflict_resolution
#     than this specific run) since that run was created. Falls back to the
#     live yaml only if metadata is missing, with a loud warning.
 
#     conflict_resolution_override, if given, wins even over the run's own
#     saved config — for deliberately evaluating a trained/baseline run under
#     a different resolver than it was built with (an ablation in its own
#     right), rather than for routine use.
#     """
#     metadata_path = run_dir / "run_metadata.json"
#     run_config = None
#     if metadata_path.exists():
#         with open(metadata_path, "r") as f:
#             metadata = json.load(f)
#         run_config = metadata.get("config")
 
#     if run_config is None:
#         print(f"⚠️  No run_metadata.json config found for {run_dir} — falling back to "
#               f"{fallback_config_path}. Results may not reflect what this run actually used "
#               f"if that file has changed since.")
#         run_config = load_config(fallback_config_path) or {}
 
#     if conflict_resolution_override is not None:
#         run_config = dict(run_config)
#         run_config["conflict_resolution"] = conflict_resolution_override
 
#     return run_config
 
 
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", default="configs/training_config.yaml")
#     ap.add_argument("--seed", type=int, default=42,
#                      help="Which seed to evaluate (ignored if --all-seeds is set).")
#     ap.add_argument("--run-id", type=str, default=None,
#                      help="Sweep to evaluate, e.g. '20260712_143000' "
#                           "(runs/run_{id}/). Defaults to the most recent sweep.")
#     ap.add_argument("--run-dir", type=str, default=None,
#                      help="Explicit override: full path to a seed folder, "
#                           "e.g. runs/run_20260712_143000/seed_42. Ignored if --all-seeds is set.")
#     ap.add_argument("--all-seeds", action="store_true",
#                      help="Evaluate every seed_* trained in this sweep (runs/run_{id}/seed_*/) "
#                           "instead of just one, writing results into each seed's own eval_results/.")
#     ap.add_argument("--data-dir", default="data")
#     ap.add_argument("--episodes", type=int, default=50,
#                      help="Episodes PER EVAL SEED (see configs' seeds.eval) — total episodes "
#                           "pooled per train seed's det/stoch result = episodes * len(seeds.eval).")
#     ap.add_argument("--output", type=str, default=None,
#                     help="Optional override output dir (defaults to run_dir/eval_results). "
#                          "Ignored (per-seed subfolders used instead) when --all-seeds is set.")
#     ap.add_argument("--conflict-resolution", type=str, default=None,
#                      choices=["greedy", "random", "hungarian", "hungarian_bids", "capacity", "closest_than_capacity", "predicted_reward", "predicted_reward_joint"],
#                      help="Evaluate under a DIFFERENT resolver than this run was trained "
#                           "with (overrides the run's own saved config). Default: use "
#                           "whatever the run actually trained under.")
#     args = ap.parse_args()
 
#     print("\n============================")
#     print(" PPO EVALUATION")
#     print("============================\n")
 
#     agents, tasks = load_data(args.data_dir)
 
#     if args.all_seeds:
#         seed_dirs = all_seed_dirs_in_run(args.run_id)
#         print(f"Evaluating {len(seed_dirs)} train seeds: {[d.name for d in seed_dirs]}")
#         results = {}
#         for seed_dir in seed_dirs:
#             seed = int(seed_dir.name.replace("seed_", ""))
#             set_seed(seed)
#             config = load_run_config(seed_dir, args.config, args.conflict_resolution)
#             eval_seeds = get_eval_seeds(config)
#             det, sto = evaluate_one_seed(seed, seed_dir, config, agents, tasks, args.episodes,
#                                           eval_seeds=eval_seeds)
#             results[seed] = {"det_mean": det["stats"]["reward_mean"], "sto_mean": sto["stats"]["reward_mean"]}
 
#         print("\n============================")
#         print("SWEEP SUMMARY (across train seeds)")
#         print("============================")
#         print("Each train seed's det/sto number below is ALREADY pooled across every "
#               "configured eval seed (see per-seed output above) — this final line "
#               "aggregates those pooled numbers ACROSS the independently trained models, "
#               "which is a genuinely different kind of variance (model-to-model), not "
#               "episode-to-episode noise.")
#         for seed, r in results.items():
#             print(f"train seed {seed:>4}: det={r['det_mean']:8.2f}  sto={r['sto_mean']:8.2f}")
#         det_means = [r["det_mean"] for r in results.values()]
#         sto_means = [r["sto_mean"] for r in results.values()]
#         print(f"\nAcross train seeds — det: {np.mean(det_means):.2f} ± {np.std(det_means):.2f} | "
#               f"sto: {np.mean(sto_means):.2f} ± {np.std(sto_means):.2f}")
#         return
 
#     set_seed(args.seed)
#     if args.run_dir:
#         run_dir = Path(args.run_dir)
#     else:
#         run_dir = find_latest_run(args.seed, args.run_id)
 
#     config = load_run_config(run_dir, args.config, args.conflict_resolution)
#     eval_seeds = get_eval_seeds(config)
#     evaluate_one_seed(args.seed, run_dir, config, agents, tasks, args.episodes, args.output,
#                        eval_seeds=eval_seeds)
 
 
# if __name__ == "__main__":
#     main()


















"""
Evaluate trained GNN-PPO model (deterministic + stochastic)
Clean + robust + debug metrics
"""
 
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
 
from src.environment.environment import MultiAgentTaskEnv
from src.models.sb3_gnn_policy import RTGNNPolicy
from train_ppo import get_eval_seeds
 
 
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
 
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
 
 
def latest_run_id(runs_root: Path) -> Path:
    """Most recently modified runs/run_{id}/ sweep folder."""
    run_dirs = sorted(runs_root.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {runs_root}")
    return run_dirs[-1]
 
 
def find_latest_run(seed: int, run_id: str = None, runs_root: str = "runs") -> Path:
    """Resolve runs/run_{run_id}/seed_{seed}/ — layout as of the sweep-grouped
    restructure (previously runs/seed_{seed}/run_{id}/, flipped so every seed
    trained in one sweep lives together under a single run_id)."""
    root = Path(runs_root)
    run_root = (root / f"run_{run_id}") if run_id else latest_run_id(root)
 
    seed_dir = run_root / f"seed_{seed}"
    if not seed_dir.exists():
        available = sorted(p.name for p in run_root.glob("seed_*"))
        raise FileNotFoundError(f"No seed_{seed} under {run_root}. Available: {available}")
    return seed_dir
 
 
def all_seed_dirs_in_run(run_id: str = None, runs_root: str = "runs") -> List[Path]:
    """Every seed_* directory trained in one sweep — used for --all-seeds eval."""
    root = Path(runs_root)
    run_root = (root / f"run_{run_id}") if run_id else latest_run_id(root)
    seed_dirs = sorted(run_root.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {run_root}")
    return seed_dirs
 
 
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
 
    if len(tasks) == 0:
        raise FileNotFoundError(f"No task batches found in {p}")
 
    return agents, tasks
 
 
def make_env(agents, tasks, config, seed):
    def _init():
        env = MultiAgentTaskEnv(
            agents=agents,
            tasks_batches=tasks,
            K_max=config["K_max"],
            N_max=config["N_max"],
            E_max=config["E_max"],
            use_xy_pickup=config.get("use_xy_pickup", False),
            normalize_features=config.get("normalize_features", True),
            use_node_type=config.get("use_node_type", True),
            use_ego_robot=config.get("use_ego_robot", True),
            use_edge_rt=config.get("use_edge_rt", False),
            edge_features=config.get("edge_features"),
            two_hop=config.get("two_hop", False),
            two_hop_directed=config.get("two_hop_directed", False),
            vicinity_m=config.get("vicinity_m", 20.0),
            max_steps=config.get("max_steps", 1000),
            max_robot_capacity=config.get("max_robot_capacity", 2),
            max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
            max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
            decision_interval=config.get("decision_interval", 8),
            movement_speed=config.get("movement_speed", 1.0),
            capacity_method=config.get("capacity_method", "assigned"),
            W_COMP=config.get("W_COMP", 2.0),
            W_WAIT=config.get("W_WAIT", 1.0),
            W_DEADLINE=config.get("W_DEADLINE", 10.0),
            W_OBS=config.get("W_OBS", 1.0),
            conflict_resolution=config.get("conflict_resolution", "greedy"),
            candidates_sorting=config.get("candidates_sorting", "distance"),
            reward_type=config.get("reward_type", "legacy"),
            completion_mode=config.get("completion_mode", "dropoff"),
            W_TRAVEL=config.get("W_TRAVEL", 1.25),
        )
        env.reset(seed=seed)
        return env
 
    return DummyVecEnv([_init])
 
 
# =========================================================
# Model
# =========================================================
 
def pick_model(run_dir: Path, prefer_best: bool = True) -> Path:
    # Prefer best_model.zip — written by train_ppo.py's EvalCallback,
    # tracking whichever checkpoint scored highest on periodic
    # deterministic evaluation DURING training, not just whatever the
    # policy happened to be at the very last timestep. ppo_final.zip can
    # be worse than an earlier checkpoint if the policy dipped mid-training
    # and only partially recovered (this is exactly what happened to one
    # MLP seed — see the noop-collapse diagnosis this same evaluation
    # pipeline surfaced). Set prefer_best=False to force the old
    # final-model-only behavior, e.g. for reproducing/debugging older runs.
    if prefer_best:
        best = run_dir / "models" / "best_model.zip"
        if best.exists():
            print("Using best checkpoint (from EvalCallback):", best)
            return best
        print(f"⚠️  No best_model.zip found at {best} — this run may predate "
              f"the best-checkpoint-tracking EvalCallback, or it was disabled. "
              f"Falling back to ppo_final.zip.")
 
    # prefer ppo_final
    final = run_dir / "ppo_final.zip"
    print("Looking for final model:", final)
    if final.exists():
        return final
 
    # then checkpoints
    ckpt_dir = run_dir / "models"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("model_episode*_ts*.zip"))
        if ckpts:
            import re
            def ts(p):
                m = re.search(r"_ts(\d+)\.zip$", p.name)
                return int(m.group(1)) if m else -1
            return max(ckpts, key=ts)
 
    # fallback
    models = list(run_dir.glob("*.zip"))
    if not models:
        raise FileNotFoundError(f"No models found in {run_dir}")
    return max(models, key=lambda p: p.stat().st_mtime)
 
 
# =========================================================
# Evaluation core
# =========================================================
 
def _stats_from_raw(raw: dict) -> dict:
    """Compute the 'stats' summary dict from raw per-episode arrays — shared
    by run_eval() (single eval seed) and run_eval_across_seeds() (pooled
    across multiple eval seeds), so both produce the exact same stats
    shape/semantics regardless of how many seeds' episodes went into it."""
    rewards = raw["rewards"]
    r = np.array(rewards, dtype=float)
    completed = raw["completed"]
    obsolete = raw["obsolete"]
    noop_fractions = raw["noop_fractions"]
    ticks = raw["ticks"]
    ep_mask_zeros = raw["ep_mask_zero_count"]
    ep_r_comp = raw["ep_r_comp"]
    ep_r_wait = raw["ep_r_wait"]
    ep_r_deadline = raw["ep_r_deadline"]
    ep_r_obsolete = raw["ep_r_obsolete"]
    ep_invalids = raw["ep_invalid_action_count"]
    ep_totals = raw["ep_total_action_count"]
    ep_conflicts = raw["ep_conflict_dropped_count"]
    ep_capacity_rej = raw["ep_capacity_rejected_count"]
    ep_noop_forced = raw["ep_noop_forced_count"]
    ep_noop_chosen = raw["ep_noop_chosen_count"]
    ep_had_candidates = raw["ep_had_candidates_count"]
    ep_decisions = raw["ep_decisions_total"]
 
    total_actions_all = int(np.sum(ep_totals)) if ep_totals else 0
    invalid_all = int(np.sum(ep_invalids)) if ep_invalids else 0
    conflict_all = int(np.sum(ep_conflicts)) if ep_conflicts else 0
    caprej_all = int(np.sum(ep_capacity_rej)) if ep_capacity_rej else 0
 
    invalid_rate = (invalid_all / total_actions_all) if total_actions_all > 0 else 0.0
    conflict_rate = (conflict_all / total_actions_all) if total_actions_all > 0 else 0.0
    caprej_rate = (caprej_all / total_actions_all) if total_actions_all > 0 else 0.0
 
    decisions_all      = int(np.sum(ep_decisions)) if ep_decisions else 0
    noop_forced_all    = int(np.sum(ep_noop_forced)) if ep_noop_forced else 0
    noop_chosen_all    = int(np.sum(ep_noop_chosen)) if ep_noop_chosen else 0
    had_candidates_all = int(np.sum(ep_had_candidates)) if ep_had_candidates else 0
 
    noop_frac_forced_rate = (noop_forced_all / decisions_all) if decisions_all > 0 else 0.0
    noop_frac_chosen_rate = (noop_chosen_all / decisions_all) if decisions_all > 0 else 0.0
    chosen_noop_rate_when_available = (
        noop_chosen_all / had_candidates_all if had_candidates_all > 0 else 0.0
    )
 
    return {
        "reward_mean": float(r.mean()) if len(r) else 0.0,
        "reward_std": float(r.std()) if len(r) else 0.0,
        "min": float(r.min()) if len(r) else 0.0,
        "max": float(r.max()) if len(r) else 0.0,
        "completed": float(np.mean(completed)) if completed else 0.0,
        "obsolete": float(np.mean(obsolete)) if obsolete else 0.0,
        "noop_frac_mean": float(np.mean(noop_fractions)) if noop_fractions else 0.0,
        "ticks_mean": float(np.mean(ticks)) if ticks else 0.0,
 
        "invalid_action_total": invalid_all,
        "total_action_count": total_actions_all,
        "invalid_action_rate": float(invalid_rate),
        "conflict_drop_rate": float(conflict_rate),
        "capacity_reject_rate": float(caprej_rate),
        "mask_zero_mean": float(np.mean(ep_mask_zeros)) if ep_mask_zeros else 0.0,
 
        "r_comp_mean": float(np.mean(ep_r_comp)) if ep_r_comp else 0.0,
        "r_wait_mean": float(np.mean(ep_r_wait)) if ep_r_wait else 0.0,
        "r_deadline_mean": float(np.mean(ep_r_deadline)) if ep_r_deadline else 0.0,
        "r_obsolete_mean": float(np.mean(ep_r_obsolete)) if ep_r_obsolete else 0.0,
 
        "noop_frac_forced": float(noop_frac_forced_rate),
        "noop_frac_chosen": float(noop_frac_chosen_rate),
        "chosen_noop_rate_when_available": float(chosen_noop_rate_when_available),
        "decisions_total": decisions_all,
        "had_candidates_total": had_candidates_all,
    }
 
 
def _merge_raw(raw_list: List[dict]) -> dict:
    """Concatenate every list-valued field across multiple run_eval() raw
    outputs (one per eval seed) — used to pool episodes from every eval
    seed together before computing stats, so reward_mean/std reflect the
    FULL eval_seeds pool, not one seed's episodes averaged with another's
    already-averaged summary."""
    if not raw_list:
        raise ValueError("_merge_raw got an empty list")
    merged = {}
    for key in raw_list[0].keys():
        merged[key] = []
        for raw in raw_list:
            merged[key].extend(raw[key])
    return merged
 
 
def run_eval_across_seeds(model, agents, tasks, config, eval_seeds: List[int],
                           episodes_per_seed: int, deterministic: bool):
    """Runs run_eval() once per eval seed (genuinely different environment
    randomization each time — see get_eval_seeds() in train_ppo.py), pools
    every episode from every eval seed together, and computes stats over
    that pooled set. This is what 'reward_mean/reward_std' now actually
    means for a single trained model's evaluation: mean/std across
    len(eval_seeds) * episodes_per_seed episodes, not just one arbitrary
    eval seed's episodes."""
    raw_per_seed = []
    for es in eval_seeds:
        env = make_env(agents, tasks, config, es)
        if config.get("conflict_resolution") == "hungarian_bids":
            model.policy._bid_env = env
        raw = run_eval(model, env, episodes_per_seed, deterministic)
        env.close()
        raw["eval_seed"] = [es] * episodes_per_seed  # tag each episode with which eval seed produced it
        raw_per_seed.append(raw)
 
    merged = _merge_raw(raw_per_seed)
    merged["stats"] = _stats_from_raw(merged)
    merged["stats"]["eval_seeds_used"] = list(eval_seeds)
    merged["stats"]["episodes_per_eval_seed"] = episodes_per_seed
    return merged
 
 
def run_eval(model, env, episodes, deterministic):
    rewards, lengths, ticks, completed, obsolete = [], [], [], [], []
    noop_fractions, action_hists = [], []
 
    # debug episode aggregates
    ep_invalids, ep_totals, ep_valids = [], [], []
    ep_conflicts, ep_capacity_rej, ep_mask_zeros = [], [], []
    ep_r_comp, ep_r_wait, ep_r_deadline, ep_r_obsolete = [], [], [], []
    ep_noop_forced, ep_noop_chosen, ep_had_candidates, ep_decisions = [], [], [], [] 
 
    K_max = env.get_attr("action_space")[0].nvec[0] - 1  # noop index
 
    for ep in range(episodes):
        obs = env.reset()
        done = [False]
 
        ep_r, ep_l = 0.0, 0
        ep_c, ep_o = 0, 0
        ep_actions = []
        ep_time = 0.0
 
        # per-episode debug accumulators
        inv_sum, total_sum, valid_sum = 0, 0, 0
        conflict_sum, caprej_sum, maskz_sum = 0, 0, 0
        rcomp_sum, rwait_sum, rdead_sum, robs_sum = 0.0, 0.0, 0.0, 0.0
        noop_forced_sum, noop_chosen_sum, had_cand_sum, decisions_sum = 0, 0, 0, 0
 
        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = env.step(action)
            done = dones
 
            ep_r += float(r[0])
            ep_l += 1
            ep_actions.extend(np.asarray(action).flatten().tolist())
 
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info, dict):
                ep_c = info.get("completed_count", ep_c)
                ep_o = info.get("obsolete_count", ep_o)
                ep_time = info.get("time", ep_time)
 
                inv_sum += int(info.get("invalid_action_count", 0))
                total_sum += int(info.get("total_action_count", 0))
                valid_sum += int(info.get("valid_action_count", 0))
                conflict_sum += int(info.get("conflict_dropped_count", 0))
                caprej_sum += int(info.get("capacity_rejected_count", 0))
                maskz_sum += int(info.get("mask_zero_count", 0))
 
                rcomp_sum += float(info.get("r_comp", 0.0))
                rwait_sum += float(info.get("r_wait", 0.0))
                rdead_sum += float(info.get("r_deadline", 0.0))
                robs_sum += float(info.get("r_obsolete", 0.0))
 
                noop_forced_sum += int(info.get("noop_forced_count", 0))
                noop_chosen_sum += int(info.get("noop_chosen_count", 0))
                had_cand_sum    += int(info.get("had_candidates_count", 0))
                decisions_sum   += int(info.get("decisions_total", 0))
 
        rewards.append(ep_r)
        lengths.append(ep_l)
        ticks.append(ep_time)
        completed.append(ep_c)
        obsolete.append(ep_o)
 
        actions_arr = np.asarray(ep_actions)
        noop_frac = float((actions_arr == K_max).mean()) if actions_arr.size else 0.0
        noop_fractions.append(noop_frac)
        hist = np.bincount(actions_arr, minlength=K_max + 1).tolist() if actions_arr.size else []
        action_hists.append(hist)
 
        ep_invalids.append(inv_sum)
        ep_totals.append(total_sum)
        ep_valids.append(valid_sum)
        ep_conflicts.append(conflict_sum)
        ep_capacity_rej.append(caprej_sum)
        ep_mask_zeros.append(maskz_sum)
 
        ep_r_comp.append(rcomp_sum)
        ep_r_wait.append(rwait_sum)
        ep_r_deadline.append(rdead_sum)
        ep_r_obsolete.append(robs_sum)
 
        ep_noop_forced.append(noop_forced_sum)
        ep_noop_chosen.append(noop_chosen_sum)
        ep_had_candidates.append(had_cand_sum)
        ep_decisions.append(decisions_sum)
 
    raw = {
        "rewards": rewards,
        "lengths": lengths,
        "ticks": ticks,
        "completed": completed,
        "obsolete": obsolete,
        "noop_fractions": noop_fractions,
        "action_hists": action_hists,
 
        "ep_invalid_action_count": ep_invalids,
        "ep_total_action_count": ep_totals,
        "ep_valid_action_count": ep_valids,
        "ep_conflict_dropped_count": ep_conflicts,
        "ep_capacity_rejected_count": ep_capacity_rej,
        "ep_mask_zero_count": ep_mask_zeros,
 
        "ep_r_comp": ep_r_comp,
        "ep_r_wait": ep_r_wait,
        "ep_r_deadline": ep_r_deadline,
        "ep_r_obsolete": ep_r_obsolete,
 
        "ep_noop_forced_count": ep_noop_forced,
        "ep_noop_chosen_count": ep_noop_chosen,
        "ep_had_candidates_count": ep_had_candidates,
        "ep_decisions_total": ep_decisions,
    }
    raw["stats"] = _stats_from_raw(raw)
    return raw
 
 
# =========================================================
# Main
# =========================================================
 
# =========================================================
# Main
# =========================================================
 
def evaluate_one_seed(seed: int, run_dir: Path, config: Dict, agents, tasks,
                       episodes_per_eval_seed: int, output_override: str = None,
                       eval_seeds: Optional[List[int]] = None):
    """seed here is the TRAIN seed (which trained model to load — see
    run_dir). eval_seeds is the SEPARATE pool used for environment
    randomization during evaluation (get_eval_seeds() in train_ppo.py) —
    every configured eval seed gets episodes_per_eval_seed episodes, and
    ALL of them get pooled together into one reward_mean/reward_std, for
    both deterministic and stochastic readout. Falls back to [seed] if no
    eval_seeds are given, matching the old single-seed behavior."""
    if eval_seeds is None:
        eval_seeds = [seed]
 
    print(f"\n---- train seed {seed} | eval seeds {eval_seeds} ----")
    print(" Selected run:", run_dir)
 
    model_path = pick_model(run_dir)
    print("Using model:", model_path)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    model = PPO.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": RTGNNPolicy},
    )
    print("✓ Loaded model\n")
 
    print(f"Running deterministic across {len(eval_seeds)} eval seed(s)...")
    det = run_eval_across_seeds(model, agents, tasks, config, eval_seeds, episodes_per_eval_seed, True)
 
    print(f"Running stochastic across {len(eval_seeds)} eval seed(s)...")
    sto = run_eval_across_seeds(model, agents, tasks, config, eval_seeds, episodes_per_eval_seed, False)
 
    out = Path(output_override) if output_override else (run_dir / "eval_results")
    out.mkdir(parents=True, exist_ok=True)
 
    save_json(det, out / "deterministic.json")
    save_json(sto, out / "stochastic.json")
 
    debug_summary = {
        "deterministic": det["stats"],
        "stochastic": sto["stats"],
    }
    save_json(debug_summary, out / "debug_summary.json")
 
    print("\n============================")
    print(f"RESULTS — train seed {seed}  (pooled across eval seeds {eval_seeds})")
    print("============================")
    print(f"Deterministic: {det['stats']['reward_mean']:.2f} ± {det['stats']['reward_std']:.2f}  "
          f"(n={len(det['rewards'])} episodes)")
    print(f"Stochastic:    {sto['stats']['reward_mean']:.2f} ± {sto['stats']['reward_std']:.2f}  "
          f"(n={len(sto['rewards'])} episodes)")
    print(f"Det invalid rate: {det['stats']['invalid_action_rate']:.4f} | caprej: {det['stats']['capacity_reject_rate']:.4f} | conflict: {det['stats']['conflict_drop_rate']:.4f}")
    print(f"Sto invalid rate: {sto['stats']['invalid_action_rate']:.4f} | caprej: {sto['stats']['capacity_reject_rate']:.4f} | conflict: {sto['stats']['conflict_drop_rate']:.4f}")
    print(f"\nSaved → {out}\n")
 
    return det, sto
 
 
def load_run_config(run_dir: Path, fallback_config_path: str, conflict_resolution_override: str = None):
    """Load the config a given run was ACTUALLY trained/built under, from
    its own run_metadata.json, instead of blindly re-reading the live
    configs/training_config.yaml — which may have been edited (or, in a
    multi-resolver sweep, may simply have a different conflict_resolution
    than this specific run) since that run was created. Falls back to the
    live yaml only if metadata is missing, with a loud warning.
 
    conflict_resolution_override, if given, wins even over the run's own
    saved config — for deliberately evaluating a trained/baseline run under
    a different resolver than it was built with (an ablation in its own
    right), rather than for routine use.
    """
    metadata_path = run_dir / "run_metadata.json"
    run_config = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        run_config = metadata.get("config")
 
    if run_config is None:
        print(f"⚠️  No run_metadata.json config found for {run_dir} — falling back to "
              f"{fallback_config_path}. Results may not reflect what this run actually used "
              f"if that file has changed since.")
        run_config = load_config(fallback_config_path) or {}
 
    if conflict_resolution_override is not None:
        run_config = dict(run_config)
        run_config["conflict_resolution"] = conflict_resolution_override
 
    return run_config
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--seed", type=int, default=42,
                     help="Which seed to evaluate (ignored if --all-seeds is set).")
    ap.add_argument("--run-id", type=str, default=None,
                     help="Sweep to evaluate, e.g. '20260712_143000' "
                          "(runs/run_{id}/). Defaults to the most recent sweep.")
    ap.add_argument("--run-dir", type=str, default=None,
                     help="Explicit override: full path to a seed folder, "
                          "e.g. runs/run_20260712_143000/seed_42. Ignored if --all-seeds is set.")
    ap.add_argument("--all-seeds", action="store_true",
                     help="Evaluate every seed_* trained in this sweep (runs/run_{id}/seed_*/) "
                          "instead of just one, writing results into each seed's own eval_results/.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--episodes", type=int, default=50,
                     help="Episodes PER EVAL SEED (see configs' seeds.eval) — total episodes "
                          "pooled per train seed's det/stoch result = episodes * len(seeds.eval).")
    ap.add_argument("--output", type=str, default=None,
                    help="Optional override output dir (defaults to run_dir/eval_results). "
                         "Ignored (per-seed subfolders used instead) when --all-seeds is set.")
    ap.add_argument("--conflict-resolution", type=str, default=None,
                     choices=["greedy", "random", "hungarian", "hungarian_bids", "capacity", "closest_than_capacity", "predicted_reward", "predicted_reward_joint"],
                     help="Evaluate under a DIFFERENT resolver than this run was trained "
                          "with (overrides the run's own saved config). Default: use "
                          "whatever the run actually trained under.")
    args = ap.parse_args()
 
    print("\n============================")
    print(" PPO EVALUATION")
    print("============================\n")
 
    agents, tasks = load_data(args.data_dir)
 
    if args.all_seeds:
        seed_dirs = all_seed_dirs_in_run(args.run_id)
        print(f"Evaluating {len(seed_dirs)} train seeds: {[d.name for d in seed_dirs]}")
        results = {}
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.replace("seed_", ""))
            set_seed(seed)
            config = load_run_config(seed_dir, args.config, args.conflict_resolution)
            eval_seeds = get_eval_seeds(config)
            det, sto = evaluate_one_seed(seed, seed_dir, config, agents, tasks, args.episodes,
                                          eval_seeds=eval_seeds)
            results[seed] = {"det_mean": det["stats"]["reward_mean"], "sto_mean": sto["stats"]["reward_mean"]}
 
        print("\n============================")
        print("SWEEP SUMMARY (across train seeds)")
        print("============================")
        print("Each train seed's det/sto number below is ALREADY pooled across every "
              "configured eval seed (see per-seed output above) — this final line "
              "aggregates those pooled numbers ACROSS the independently trained models, "
              "which is a genuinely different kind of variance (model-to-model), not "
              "episode-to-episode noise.")
        for seed, r in results.items():
            print(f"train seed {seed:>4}: det={r['det_mean']:8.2f}  sto={r['sto_mean']:8.2f}")
        det_means = [r["det_mean"] for r in results.values()]
        sto_means = [r["sto_mean"] for r in results.values()]
        print(f"\nAcross train seeds — det: {np.mean(det_means):.2f} ± {np.std(det_means):.2f} | "
              f"sto: {np.mean(sto_means):.2f} ± {np.std(sto_means):.2f}")
        return
 
    set_seed(args.seed)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(args.seed, args.run_id)
 
    config = load_run_config(run_dir, args.config, args.conflict_resolution)
    eval_seeds = get_eval_seeds(config)
    evaluate_one_seed(args.seed, run_dir, config, agents, tasks, args.episodes, args.output,
                       eval_seeds=eval_seeds)
 
 
if __name__ == "__main__":
    main()