
# import argparse
# import json
# import sys
# from pathlib import Path
# from typing import Any, Dict, List, Optional
 
# import numpy as np
# import yaml
 
# from src.environment.environment import MultiAgentTaskEnv
 
# # Import data loading function from train_ppo
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from train_ppo import load_generated_data
 
 
# POLICIES = ["random", "greedy", "unique", "pickup_deadline", "pickup_deadline_distance", "predicted_reward", "predicted_reward_joint", "proposal_joint_competition"]
 
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
 
 
# def load_config(config_path: str) -> dict:
#     """Load YAML config file."""
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)
 
 
# def load_run_config(run_dir: Path, fallback_config_path: str, conflict_resolution_override: str = None):
#     """Load the config a given run was ACTUALLY trained/built under, from
#     its own run_metadata.json, instead of blindly re-reading the live
#     configs/training_config.yaml. Matters especially for baselines in a
#     multi-resolver comparison: baselines go through the same environment's
#     conflict resolver as PPO does, so they must be evaluated under the
#     SAME resolver as the run they're being compared against — not whatever
#     happens to currently be in the yaml. See eval_ppo.py's identical
#     helper for the full rationale."""
#     metadata_path = run_dir / "run_metadata.json"
#     run_config = None
#     if metadata_path.exists():
#         with open(metadata_path, "r") as f:
#             metadata = json.load(f)
#         run_config = metadata.get("config")
 
#     if run_config is None:
#         print(f"⚠️  No run_metadata.json config found for {run_dir} — falling back to "
#               f"{fallback_config_path}. Baseline results may not reflect the resolver "
#               f"this run actually used if that file has changed since.")
#         run_config = load_config(fallback_config_path) or {}
 
#     if conflict_resolution_override is not None:
#         run_config = dict(run_config)
#         run_config["conflict_resolution"] = conflict_resolution_override
 
#     return run_config
 
 
# def _get_mask(obs: Any, info: Any) -> Optional[np.ndarray]:
#     """Extract action mask from observation or info."""
#     if isinstance(info, dict) and "action_mask" in info:
#         return np.asarray(info["action_mask"])
#     if isinstance(obs, dict) and "action_mask" in obs:
#         return np.asarray(obs["action_mask"])
#     return None
 
 
# def _get_last_cand_task_ids(env) -> Optional[List[List[int]]]:
#     """Get candidate task IDs from environment."""
#     try:
#         env0 = env.unwrapped
#         return getattr(env0, "_last_cand_task_ids", None)
#     except Exception:
#         return None
 
 
# def _infer_decision_interval(env, config: Dict) -> int:
#     """Infer decision interval from environment or config."""
#     # For our simplified env, decision interval is effectively 1
#     # (we make decisions every step)
#     return 1
 
 
# def greedy_nearest_action(mask: np.ndarray, env, R: int, K_max: int, NOOP: int) -> np.ndarray:
#     """
#     Greedy policy: pick whichever valid candidate is ACTUALLY nearest.
 
#     Looks up real distance via env instead of assuming slot 0 == nearest —
#     that assumption only holds when the environment's candidates_sorting
#     is 'distance' (the default). With candidates_sorting='randomized'
#     (matches the reference config, avoids the GNN learning a lazy
#     "prefer low slot index" shortcut), slot order carries no distance
#     information at all, so this function must compute distance itself to
#     remain a genuine nearest-neighbor baseline either way.
#     """
#     cand_ids = _get_last_cand_task_ids(env)
#     a = np.full((R,), NOOP, dtype=np.int64)
#     if cand_ids is None:
#         return a
 
#     base_env = env.unwrapped
#     robot_ids = sorted(base_env.robots.keys())
 
#     for r in range(R):
#         if r >= len(robot_ids):
#             continue
#         robot = base_env.robots[robot_ids[r]]
#         best_k, best_dist = None, None
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 task_id = str(int(cand_ids[r][k]))
#             except (IndexError, TypeError, ValueError):
#                 continue
#             task = base_env.tasks.get(task_id)
#             if task is None:
#                 continue
#             dist = (task["pickup_x"] - robot["x"]) ** 2 + (task["pickup_y"] - robot["y"]) ** 2
#             if best_dist is None or dist < best_dist:
#                 best_dist = dist
#                 best_k = k
#         a[r] = best_k if best_k is not None else NOOP
 
#     return a
 
 
# def random_valid_action(
#     mask: np.ndarray, R: int, NOOP: int, rng: np.random.Generator
# ) -> np.ndarray:
#     """
#     Random policy: uniformly sample among all valid actions per robot.
    
#     Behavior:
#     - For each robot, collect all valid action indices (including NOOP)
#     - Uniformly sample one valid index
#     - Fallback to NOOP if no valid actions
#     """
#     a = np.full((R,), NOOP, dtype=np.int64)
#     for r in range(R):
#         allowed = np.flatnonzero(mask[r] == 1)
#         if allowed.size > 0:
#             a[r] = int(rng.choice(allowed))
#         else:
#             a[r] = NOOP
#     return a
 
 
# def greedy_unique_action(
#     mask: np.ndarray,
#     env,
#     R: int,
#     K_max: int,
#     NOOP: int,
#     robot_order: np.ndarray,
#     shuffle_k: bool,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     """
#     Unique policy: greedy with task deduplication.
    
#     Behavior:
#     - Read candidate task IDs from env.unwrapped._last_cand_task_ids
#     - Iterate robots in order
#     - For each robot, scan candidate slots and take first valid candidate
#       whose task_id has not been chosen by another robot in this step
#     - Fallback: if candidate IDs missing, use greedy behavior
#     - Optional: shuffle k scan order or robot order
#     """
#     cand_ids = _get_last_cand_task_ids(env)
#     if cand_ids is None:
#         # Fallback to greedy if no candidate IDs available
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     chosen = set()
#     a = np.full((R,), NOOP, dtype=np.int64)
 
#     for r in robot_order:
#         # Optional: shuffle k scan order
#         if shuffle_k:
#             ks = np.arange(K_max, dtype=int)
#             rng.shuffle(ks)
#         else:
#             ks = range(K_max)
 
#         # Scan candidate slots for this robot
#         for k in ks:
#             if mask[r, k] != 1:
#                 continue
            
#             try:
#                 task_id = int(cand_ids[r][k])
#             except (IndexError, TypeError, ValueError):
#                 continue
            
#             # Skip if task_id < 0 (invalid sentinel)
#             if task_id < 0:
#                 continue
            
#             # Skip if task already chosen by another robot this step
#             if task_id in chosen:
#                 continue
            
#             # This robot takes this task
#             chosen.add(task_id)
#             a[r] = int(k)
#             break
 
#     return a
 
 
# def pickup_deadline_action(
#     mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
# ) -> np.ndarray:
#     """Pickup-deadline policy (matches reference repo's 'pickup_deadline'):
#     for each robot, among its valid candidate slots, pick the one whose
#     task has the SOONEST pickup_deadline. Ties are broken by lowest
#     task_id, NOT by slot order, since slot order is always distance-sorted
#     in this environment and using it as an implicit tie-break would make
#     this behaviorally identical to pickup_deadline_distance_action below."""
#     cand_ids = _get_last_cand_task_ids(env)
#     a = np.full((R,), NOOP, dtype=np.int64)
#     if cand_ids is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     tasks = getattr(env.unwrapped, "tasks", None)
#     if tasks is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     for r in range(R):
#         best_k, best_deadline, best_task_id = None, None, None
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 task_id = int(cand_ids[r][k])
#             except (IndexError, TypeError, ValueError):
#                 continue
#             task = tasks.get(str(task_id))
#             if task is None:
#                 continue
#             deadline = task.get("pickup_deadline", float("inf"))
#             if (
#                 best_deadline is None
#                 or deadline < best_deadline
#                 or (deadline == best_deadline and task_id < best_task_id)
#             ):
#                 best_deadline = deadline
#                 best_task_id = task_id
#                 best_k = k
#         a[r] = best_k if best_k is not None else NOOP
 
#     return a
 
 
# def pickup_deadline_distance_action(
#     mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
# ) -> np.ndarray:
#     """Pickup-deadline-distance policy (matches reference repo's
#     'pickup_deadline_distance'): same urgency-first ranking as
#     pickup_deadline_action, but ties between candidates with an equal
#     deadline are broken by distance — since candidate slots are already
#     distance-sorted ascending by _get_candidate_tasks(), the lowest slot
#     index among tied-deadline candidates is already the nearest one, so
#     ties are broken naturally by iterating slots in order with a strict
#     '<' comparison (an earlier, nearer slot keeps its claim unless a
#     STRICTLY earlier deadline appears)."""
#     cand_ids = _get_last_cand_task_ids(env)
#     a = np.full((R,), NOOP, dtype=np.int64)
#     if cand_ids is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     tasks = getattr(env.unwrapped, "tasks", None)
#     if tasks is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     for r in range(R):
#         best_k, best_deadline = None, None
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 task_id = str(int(cand_ids[r][k]))
#             except (IndexError, TypeError, ValueError):
#                 continue
#             task = tasks.get(task_id)
#             if task is None:
#                 continue
#             deadline = task.get("pickup_deadline", float("inf"))
#             if best_deadline is None or deadline < best_deadline:
#                 best_deadline = deadline
#                 best_k = k
#         a[r] = best_k if best_k is not None else NOOP
 
#     return a
 
 
# def predicted_reward_action(
#     mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
# ) -> np.ndarray:
#     """predicted_reward baseline: for each robot, score every valid
#     candidate with env.predict_candidate_score() (simulated route
#     insertion, see environment.py) and pick the max-scoring one. Falls
#     back to NOOP for a robot if every candidate scores -inf (unreachable/
#     infeasible in the simulated walk)."""
#     cand_ids = _get_last_cand_task_ids(env)
#     a = np.full((R,), NOOP, dtype=np.int64)
#     if cand_ids is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     base_env = env.unwrapped
#     robot_ids = sorted(base_env.robots.keys())
 
#     for r in range(R):
#         if r >= len(robot_ids):
#             continue
#         robot_id = robot_ids[r]
#         best_k, best_score = None, float("-inf")
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 task_id = str(int(cand_ids[r][k]))
#             except (IndexError, TypeError, ValueError):
#                 continue
#             score = base_env.predict_candidate_score(robot_id, task_id)
#             if score > best_score:
#                 best_score = score
#                 best_k = k
#         a[r] = best_k if (best_k is not None and best_score > float("-inf")) else NOOP
 
#     return a
 
 
# def predicted_reward_joint_action(
#     mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
# ) -> np.ndarray:
#     """predicted_reward_joint baseline: same as predicted_reward_action,
#     but scores candidates with env.predict_candidate_score_joint()
#     (marginal R_after - R_before over the whole route) instead of the
#     candidate's own predicted score in isolation."""
#     cand_ids = _get_last_cand_task_ids(env)
#     a = np.full((R,), NOOP, dtype=np.int64)
#     if cand_ids is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     base_env = env.unwrapped
#     robot_ids = sorted(base_env.robots.keys())
 
#     for r in range(R):
#         if r >= len(robot_ids):
#             continue
#         robot_id = robot_ids[r]
#         best_k, best_score = None, float("-inf")
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 task_id = str(int(cand_ids[r][k]))
#             except (IndexError, TypeError, ValueError):
#                 continue
#             score = base_env.predict_candidate_score_joint(robot_id, task_id)
#             if score > best_score:
#                 best_score = score
#                 best_k = k
#         a[r] = best_k if (best_k is not None and best_score > float("-inf")) else NOOP
 
#     return a
 
 
# def proposal_joint_competition_action(
#     mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
# ) -> np.ndarray:
#     """proposal_joint_competition baseline: for each candidate task, a
#     robot only bids on it if its own predicted_reward_joint score is
#     STRICTLY higher than every other robot's score for the SAME task
#     (each competitor's score computed from their own position/committed
#     route via predict_candidate_score_joint) — i.e. it only commits when
#     it's confident it's the best-positioned robot, which is what gives
#     this policy conflict_rate == 0 "by construction" per the reference
#     repo: every robot independently reaches the same conclusion about
#     who the best bidder is, so no separate conflict-resolution step is
#     needed (assuming perfect information and consistent tie-breaking,
#     both of which hold here since every robot's score is computed with
#     the same deterministic function).
 
#     Competitor set: every OTHER robot that also has this task in its OWN
#     candidate list this step (i.e. anyone who could plausibly also want
#     it). This is the natural baseline-level equivalent of the reference's
#     2-hop-graph-restricted competitor set (a GNN-internal notion this
#     baseline doesn't have access to) — it's a superset of the reference's
#     2-hop set, so if anything this baseline bids somewhat LESS often
#     (more candidates count as "in contention"), not more.
 
#     Pair this with the 'greedy' resolver, not 'hungarian'/'hungarian_bids'
#     — this proposer is designed to make conflicts rare/nonexistent on its
#     own, so a cheap fallback resolver for the rare residual conflict is
#     all that's needed.
#     """
#     cand_ids = _get_last_cand_task_ids(env)
#     a = np.full((R,), NOOP, dtype=np.int64)
#     if cand_ids is None:
#         return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#     base_env = env.unwrapped
#     robot_ids = sorted(base_env.robots.keys())
 
#     # Precompute: for every valid (robot, task) candidate pairing this
#     # step, who else also has that same task available.
#     task_to_robots = {}
#     for r in range(R):
#         if r >= len(robot_ids):
#             continue
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 tid = str(int(cand_ids[r][k]))
#             except (IndexError, TypeError, ValueError):
#                 continue
#             task_to_robots.setdefault(tid, []).append(r)
 
#     for r in range(R):
#         if r >= len(robot_ids):
#             continue
#         robot_id = robot_ids[r]
#         best_k, best_score = None, float("-inf")
 
#         for k in range(K_max):
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 tid = str(int(cand_ids[r][k]))
#             except (IndexError, TypeError, ValueError):
#                 continue
 
#             ego_score = base_env.predict_candidate_score_joint(robot_id, tid)
#             if ego_score == float("-inf"):
#                 continue
 
#             competitor_idxs = [c for c in task_to_robots.get(tid, []) if c != r]
#             is_best = True
#             for c in competitor_idxs:
#                 if c >= len(robot_ids):
#                     continue
#                 comp_robot_id = robot_ids[c]
#                 comp_score = base_env.predict_candidate_score_joint(comp_robot_id, tid)
#                 if comp_score >= ego_score:  # strict: ego must beat EVERY competitor
#                     is_best = False
#                     break
 
#             if not is_best:
#                 continue  # a competitor is equally/more entitled -> don't bid
 
#             if ego_score > best_score:
#                 best_score = ego_score
#                 best_k = k
 
#         a[r] = best_k if best_k is not None else NOOP
 
#     return a
 
 
# def evaluate_policy(
#     env,
#     config: Dict,
#     policy_name: str,
#     n_episodes: int,
#     seed: int,
#     debug: bool = False,
#     shuffle_robots: bool = False,
#     unique_shuffle_k: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Evaluate a baseline policy over n_episodes.
    
#     Returns dict with per-episode metrics and aggregated stats.
#     """
#     rng = np.random.default_rng(seed)
 
#     action_space = env.action_space
#     assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"
    
#     R = int(len(action_space.nvec))
#     Kp1 = int(action_space.nvec[0])
#     K_max = Kp1 - 1
#     NOOP = K_max
 
#     decision_interval = _infer_decision_interval(env, config)
 
#     # Metrics tracking
#     ep_rewards: List[float] = []
#     ep_lengths: List[int] = []
#     ep_completed: List[int] = []
#     ep_obsolete: List[int] = []
 
#     for ep in range(n_episodes):
#         obs, info = env.reset(seed=seed + ep)
#         done = False
#         ep_rew = 0.0
#         ep_len = 0
 
#         last_action = np.full((R,), NOOP, dtype=np.int64)
 
#         while not done:
#             # Make decision at each step (decision_interval=1)
#             if ep_len % max(1, decision_interval) == 0:
#                 mask = _get_mask(obs, info)
#                 if mask is None:
#                     raise RuntimeError("No action_mask found in info or obs.")
                
#                 mask = (np.asarray(mask) == 1).astype(np.int32)
 
#                 # Robot order (optional weakening knob)
#                 if shuffle_robots:
#                     robot_order = np.arange(R, dtype=int)
#                     rng.shuffle(robot_order)
#                 else:
#                     robot_order = np.arange(R, dtype=int)
 
#                 if policy_name == "greedy":
#                     last_action = greedy_nearest_action(mask, env, R, K_max, NOOP)
 
#                 elif policy_name == "random":
#                     last_action = random_valid_action(mask, R, NOOP, rng)
 
#                 elif policy_name == "unique":
#                     last_action = greedy_unique_action(
#                         mask=mask,
#                         env=env,
#                         R=R,
#                         K_max=K_max,
#                         NOOP=NOOP,
#                         robot_order=robot_order,
#                         shuffle_k=unique_shuffle_k,
#                         rng=rng,
#                     )
 
#                 elif policy_name == "pickup_deadline":
#                     last_action = pickup_deadline_action(mask, env, R, K_max, NOOP)
 
#                 elif policy_name == "pickup_deadline_distance":
#                     last_action = pickup_deadline_distance_action(mask, env, R, K_max, NOOP)
 
#                 elif policy_name == "predicted_reward":
#                     last_action = predicted_reward_action(mask, env, R, K_max, NOOP)
 
#                 elif policy_name == "predicted_reward_joint":
#                     last_action = predicted_reward_joint_action(mask, env, R, K_max, NOOP)
 
#                 elif policy_name == "proposal_joint_competition":
#                     last_action = proposal_joint_competition_action(mask, env, R, K_max, NOOP)
 
#                 else:
#                     raise ValueError(f"Unknown policy: {policy_name}")
 
#                 if debug and ep == 0 and ep_len < 5:
#                     valid_slots = mask[:, :K_max].sum(axis=1).astype(int).tolist()
#                     print(f"[DEBUG step {ep_len}] valid_slots_per_robot(excl NOOP): {valid_slots}")
#                     print(f"[DEBUG step {ep_len}] has_cand_ids: {_get_last_cand_task_ids(env) is not None}")
#                     print(f"[DEBUG step {ep_len}] action: {last_action.tolist()}")
 
#             # Step environment
#             obs, reward, terminated, truncated, info = env.step(last_action)
#             ep_rew += float(reward)
#             ep_len += 1
#             done = bool(terminated or truncated)
 
#         ep_rewards.append(ep_rew)
#         ep_lengths.append(ep_len)
        
#         # Try to extract completion/obsolete counts from info
#         if isinstance(info, dict):
#             ep_completed.append(int(info.get("completed_count", 0)))
#             ep_obsolete.append(int(info.get("obsolete_count", 0)))
#         else:
#             ep_completed.append(0)
#             ep_obsolete.append(0)
 
#         if debug or (ep + 1) % max(1, max(n_episodes // 5, 1)) == 0:
#             print(
#                 f"  {policy_name:8s} | Episode {ep+1:3d}/{n_episodes} | "
#                 f"Reward: {ep_rew:8.2f} | Length: {ep_len:4d}"
#             )
 
#     # Aggregate statistics
#     rr = np.asarray(ep_rewards, dtype=float)
#     return {
#         "policy": policy_name,
#         "decision_interval": int(decision_interval),
#         "rewards": [float(x) for x in ep_rewards],
#         "completed": [int(x) for x in ep_completed],
#         "obsolete": [int(x) for x in ep_obsolete],
#         "lengths": [int(x) for x in ep_lengths],
#         "stats": {
#             "reward_mean": float(rr.mean()) if rr.size else 0.0,
#             "reward_std": float(rr.std()) if rr.size else 0.0,
#             "completed_mean": float(np.mean(ep_completed)) if ep_completed else 0.0,
#             "obsolete_mean": float(np.mean(ep_obsolete)) if ep_obsolete else 0.0,
#             "length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
#         },
#     }
 
 
# def _concat_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Concatenate results from multiple seeds."""
#     out = {"rewards": [], "completed": [], "obsolete": [], "lengths": []}
#     for r in results_list:
#         for k in out.keys():
#             out[k].extend(r.get(k, []))
 
#     rr = np.asarray(out["rewards"], dtype=float)
#     out["stats"] = {
#         "reward_mean": float(rr.mean()) if rr.size else 0.0,
#         "reward_std": float(rr.std()) if rr.size else 0.0,
#         "completed_mean": float(np.mean(out["completed"])) if out["completed"] else 0.0,
#         "obsolete_mean": float(np.mean(out["obsolete"])) if out["obsolete"] else 0.0,
#         "length_mean": float(np.mean(out["lengths"])) if out["lengths"] else 0.0,
#     }
#     return out
 
 
# def main_one_seedd() -> None:
#     ap = argparse.ArgumentParser(
#         description="Evaluate baseline policies on MultiAgentTaskEnv"
#     )
#     ap.add_argument("--config", type=str, default="configs/training_config.yaml",
#                     help="Path to config YAML")
#     ap.add_argument("--data-dir", type=str, default="data",
#                     help="Path to generated data directory")
#     ap.add_argument("--episodes", type=int, default=20,
#                     help="Episodes per policy per seed")
#     ap.add_argument("--output-dir", type=str, default="baseline_results",
#                     help="Output directory for results")
#     ap.add_argument("--seed", type=int, default=None,
#                     help="Single seed (overrides config)")
#     ap.add_argument("--debug", action="store_true",
#                     help="Print debug info")
#     ap.add_argument("--shuffle-robots", action="store_true",
#                     help="Shuffle robot iteration order (for unique policy)")
#     ap.add_argument("--unique-shuffle-k", action="store_true",
#                     help="Shuffle k scan order (for unique policy)")
#     ap.add_argument("--run-id",type=str, default=None,
#                     help="Run ID created by train_ppo.py")
#     ap.add_argument("--all-seeds", action="store_true",
#                 help="Evaluate every seed in the selected run.")
#     args = ap.parse_args()
 
#     config = load_config(args.config)
 
#     # Determine seeds
#     seeds = config.get("experiment", {}).get("seeds", None)
#     if args.seed is not None:
#         seeds = [int(args.seed)]
#     elif not seeds:
#         seeds = [int(config["experiment"]["seed"])]
#     else:
#         seeds = [int(s) for s in seeds]
 
#     out_dir = Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
 
#     # Load data once
#     print("="*80)
#     print("Loading Generated Data")
#     print("="*80 + "\n")
    
#     try:
#         agents, tasks_batches = load_generated_data(args.data_dir)
#         print(f"✓ Data loaded!")
#         print(f"  Robots: {len(agents)}")
#         print(f"  Batches: {len(tasks_batches)}")
#         print(f"  Tasks: {sum(len(b) for b in tasks_batches)}\n")
#     except Exception as e:
#         print(f" Error loading data: {e}")
#         return
 
#     per_seed: Dict[str, Dict[str, Any]] = {}
#     per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}
 
#     for seed in seeds:
#         print("=" * 80)
#         print(f"Seed {seed} | Episodes per policy: {args.episodes}")
#         print("=" * 80 + "\n")
 
#         per_seed[str(seed)] = {}
 
#         for policy_name in POLICIES:
#             print(f"  Evaluating {policy_name}...")
 
#             # Create environment
#             try:
#                 base_env = MultiAgentTaskEnv(
#                     agents=agents,
#                     tasks_batches=tasks_batches,
#                     K_max=config.get("K_max", 5),
#                     N_max=config.get("N_max", 15),
#                     E_max=config.get("E_max", 50),
#                     use_xy_pickup=config.get("use_xy_pickup", False),
#                     normalize_features=config.get("normalize_features", True),
#                     use_node_type=config.get("use_node_type", True),
#                     use_ego_robot=config.get("use_ego_robot", True),
#                     use_edge_rt=config.get("use_edge_rt", False),
#                     two_hop=config.get("two_hop", False),
#                     vicinity_m=config.get("vicinity_m", 20.0),
#                     max_steps=config.get("max_steps", 1000),
#                 )
#             except Exception as e:
#                 print(f"  Error creating environment: {e}")
#                 continue
 
#             # Evaluate policy
#             try:
#                 res = evaluate_policy(
#                     env=base_env,
#                     config=config,
#                     policy_name=policy_name,
#                     n_episodes=args.episodes,
#                     seed=seed,
#                     debug=args.debug,
#                     shuffle_robots=args.shuffle_robots,
#                     unique_shuffle_k=args.unique_shuffle_k,
#                 )
#                 per_seed[str(seed)][policy_name] = res
#                 per_policy_allseeds[policy_name].append(res)
 
#                 # Save per-policy results
#                 p = out_dir / f"baseline_{policy_name}_seed_{seed}.json"
#                 p.write_text(json.dumps(res, indent=2))
#                 print(f"   {policy_name}: {res['stats']['reward_mean']:.2f} ± {res['stats']['reward_std']:.2f}\n")
 
#             except Exception as e:
#                 print(f"   Error evaluating {policy_name}: {e}")
#                 import traceback
#                 traceback.print_exc()
 
#             try:
#                 base_env.close()
#             except Exception:
#                 pass
 
#     # Combine results across seeds
#     print("\n" + "="*80)
#     print("Summary (All Seeds Combined)")
#     print("="*80 + "\n")
 
#     combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
#     combined_results["num_episodes_per_seed"] = int(args.episodes)
#     combined_results["num_seeds"] = len(seeds)
#     combined_results["seeds"] = seeds
 
#     # Print summary table
#     print(f"{'Policy':<12} {'Reward Mean':<15} {'Reward Std':<15} {'Completed':<15}")
#     # print("-" * 60)
#     for policy in POLICIES:
#         if policy in combined_results:
#             stats = combined_results[policy]["stats"]
#             print(
#                 f"{policy:<12} "
#                 f"{stats['reward_mean']:>12.2f}     "
#                 f"{stats['reward_std']:>12.2f}     "
#                 f"{stats['completed_mean']:>12.2f}"
#             )
 
#     # Save combined results
#     (out_dir / "baseline_results_all.json").write_text(json.dumps(combined_results, indent=2))
#     (out_dir / "baseline_results_per_seed.json").write_text(json.dumps(per_seed, indent=2))
 
#     print(f"\n✓ Saved combined results to {out_dir / 'baseline_results_all.json'}")
#     print(f"✓ Saved per-seed results to {out_dir / 'baseline_results_per_seed.json'}\n")
 
# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Evaluate baseline policies on MultiAgentTaskEnv"
#     )
#     ap.add_argument("--config", type=str, default="configs/training_config.yaml",
#                     help="Path to config YAML")
#     ap.add_argument("--data-dir", type=str, default="data",
#                     help="Path to generated data directory")
#     ap.add_argument("--episodes", type=int, default=20,
#                     help="Episodes per policy per seed")
#     ap.add_argument("--seed", type=int, default=42,
#                     help="Seed to evaluate (ignored if --all-seeds)")
#     ap.add_argument("--debug", action="store_true",
#                     help="Print debug info")
#     ap.add_argument("--shuffle-robots", action="store_true",
#                     help="Shuffle robot iteration order (for unique policy)")
#     ap.add_argument("--unique-shuffle-k", action="store_true",
#                     help="Shuffle k scan order (for unique policy)")
#     ap.add_argument("--run-id", type=str, default=None,
#                     help="Run ID created by train_ppo.py (defaults to latest run)")
#     ap.add_argument("--all-seeds", action="store_true",
#                     help="Evaluate every seed in the selected run.")
#     ap.add_argument("--conflict-resolution", type=str, default=None,
#                     choices=["greedy", "random", "hungarian", "hungarian_bids", "capacity", "closest_than_capacity", "predicted_reward", "predicted_reward_joint"],
#                     help="Evaluate baselines under a DIFFERENT resolver than this run's "
#                          "own saved config. Default: use whatever the run actually used, "
#                          "so PPO and baselines are compared under the same resolver.")
 
#     args = ap.parse_args()
 
#     print("=" * 80)
#     print("Loading Generated Data")
#     print("=" * 80 + "\n")
 
#     try:
#         agents, tasks_batches = load_generated_data(args.data_dir)
#         print(f"✓ Data loaded!")
#         print(f"  Robots: {len(agents)}")
#         print(f"  Batches: {len(tasks_batches)}")
#         print(f"  Tasks: {sum(len(b) for b in tasks_batches)}\n")
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return
 
#     # -------------------------------------------------------------
#     # Determine which seed directories to evaluate
#     # -------------------------------------------------------------
#     if args.all_seeds:
#         seed_dirs = all_seed_dirs_in_run(args.run_id)
#     else:
#         seed_dirs = [find_latest_run(args.seed, args.run_id)]
 
#     for seed_dir in seed_dirs:
#         config = load_run_config(seed_dir, args.config, args.conflict_resolution)
 
#         seed = int(seed_dir.name.replace("seed_", ""))
#         eval_dir = seed_dir / "eval_results"
#         eval_dir.mkdir(parents=True, exist_ok=True)
 
#         print("=" * 80)
#         print(f"Seed {seed} | Episodes per policy: {args.episodes}")
#         print("=" * 80 + "\n")
 
#         per_seed: Dict[str, Dict[str, Any]] = {str(seed): {}}
#         per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {
#             p: [] for p in POLICIES
#         }
 
#         for policy_name in POLICIES:
 
#             print(f"  Evaluating {policy_name}...")
 
#             # Baselines pick actions via pure heuristics — there is no
#             # policy in this loop at all, so there are no logits to bid
#             # with. 'hungarian_bids' is meaningless here; fall back to
#             # plain distance-based 'hungarian' as the closest well-defined
#             # equivalent, so baselines still get a real, comparable
#             # centralized-assignment resolver rather than silently
#             # crashing (see set_pending_logits()'s RuntimeError in
#             # environment.py for why it would otherwise).
#             env_conflict_resolution = config.get("conflict_resolution", "greedy")
#             if env_conflict_resolution == "hungarian_bids":
#                 env_conflict_resolution = "hungarian"
#                 print("    (conflict_resolution='hungarian_bids' has no meaning for baselines — "
#                       "no policy/logits exist here — using 'hungarian' distance-based assignment instead)")
 
#             try:
#                 base_env = MultiAgentTaskEnv(
#                     agents=agents,
#                     tasks_batches=tasks_batches,
#                     K_max=config["K_max"],
#                     N_max=config["N_max"],
#                     E_max=config["E_max"],
#                     use_xy_pickup=config.get("use_xy_pickup", False),
#                     normalize_features=config.get("normalize_features", True),
#                     use_node_type=config.get("use_node_type", True),
#                     use_ego_robot=config.get("use_ego_robot", True),
#                     use_edge_rt=config.get("use_edge_rt", False),
#                     edge_features=config.get("edge_features"),
#                     two_hop=config.get("two_hop", False),
#                     two_hop_directed=config.get("two_hop_directed", False),
#                     vicinity_m=config.get("vicinity_m", 20.0),
#                     max_steps=config.get("max_steps", 1000),
#                     max_robot_capacity=config.get("max_robot_capacity", 2),
#                     max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
#                     max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
#                     decision_interval=config.get("decision_interval", 8),
#                     movement_speed=config.get("movement_speed", 1.0),
#                     capacity_method=config.get("capacity_method", "assigned"),
#                     W_COMP=config.get("W_COMP", 2.0),
#                     W_WAIT=config.get("W_WAIT", 1.0),
#                     W_DEADLINE=config.get("W_DEADLINE", 10.0),
#                     W_OBS=config.get("W_OBS", 1.0),
#                     conflict_resolution=env_conflict_resolution,
#                     candidates_sorting=config.get("candidates_sorting", "distance"),
#                 )
#             except Exception as e:
#                 print(f"  Error creating environment: {e}")
#                 continue
 
#             try:
#                 res = evaluate_policy(
#                     env=base_env,
#                     config=config,
#                     policy_name=policy_name,
#                     n_episodes=args.episodes,
#                     seed=seed,
#                     debug=args.debug,
#                     shuffle_robots=args.shuffle_robots,
#                     unique_shuffle_k=args.unique_shuffle_k,
#                 )
 
#                 per_seed[str(seed)][policy_name] = res
#                 per_policy_allseeds[policy_name].append(res)
 
#                 # save individual policy result
#                 (eval_dir / f"baseline_{policy_name}.json").write_text(
#                     json.dumps(res, indent=2)
#                 )
 
#                 print(
#                     f"   {policy_name}: "
#                     f"{res['stats']['reward_mean']:.2f} ± "
#                     f"{res['stats']['reward_std']:.2f}\n"
#                 )
 
#             except Exception as e:
#                 print(f"   Error evaluating {policy_name}: {e}")
#                 import traceback
#                 traceback.print_exc()
 
#             finally:
#                 try:
#                     base_env.close()
#                 except Exception:
#                     pass
 
#         # ---------------------------------------------------------
#         # Combined summary for this seed
#         # ---------------------------------------------------------
#         combined_results = {
#             p: _concat_results(per_policy_allseeds[p])
#             for p in POLICIES
#         }
#         combined_results["num_episodes_per_seed"] = int(args.episodes)
#         combined_results["num_seeds"] = 1
#         combined_results["seeds"] = [seed]
 
#         (eval_dir / "baseline_results_all.json").write_text(
#             json.dumps(combined_results, indent=2)
#         )
#         (eval_dir / "baseline_results_per_seed.json").write_text(
#             json.dumps(per_seed, indent=2)
#         )
 
#         print("\n" + "=" * 80)
#         print(f"Summary (Seed {seed})")
#         print("=" * 80 + "\n")
 
#         print(f"{'Policy':<12} {'Reward Mean':<15} {'Reward Std':<15} {'Completed':<15}")
#         for policy in POLICIES:
#             stats = combined_results[policy]["stats"]
#             print(
#                 f"{policy:<12}"
#                 f"{stats['reward_mean']:>12.2f}     "
#                 f"{stats['reward_std']:>12.2f}     "
#                 f"{stats['completed_mean']:>12.2f}"
#             )
 
#         print(f"\n✓ Results saved to {eval_dir}\n")
        
# if __name__ == "__main__":
#     main()




import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
 
import numpy as np
import yaml
 
from src.environment.environment import MultiAgentTaskEnv
 
# Import data loading function from train_ppo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ppo import load_generated_data, get_eval_seeds
 
 
POLICIES = ["random", "greedy", "unique", "pickup_deadline", "pickup_deadline_distance", "predicted_reward", "predicted_reward_joint", "proposal_joint_competition"]
 
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
 
 
def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
 
 
def load_run_config(run_dir: Path, fallback_config_path: str, conflict_resolution_override: str = None):
    """Load the config a given run was ACTUALLY trained/built under, from
    its own run_metadata.json, instead of blindly re-reading the live
    configs/training_config.yaml. Matters especially for baselines in a
    multi-resolver comparison: baselines go through the same environment's
    conflict resolver as PPO does, so they must be evaluated under the
    SAME resolver as the run they're being compared against — not whatever
    happens to currently be in the yaml. See eval_ppo.py's identical
    helper for the full rationale."""
    metadata_path = run_dir / "run_metadata.json"
    run_config = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        run_config = metadata.get("config")
 
    if run_config is None:
        print(f"⚠️  No run_metadata.json config found for {run_dir} — falling back to "
              f"{fallback_config_path}. Baseline results may not reflect the resolver "
              f"this run actually used if that file has changed since.")
        run_config = load_config(fallback_config_path) or {}
 
    if conflict_resolution_override is not None:
        run_config = dict(run_config)
        run_config["conflict_resolution"] = conflict_resolution_override
 
    return run_config
 
 
def _get_mask(obs: Any, info: Any) -> Optional[np.ndarray]:
    """Extract action mask from observation or info."""
    if isinstance(info, dict) and "action_mask" in info:
        return np.asarray(info["action_mask"])
    if isinstance(obs, dict) and "action_mask" in obs:
        return np.asarray(obs["action_mask"])
    return None
 
 
def _get_last_cand_task_ids(env) -> Optional[List[List[int]]]:
    """Get candidate task IDs from environment."""
    try:
        env0 = env.unwrapped
        return getattr(env0, "_last_cand_task_ids", None)
    except Exception:
        return None
 
 
def _infer_decision_interval(env, config: Dict) -> int:
    """Infer decision interval from environment or config."""
    # For our simplified env, decision interval is effectively 1
    # (we make decisions every step)
    return 1
 
 
def greedy_nearest_action(mask: np.ndarray, env, R: int, K_max: int, NOOP: int) -> np.ndarray:
    """
    Greedy policy: pick whichever valid candidate is ACTUALLY nearest.
 
    Looks up real distance via env instead of assuming slot 0 == nearest —
    that assumption only holds when the environment's candidates_sorting
    is 'distance' (the default). With candidates_sorting='randomized'
    (matches the reference config, avoids the GNN learning a lazy
    "prefer low slot index" shortcut), slot order carries no distance
    information at all, so this function must compute distance itself to
    remain a genuine nearest-neighbor baseline either way.
    """
    cand_ids = _get_last_cand_task_ids(env)
    a = np.full((R,), NOOP, dtype=np.int64)
    if cand_ids is None:
        return a
 
    base_env = env.unwrapped
    robot_ids = sorted(base_env.robots.keys())
 
    for r in range(R):
        if r >= len(robot_ids):
            continue
        robot = base_env.robots[robot_ids[r]]
        best_k, best_dist = None, None
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                task_id = str(int(cand_ids[r][k]))
            except (IndexError, TypeError, ValueError):
                continue
            task = base_env.tasks.get(task_id)
            if task is None:
                continue
            dist = (task["pickup_x"] - robot["x"]) ** 2 + (task["pickup_y"] - robot["y"]) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_k = k
        a[r] = best_k if best_k is not None else NOOP
 
    return a
 
 
def random_valid_action(
    mask: np.ndarray, R: int, NOOP: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Random policy: uniformly sample among all valid actions per robot.
    
    Behavior:
    - For each robot, collect all valid action indices (including NOOP)
    - Uniformly sample one valid index
    - Fallback to NOOP if no valid actions
    """
    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        allowed = np.flatnonzero(mask[r] == 1)
        if allowed.size > 0:
            a[r] = int(rng.choice(allowed))
        else:
            a[r] = NOOP
    return a
 
 
def greedy_unique_action(
    mask: np.ndarray,
    env,
    R: int,
    K_max: int,
    NOOP: int,
    robot_order: np.ndarray,
    shuffle_k: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Unique policy: greedy with task deduplication.
    
    Behavior:
    - Read candidate task IDs from env.unwrapped._last_cand_task_ids
    - Iterate robots in order
    - For each robot, scan candidate slots and take first valid candidate
      whose task_id has not been chosen by another robot in this step
    - Fallback: if candidate IDs missing, use greedy behavior
    - Optional: shuffle k scan order or robot order
    """
    cand_ids = _get_last_cand_task_ids(env)
    if cand_ids is None:
        # Fallback to greedy if no candidate IDs available
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    chosen = set()
    a = np.full((R,), NOOP, dtype=np.int64)
 
    for r in robot_order:
        # Optional: shuffle k scan order
        if shuffle_k:
            ks = np.arange(K_max, dtype=int)
            rng.shuffle(ks)
        else:
            ks = range(K_max)
 
        # Scan candidate slots for this robot
        for k in ks:
            if mask[r, k] != 1:
                continue
            
            try:
                task_id = int(cand_ids[r][k])
            except (IndexError, TypeError, ValueError):
                continue
            
            # Skip if task_id < 0 (invalid sentinel)
            if task_id < 0:
                continue
            
            # Skip if task already chosen by another robot this step
            if task_id in chosen:
                continue
            
            # This robot takes this task
            chosen.add(task_id)
            a[r] = int(k)
            break
 
    return a
 
 
def pickup_deadline_action(
    mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
) -> np.ndarray:
    """Pickup-deadline policy (matches reference repo's 'pickup_deadline'):
    for each robot, among its valid candidate slots, pick the one whose
    task has the SOONEST pickup_deadline. Ties are broken by lowest
    task_id, NOT by slot order, since slot order is always distance-sorted
    in this environment and using it as an implicit tie-break would make
    this behaviorally identical to pickup_deadline_distance_action below."""
    cand_ids = _get_last_cand_task_ids(env)
    a = np.full((R,), NOOP, dtype=np.int64)
    if cand_ids is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    tasks = getattr(env.unwrapped, "tasks", None)
    if tasks is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    for r in range(R):
        best_k, best_deadline, best_task_id = None, None, None
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                task_id = int(cand_ids[r][k])
            except (IndexError, TypeError, ValueError):
                continue
            task = tasks.get(str(task_id))
            if task is None:
                continue
            deadline = task.get("pickup_deadline", float("inf"))
            if (
                best_deadline is None
                or deadline < best_deadline
                or (deadline == best_deadline and task_id < best_task_id)
            ):
                best_deadline = deadline
                best_task_id = task_id
                best_k = k
        a[r] = best_k if best_k is not None else NOOP
 
    return a
 
 
def pickup_deadline_distance_action(
    mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
) -> np.ndarray:
    """Pickup-deadline-distance policy (matches reference repo's
    'pickup_deadline_distance'): same urgency-first ranking as
    pickup_deadline_action, but ties between candidates with an equal
    deadline are broken by distance — since candidate slots are already
    distance-sorted ascending by _get_candidate_tasks(), the lowest slot
    index among tied-deadline candidates is already the nearest one, so
    ties are broken naturally by iterating slots in order with a strict
    '<' comparison (an earlier, nearer slot keeps its claim unless a
    STRICTLY earlier deadline appears)."""
    cand_ids = _get_last_cand_task_ids(env)
    a = np.full((R,), NOOP, dtype=np.int64)
    if cand_ids is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    tasks = getattr(env.unwrapped, "tasks", None)
    if tasks is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    for r in range(R):
        best_k, best_deadline = None, None
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                task_id = str(int(cand_ids[r][k]))
            except (IndexError, TypeError, ValueError):
                continue
            task = tasks.get(task_id)
            if task is None:
                continue
            deadline = task.get("pickup_deadline", float("inf"))
            if best_deadline is None or deadline < best_deadline:
                best_deadline = deadline
                best_k = k
        a[r] = best_k if best_k is not None else NOOP
 
    return a
 
 
def predicted_reward_action(
    mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
) -> np.ndarray:
    """predicted_reward baseline: for each robot, score every valid
    candidate with env.predict_candidate_score() (simulated route
    insertion, see environment.py) and pick the max-scoring one. Falls
    back to NOOP for a robot if every candidate scores -inf (unreachable/
    infeasible in the simulated walk)."""
    cand_ids = _get_last_cand_task_ids(env)
    a = np.full((R,), NOOP, dtype=np.int64)
    if cand_ids is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    base_env = env.unwrapped
    robot_ids = sorted(base_env.robots.keys())
 
    for r in range(R):
        if r >= len(robot_ids):
            continue
        robot_id = robot_ids[r]
        best_k, best_score = None, float("-inf")
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                task_id = str(int(cand_ids[r][k]))
            except (IndexError, TypeError, ValueError):
                continue
            score = base_env.predict_candidate_score(robot_id, task_id)
            if score > best_score:
                best_score = score
                best_k = k
        a[r] = best_k if (best_k is not None and best_score > float("-inf")) else NOOP
 
    return a
 
 
def predicted_reward_joint_action(
    mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
) -> np.ndarray:
    """predicted_reward_joint baseline: same as predicted_reward_action,
    but scores candidates with env.predict_candidate_score_joint()
    (marginal R_after - R_before over the whole route) instead of the
    candidate's own predicted score in isolation."""
    cand_ids = _get_last_cand_task_ids(env)
    a = np.full((R,), NOOP, dtype=np.int64)
    if cand_ids is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    base_env = env.unwrapped
    robot_ids = sorted(base_env.robots.keys())
 
    for r in range(R):
        if r >= len(robot_ids):
            continue
        robot_id = robot_ids[r]
        best_k, best_score = None, float("-inf")
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                task_id = str(int(cand_ids[r][k]))
            except (IndexError, TypeError, ValueError):
                continue
            score = base_env.predict_candidate_score_joint(robot_id, task_id)
            if score > best_score:
                best_score = score
                best_k = k
        a[r] = best_k if (best_k is not None and best_score > float("-inf")) else NOOP
 
    return a
 
 
def proposal_joint_competition_action(
    mask: np.ndarray, env, R: int, K_max: int, NOOP: int,
) -> np.ndarray:
    """proposal_joint_competition baseline: for each candidate task, a
    robot only bids on it if its own predicted_reward_joint score is
    STRICTLY higher than every other robot's score for the SAME task
    (each competitor's score computed from their own position/committed
    route via predict_candidate_score_joint) — i.e. it only commits when
    it's confident it's the best-positioned robot, which is what gives
    this policy conflict_rate == 0 "by construction" per the reference
    repo: every robot independently reaches the same conclusion about
    who the best bidder is, so no separate conflict-resolution step is
    needed (assuming perfect information and consistent tie-breaking,
    both of which hold here since every robot's score is computed with
    the same deterministic function).
 
    Competitor set: every OTHER robot that also has this task in its OWN
    candidate list this step (i.e. anyone who could plausibly also want
    it). This is the natural baseline-level equivalent of the reference's
    2-hop-graph-restricted competitor set (a GNN-internal notion this
    baseline doesn't have access to) — it's a superset of the reference's
    2-hop set, so if anything this baseline bids somewhat LESS often
    (more candidates count as "in contention"), not more.
 
    Pair this with the 'greedy' resolver, not 'hungarian'/'hungarian_bids'
    — this proposer is designed to make conflicts rare/nonexistent on its
    own, so a cheap fallback resolver for the rare residual conflict is
    all that's needed.
    """
    cand_ids = _get_last_cand_task_ids(env)
    a = np.full((R,), NOOP, dtype=np.int64)
    if cand_ids is None:
        return greedy_nearest_action(mask, env, R, K_max, NOOP)
 
    base_env = env.unwrapped
    robot_ids = sorted(base_env.robots.keys())
 
    # Precompute: for every valid (robot, task) candidate pairing this
    # step, who else also has that same task available.
    task_to_robots = {}
    for r in range(R):
        if r >= len(robot_ids):
            continue
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                tid = str(int(cand_ids[r][k]))
            except (IndexError, TypeError, ValueError):
                continue
            task_to_robots.setdefault(tid, []).append(r)
 
    for r in range(R):
        if r >= len(robot_ids):
            continue
        robot_id = robot_ids[r]
        best_k, best_score = None, float("-inf")
 
        for k in range(K_max):
            if mask[r, k] != 1:
                continue
            try:
                tid = str(int(cand_ids[r][k]))
            except (IndexError, TypeError, ValueError):
                continue
 
            ego_score = base_env.predict_candidate_score_joint(robot_id, tid)
            if ego_score == float("-inf"):
                continue
 
            competitor_idxs = [c for c in task_to_robots.get(tid, []) if c != r]
            is_best = True
            for c in competitor_idxs:
                if c >= len(robot_ids):
                    continue
                comp_robot_id = robot_ids[c]
                comp_score = base_env.predict_candidate_score_joint(comp_robot_id, tid)
                if comp_score >= ego_score:  # strict: ego must beat EVERY competitor
                    is_best = False
                    break
 
            if not is_best:
                continue  # a competitor is equally/more entitled -> don't bid
 
            if ego_score > best_score:
                best_score = ego_score
                best_k = k
 
        a[r] = best_k if best_k is not None else NOOP
 
    return a
 
 
def evaluate_policy(
    env,
    config: Dict,
    policy_name: str,
    n_episodes: int,
    seed: int,
    debug: bool = False,
    shuffle_robots: bool = False,
    unique_shuffle_k: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a baseline policy over n_episodes.
    
    Returns dict with per-episode metrics and aggregated stats.
    """
    rng = np.random.default_rng(seed)
 
    action_space = env.action_space
    assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"
    
    R = int(len(action_space.nvec))
    Kp1 = int(action_space.nvec[0])
    K_max = Kp1 - 1
    NOOP = K_max
 
    decision_interval = _infer_decision_interval(env, config)
 
    # Metrics tracking
    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    ep_completed: List[int] = []
    ep_obsolete: List[int] = []
 
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_rew = 0.0
        ep_len = 0
 
        last_action = np.full((R,), NOOP, dtype=np.int64)
 
        while not done:
            # Make decision at each step (decision_interval=1)
            if ep_len % max(1, decision_interval) == 0:
                mask = _get_mask(obs, info)
                if mask is None:
                    raise RuntimeError("No action_mask found in info or obs.")
                
                mask = (np.asarray(mask) == 1).astype(np.int32)
 
                # Robot order (optional weakening knob)
                if shuffle_robots:
                    robot_order = np.arange(R, dtype=int)
                    rng.shuffle(robot_order)
                else:
                    robot_order = np.arange(R, dtype=int)
 
                if policy_name == "greedy":
                    last_action = greedy_nearest_action(mask, env, R, K_max, NOOP)
 
                elif policy_name == "random":
                    last_action = random_valid_action(mask, R, NOOP, rng)
 
                elif policy_name == "unique":
                    last_action = greedy_unique_action(
                        mask=mask,
                        env=env,
                        R=R,
                        K_max=K_max,
                        NOOP=NOOP,
                        robot_order=robot_order,
                        shuffle_k=unique_shuffle_k,
                        rng=rng,
                    )
 
                elif policy_name == "pickup_deadline":
                    last_action = pickup_deadline_action(mask, env, R, K_max, NOOP)
 
                elif policy_name == "pickup_deadline_distance":
                    last_action = pickup_deadline_distance_action(mask, env, R, K_max, NOOP)
 
                elif policy_name == "predicted_reward":
                    last_action = predicted_reward_action(mask, env, R, K_max, NOOP)
 
                elif policy_name == "predicted_reward_joint":
                    last_action = predicted_reward_joint_action(mask, env, R, K_max, NOOP)
 
                elif policy_name == "proposal_joint_competition":
                    last_action = proposal_joint_competition_action(mask, env, R, K_max, NOOP)
 
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")
 
                if debug and ep == 0 and ep_len < 5:
                    valid_slots = mask[:, :K_max].sum(axis=1).astype(int).tolist()
                    print(f"[DEBUG step {ep_len}] valid_slots_per_robot(excl NOOP): {valid_slots}")
                    print(f"[DEBUG step {ep_len}] has_cand_ids: {_get_last_cand_task_ids(env) is not None}")
                    print(f"[DEBUG step {ep_len}] action: {last_action.tolist()}")
 
            # Step environment
            obs, reward, terminated, truncated, info = env.step(last_action)
            ep_rew += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)
 
        ep_rewards.append(ep_rew)
        ep_lengths.append(ep_len)
        
        # Try to extract completion/obsolete counts from info
        if isinstance(info, dict):
            ep_completed.append(int(info.get("completed_count", 0)))
            ep_obsolete.append(int(info.get("obsolete_count", 0)))
        else:
            ep_completed.append(0)
            ep_obsolete.append(0)
 
        if debug or (ep + 1) % max(1, max(n_episodes // 5, 1)) == 0:
            print(
                f"  {policy_name:8s} | Episode {ep+1:3d}/{n_episodes} | "
                f"Reward: {ep_rew:8.2f} | Length: {ep_len:4d}"
            )
 
    # Aggregate statistics
    rr = np.asarray(ep_rewards, dtype=float)
    return {
        "policy": policy_name,
        "decision_interval": int(decision_interval),
        "rewards": [float(x) for x in ep_rewards],
        "completed": [int(x) for x in ep_completed],
        "obsolete": [int(x) for x in ep_obsolete],
        "lengths": [int(x) for x in ep_lengths],
        "stats": {
            "reward_mean": float(rr.mean()) if rr.size else 0.0,
            "reward_std": float(rr.std()) if rr.size else 0.0,
            "completed_mean": float(np.mean(ep_completed)) if ep_completed else 0.0,
            "obsolete_mean": float(np.mean(ep_obsolete)) if ep_obsolete else 0.0,
            "length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        },
    }
 
 
def _concat_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate results from multiple seeds."""
    out = {"rewards": [], "completed": [], "obsolete": [], "lengths": []}
    for r in results_list:
        for k in out.keys():
            out[k].extend(r.get(k, []))
 
    rr = np.asarray(out["rewards"], dtype=float)
    out["stats"] = {
        "reward_mean": float(rr.mean()) if rr.size else 0.0,
        "reward_std": float(rr.std()) if rr.size else 0.0,
        "completed_mean": float(np.mean(out["completed"])) if out["completed"] else 0.0,
        "obsolete_mean": float(np.mean(out["obsolete"])) if out["obsolete"] else 0.0,
        "length_mean": float(np.mean(out["lengths"])) if out["lengths"] else 0.0,
    }
    return out
 
 
def main_one_seedd() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate baseline policies on MultiAgentTaskEnv"
    )
    ap.add_argument("--config", type=str, default="configs/training_config.yaml",
                    help="Path to config YAML")
    ap.add_argument("--data-dir", type=str, default="data",
                    help="Path to generated data directory")
    ap.add_argument("--episodes", type=int, default=20,
                    help="Episodes per policy per seed")
    ap.add_argument("--output-dir", type=str, default="baseline_results",
                    help="Output directory for results")
    ap.add_argument("--seed", type=int, default=None,
                    help="Single seed (overrides config)")
    ap.add_argument("--debug", action="store_true",
                    help="Print debug info")
    ap.add_argument("--shuffle-robots", action="store_true",
                    help="Shuffle robot iteration order (for unique policy)")
    ap.add_argument("--unique-shuffle-k", action="store_true",
                    help="Shuffle k scan order (for unique policy)")
    ap.add_argument("--run-id",type=str, default=None,
                    help="Run ID created by train_ppo.py")
    ap.add_argument("--all-seeds", action="store_true",
                help="Evaluate every seed in the selected run.")
    args = ap.parse_args()
 
    config = load_config(args.config)
 
    # Determine seeds
    seeds = config.get("experiment", {}).get("seeds", None)
    if args.seed is not None:
        seeds = [int(args.seed)]
    elif not seeds:
        seeds = [int(config["experiment"]["seed"])]
    else:
        seeds = [int(s) for s in seeds]
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # Load data once
    print("="*80)
    print("Loading Generated Data")
    print("="*80 + "\n")
    
    try:
        agents, tasks_batches = load_generated_data(args.data_dir)
        print(f"✓ Data loaded!")
        print(f"  Robots: {len(agents)}")
        print(f"  Batches: {len(tasks_batches)}")
        print(f"  Tasks: {sum(len(b) for b in tasks_batches)}\n")
    except Exception as e:
        print(f" Error loading data: {e}")
        return
 
    per_seed: Dict[str, Dict[str, Any]] = {}
    per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}
 
    for seed in seeds:
        print("=" * 80)
        print(f"Seed {seed} | Episodes per policy: {args.episodes}")
        print("=" * 80 + "\n")
 
        per_seed[str(seed)] = {}
 
        for policy_name in POLICIES:
            print(f"  Evaluating {policy_name}...")
 
            # Create environment
            try:
                base_env = MultiAgentTaskEnv(
                    agents=agents,
                    tasks_batches=tasks_batches,
                    K_max=config.get("K_max", 5),
                    N_max=config.get("N_max", 15),
                    E_max=config.get("E_max", 50),
                    use_xy_pickup=config.get("use_xy_pickup", False),
                    normalize_features=config.get("normalize_features", True),
                    use_node_type=config.get("use_node_type", True),
                    use_ego_robot=config.get("use_ego_robot", True),
                    use_edge_rt=config.get("use_edge_rt", False),
                    two_hop=config.get("two_hop", False),
                    vicinity_m=config.get("vicinity_m", 20.0),
                    max_steps=config.get("max_steps", 1000),
                )
            except Exception as e:
                print(f"  Error creating environment: {e}")
                continue
 
            # Evaluate policy
            try:
                res = evaluate_policy(
                    env=base_env,
                    config=config,
                    policy_name=policy_name,
                    n_episodes=args.episodes,
                    seed=seed,
                    debug=args.debug,
                    shuffle_robots=args.shuffle_robots,
                    unique_shuffle_k=args.unique_shuffle_k,
                )
                per_seed[str(seed)][policy_name] = res
                per_policy_allseeds[policy_name].append(res)
 
                # Save per-policy results
                p = out_dir / f"baseline_{policy_name}_seed_{seed}.json"
                p.write_text(json.dumps(res, indent=2))
                print(f"   {policy_name}: {res['stats']['reward_mean']:.2f} ± {res['stats']['reward_std']:.2f}\n")
 
            except Exception as e:
                print(f"   Error evaluating {policy_name}: {e}")
                import traceback
                traceback.print_exc()
 
            try:
                base_env.close()
            except Exception:
                pass
 
    # Combine results across seeds
    print("\n" + "="*80)
    print("Summary (All Seeds Combined)")
    print("="*80 + "\n")
 
    combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
    combined_results["num_episodes_per_seed"] = int(args.episodes)
    combined_results["num_seeds"] = len(seeds)
    combined_results["seeds"] = seeds
 
    # Print summary table
    print(f"{'Policy':<12} {'Reward Mean':<15} {'Reward Std':<15} {'Completed':<15}")
    # print("-" * 60)
    for policy in POLICIES:
        if policy in combined_results:
            stats = combined_results[policy]["stats"]
            print(
                f"{policy:<12} "
                f"{stats['reward_mean']:>12.2f}     "
                f"{stats['reward_std']:>12.2f}     "
                f"{stats['completed_mean']:>12.2f}"
            )
 
    # Save combined results
    (out_dir / "baseline_results_all.json").write_text(json.dumps(combined_results, indent=2))
    (out_dir / "baseline_results_per_seed.json").write_text(json.dumps(per_seed, indent=2))
 
    print(f"\n✓ Saved combined results to {out_dir / 'baseline_results_all.json'}")
    print(f"✓ Saved per-seed results to {out_dir / 'baseline_results_per_seed.json'}\n")
 
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate baseline policies on MultiAgentTaskEnv"
    )
    ap.add_argument("--config", type=str, default="configs/training_config.yaml",
                    help="Path to config YAML")
    ap.add_argument("--data-dir", type=str, default="data",
                    help="Path to generated data directory")
    ap.add_argument("--episodes", type=int, default=20,
                    help="Episodes per policy per seed")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed to evaluate (ignored if --all-seeds)")
    ap.add_argument("--debug", action="store_true",
                    help="Print debug info")
    ap.add_argument("--shuffle-robots", action="store_true",
                    help="Shuffle robot iteration order (for unique policy)")
    ap.add_argument("--unique-shuffle-k", action="store_true",
                    help="Shuffle k scan order (for unique policy)")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Run ID created by train_ppo.py (defaults to latest run)")
    ap.add_argument("--all-seeds", action="store_true",
                    help="Evaluate every seed in the selected run.")
    ap.add_argument("--conflict-resolution", type=str, default=None,
                    choices=["greedy", "random", "hungarian", "hungarian_bids", "capacity", "closest_than_capacity", "predicted_reward", "predicted_reward_joint"],
                    help="Evaluate baselines under a DIFFERENT resolver than this run's "
                         "own saved config. Default: use whatever the run actually used, "
                         "so PPO and baselines are compared under the same resolver.")
 
    args = ap.parse_args()
 
    print("=" * 80)
    print("Loading Generated Data")
    print("=" * 80 + "\n")
 
    try:
        agents, tasks_batches = load_generated_data(args.data_dir)
        print(f"✓ Data loaded!")
        print(f"  Robots: {len(agents)}")
        print(f"  Batches: {len(tasks_batches)}")
        print(f"  Tasks: {sum(len(b) for b in tasks_batches)}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
 
    # -------------------------------------------------------------
    # Determine which seed directories to evaluate
    # -------------------------------------------------------------
    if args.all_seeds:
        seed_dirs = all_seed_dirs_in_run(args.run_id)
    else:
        seed_dirs = [find_latest_run(args.seed, args.run_id)]
 
    for seed_dir in seed_dirs:
        config = load_run_config(seed_dir, args.config, args.conflict_resolution)
        eval_seeds = get_eval_seeds(config)
 
        train_seed = int(seed_dir.name.replace("seed_", ""))
        eval_dir = seed_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
 
        print("=" * 80)
        print(f"Train seed {train_seed} | Eval seeds {eval_seeds} | "
              f"Episodes per eval seed: {args.episodes}")
        print("=" * 80 + "\n")
 
        per_seed: Dict[str, Dict[str, Any]] = {str(train_seed): {}}
        per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {
            p: [] for p in POLICIES
        }
 
        for policy_name in POLICIES:
 
            print(f"  Evaluating {policy_name} across {len(eval_seeds)} eval seed(s)...")

            # Baselines pick actions via pure heuristics — there is no
            # policy in this loop at all, so there are no logits to bid
            # with. 'hungarian_bids' is meaningless here; fall back to
            # plain distance-based 'hungarian' as the closest well-defined
            # equivalent, so baselines still get a real, comparable
            # centralized-assignment resolver rather than silently
            # crashing (see set_pending_logits()'s RuntimeError in
            # environment.py for why it would otherwise).
            env_conflict_resolution = config.get("conflict_resolution", "greedy")
            if env_conflict_resolution == "hungarian_bids":
                env_conflict_resolution = "hungarian"
                print("    (conflict_resolution='hungarian_bids' has no meaning for baselines — "
                      "no policy/logits exist here — using 'hungarian' distance-based assignment instead)")

            # One env per EVAL seed (not one shared env reused across all
            # of them) — matches eval_ppo.py's approach: genuinely
            # different environment randomization per eval seed, decoupled
            # from the train seed's own value entirely (baselines don't
            # even need a trained model, so there's no reason their
            # statistical power should be bottlenecked by however many
            # train seeds happen to exist).
            per_eval_seed_results: List[Dict[str, Any]] = []
            for es in eval_seeds:
                try:
                    base_env = MultiAgentTaskEnv(
                        agents=agents,
                        tasks_batches=tasks_batches,
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
                        conflict_resolution=env_conflict_resolution,
                        candidates_sorting=config.get("candidates_sorting", "distance"),
                    )
                except Exception as e:
                    print(f"  Error creating environment for eval_seed={es}: {e}")
                    continue

                try:
                    res = evaluate_policy(
                        env=base_env,
                        config=config,
                        policy_name=policy_name,
                        n_episodes=args.episodes,
                        seed=es,
                        debug=args.debug,
                        shuffle_robots=args.shuffle_robots,
                        unique_shuffle_k=args.unique_shuffle_k,
                    )
                    per_eval_seed_results.append(res)
                except Exception as e:
                    print(f"   Error evaluating {policy_name} (eval_seed={es}): {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    try:
                        base_env.close()
                    except Exception:
                        pass

            if not per_eval_seed_results:
                print(f"   ⚠️  No successful eval-seed runs for {policy_name}, skipping.")
                continue

            res = _concat_results(per_eval_seed_results)  # pooled across ALL eval seeds
            per_seed[str(train_seed)][policy_name] = res
            per_policy_allseeds[policy_name].append(res)

            # save individual policy result
            (eval_dir / f"baseline_{policy_name}.json").write_text(
                json.dumps(res, indent=2)
            )

            print(
                f"   {policy_name}: "
                f"{res['stats']['reward_mean']:.2f} ± "
                f"{res['stats']['reward_std']:.2f}  "
                f"(n={len(res['rewards'])} episodes pooled across {len(eval_seeds)} eval seeds)\n"
            )
 
        # ---------------------------------------------------------
        # Combined summary for this seed
        # ---------------------------------------------------------
        combined_results = {
            p: _concat_results(per_policy_allseeds[p])
            for p in POLICIES
        }
        combined_results["num_episodes_per_eval_seed"] = int(args.episodes)
        combined_results["eval_seeds"] = eval_seeds
        combined_results["train_seed"] = train_seed
 
        (eval_dir / "baseline_results_all.json").write_text(
            json.dumps(combined_results, indent=2)
        )
        (eval_dir / "baseline_results_per_seed.json").write_text(
            json.dumps(per_seed, indent=2)
        )
 
        print("\n" + "=" * 80)
        print(f"Summary (Train Seed {train_seed})")
        print("=" * 80 + "\n")
 
        print(f"{'Policy':<12} {'Reward Mean':<15} {'Reward Std':<15} {'Completed':<15}")
        for policy in POLICIES:
            stats = combined_results[policy]["stats"]
            print(
                f"{policy:<12}"
                f"{stats['reward_mean']:>12.2f}     "
                f"{stats['reward_std']:>12.2f}     "
                f"{stats['completed_mean']:>12.2f}"
            )
 
        print(f"\n✓ Results saved to {eval_dir}\n")
        
if __name__ == "__main__":
    main()