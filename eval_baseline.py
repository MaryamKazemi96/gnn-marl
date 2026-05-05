
# #!/usr/bin/env python3
# """
# Warehouse baselines, colleague-style.

# Policies (matches colleague logic):
# - greedy: choose slot 0 if valid else NOOP
# - random: choose uniformly among allowed actions in mask (includes NOOP if allowed)
# - unique: choose smallest k that is valid and has unseen task_id; else NOOP; fallback to greedy if no cand_ids

# Environment:
# - MultiDiscrete([K+1]*R), NOOP index is K_max (last action)
# - decision steps every assignment_interval (macro-action reused between decision steps)

# Usage:
#   python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --output-dir checkpoints_ppo
#   python3 eval_baselines_warehouse.py --config configs/training_config.yaml --episodes 20 --seed 42 --debug

# Optional knobs to make greedy/unique weaker (while keeping same rule shape):
#   --shuffle-robots        # randomize robot iteration order each decision step
#   --unique-shuffle-k      # randomize the k scan order inside unique policy each decision step
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np

# from train_ppo import make_env, load_config


# POLICIES = ["random", "greedy", "unique"]


# def _get_mask(obs: Any, info: Any) -> Optional[np.ndarray]:
#     if isinstance(info, dict) and "action_mask" in info:
#         return np.asarray(info["action_mask"])
#     if isinstance(obs, dict) and "action_mask" in obs:
#         return np.asarray(obs["action_mask"])
#     return None


# def _get_last_cand_task_ids(env) -> Optional[List[List[int]]]:
#     try:
#         env0 = env.unwrapped
#         return getattr(env0, "_last_cand_task_ids", None)
#     except Exception:
#         return None


# def _infer_decision_interval(env, config: Dict) -> int:
#     # prefer wrapper attr, else config
#     for e in [env, getattr(env, "unwrapped", None)]:
#         if e is None:
#             continue
#         if hasattr(e, "assignment_interval"):
#             try:
#                 v = int(getattr(e, "assignment_interval"))
#                 if v > 0:
#                     return v
#             except Exception:
#                 pass
#     return int(config.get("environment", {}).get("assignment_interval", 1))


# def greedy_nearest_action(mask: np.ndarray, R: int, NOOP: int) -> np.ndarray:
#     """Exactly like colleague: slot 0 if valid else NOOP."""
#     a = np.full((R,), NOOP, dtype=np.int64)
#     for r in range(R):
#         if mask[r, 0] == 1:
#             a[r] = 0
#         else:
#             a[r] = NOOP
#     return a


# def random_valid_action(mask: np.ndarray, R: int, NOOP: int, rng: np.random.Generator) -> np.ndarray:
#     """Exactly like colleague: choose uniformly among allowed actions (including NOOP if mask allows it)."""
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
#     Exactly like colleague:
#     - uses env.unwrapped._last_cand_task_ids
#     - assigns first valid k whose task_id not yet chosen
#     - fallback to greedy_nearest_action if cand_ids missing
#     """
#     cand_ids = _get_last_cand_task_ids(env)
#     if cand_ids is None:
#         return greedy_nearest_action(mask, R, NOOP)

#     chosen = set()
#     a = np.full((R,), NOOP, dtype=np.int64)

#     for r in robot_order:
#         if shuffle_k:
#             ks = np.arange(K_max, dtype=int)
#             rng.shuffle(ks)
#         else:
#             ks = range(K_max)

#         for k in ks:
#             if mask[r, k] != 1:
#                 continue
#             try:
#                 task_id = int(cand_ids[r][k])
#             except Exception:
#                 continue
#             if task_id < 0:
#                 continue
#             if task_id in chosen:
#                 continue
#             chosen.add(task_id)
#             a[r] = int(k)
#             break

#     return a


# def evaluate_policy(
#     env,
#     config: Dict,
#     policy_name: str,
#     n_episodes: int,
#     seed: int,
#     debug: bool,
#     shuffle_robots: bool,
#     unique_shuffle_k: bool,
# ) -> Dict[str, Any]:
#     rng = np.random.default_rng(seed)

#     action_space = env.action_space
#     assert hasattr(action_space, "nvec"), "Expected MultiDiscrete action space"
#     R = int(len(action_space.nvec))
#     Kp1 = int(action_space.nvec[0])
#     K_max = Kp1 - 1
#     NOOP = K_max

#     decision_interval = _infer_decision_interval(env, config)

#     ep_rewards: List[float] = []
#     ep_completions: List[int] = []
#     ep_obsolete: List[int] = []
#     ep_lengths: List[int] = []

#     for ep in range(n_episodes):
#         obs, info = env.reset(seed=seed + ep)
#         done = False
#         ep_rew = 0.0
#         ep_len = 0

#         last_action = np.full((R,), NOOP, dtype=np.int64)

#         while not done:
#             if ep_len % max(1, decision_interval) == 0:
#                 mask = _get_mask(obs, info)
#                 if mask is None:
#                     raise RuntimeError("No action_mask found in info or obs.")
#                 mask = (np.asarray(mask) == 1).astype(np.int32)

#                 # robot order (optional weakening knob)
#                 if shuffle_robots:
#                     robot_order = np.arange(R, dtype=int)
#                     rng.shuffle(robot_order)
#                 else:
#                     robot_order = np.arange(R, dtype=int)

#                 if policy_name == "greedy":
#                     # colleague logic is always r=0..R-1; robot_order is only for unique.
#                     last_action = greedy_nearest_action(mask, R, NOOP)

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
#                 else:
#                     raise ValueError(f"Unknown policy: {policy_name}")

#                 if debug and ep == 0:
#                     valid_slots = mask[:, :NOOP].sum(axis=1).astype(int).tolist()
#                     print("[DEBUG] valid_slots_per_robot(excl NOOP):", valid_slots)
#                     print("[DEBUG] cand_present:", _get_last_cand_task_ids(env) is not None)
#                     print("[DEBUG] action:", last_action.tolist())

#             obs, reward, terminated, truncated, info = env.step(last_action)
#             ep_rew += float(reward)
#             ep_len += 1
#             done = bool(terminated or truncated)

#         completed = int(info.get("episode_completed", 0)) if isinstance(info, dict) else 0
#         obsolete = int(info.get("episode_obsolete", 0)) if isinstance(info, dict) else 0

#         ep_rewards.append(ep_rew)
#         ep_completions.append(completed)
#         ep_obsolete.append(obsolete)
#         ep_lengths.append(ep_len)

#     rr = np.asarray(ep_rewards, dtype=float)
#     return {
#         "policy": policy_name,
#         "decision_interval": int(decision_interval),
#         "rewards": [float(x) for x in ep_rewards],
#         "completions": [int(x) for x in ep_completions],
#         "obsolete": [int(x) for x in ep_obsolete],
#         "lengths": [int(x) for x in ep_lengths],
#         "stats": {
#             "reward_mean": float(rr.mean()) if rr.size else 0.0,
#             "reward_std": float(rr.std()) if rr.size else 0.0,
#             "completion_mean": float(np.mean(ep_completions)) if ep_completions else 0.0,
#             "obsolete_mean": float(np.mean(ep_obsolete)) if ep_obsolete else 0.0,
#             "length_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
#         },
#     }


# def _concat_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#     out = {"rewards": [], "completions": [], "obsolete": [], "lengths": []}
#     for r in results_list:
#         for k in out.keys():
#             out[k].extend(r.get(k, []))

#     rr = np.asarray(out["rewards"], dtype=float)
#     out["stats"] = {
#         "reward_mean": float(rr.mean()) if rr.size else 0.0,
#         "reward_std": float(rr.std()) if rr.size else 0.0,
#         "completion_mean": float(np.mean(out["completions"])) if out["completions"] else 0.0,
#         "obsolete_mean": float(np.mean(out["obsolete"])) if out["obsolete"] else 0.0,
#         "length_mean": float(np.mean(out["lengths"])) if out["lengths"] else 0.0,
#     }
#     return out


# def main() -> None:
#     ap = argparse.ArgumentParser(description="Evaluate baseline policies (warehouse)")
#     ap.add_argument("--config", type=str, default="configs/training_config.yaml")
#     ap.add_argument("--episodes", type=int, default=20)
#     ap.add_argument("--output-dir", type=str, default="checkpoints_ppo")
#     ap.add_argument("--seed", type=int, default=None)
#     ap.add_argument("--debug", action="store_true")

#     # weakening knobs (optional)
#     ap.add_argument("--shuffle-robots", action="store_true",
#                     help="Shuffle robot iteration order on each decision step (makes unique less deterministic).")
#     ap.add_argument("--unique-shuffle-k", action="store_true",
#                     help="Shuffle k scan order inside unique baseline on each decision step (makes unique weaker).")

#     args = ap.parse_args()

#     config = load_config(args.config)

#     seeds = config.get("experiment", {}).get("seeds", None)
#     if args.seed is not None:
#         seeds = [int(args.seed)]
#     elif not seeds:
#         seeds = [int(config["experiment"]["seed"])]
#     else:
#         seeds = [int(s) for s in seeds]

#     out_dir = Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     per_seed: Dict[str, Dict[str, Any]] = {}
#     per_policy_allseeds: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICIES}

#     for seed in seeds:
#         print("\n" + "=" * 70)
#         print(f"Seed {seed} | episodes per policy: {args.episodes}")
#         print("=" * 70)

#         per_seed[str(seed)] = {}

#         for policy_name in POLICIES:
#             env = make_env(config, seed=seed)
#             res = evaluate_policy(
#                 env=env,
#                 config=config,
#                 policy_name=policy_name,
#                 n_episodes=args.episodes,
#                 seed=seed,
#                 debug=args.debug,
#                 shuffle_robots=args.shuffle_robots,
#                 unique_shuffle_k=args.unique_shuffle_k,
#             )
#             per_seed[str(seed)][policy_name] = res
#             per_policy_allseeds[policy_name].append(res)

#             try:
#                 env.close()
#             except Exception:
#                 pass

#             p = out_dir / f"baseline_{policy_name}_seed_{seed}.json"
#             p.write_text(json.dumps(res, indent=2))
#             print(f"✓ Saved {p}")

#     combined_results = {p: _concat_results(per_policy_allseeds[p]) for p in POLICIES}
#     combined_results["num_episodes_per_seed"] = int(args.episodes)
#     combined_results["seeds"] = seeds
#     (out_dir / "baseline_results_all.json").write_text(json.dumps(combined_results, indent=2))
#     (out_dir / "baseline_results_per_seed.json").write_text(json.dumps(per_seed, indent=2))

#     print(f"\n✓ Saved combined results to {out_dir / 'baseline_results_all.json'}")


# if __name__ == "__main__":
#     main()

"""
Evaluate baseline policies (random, greedy, unique) on MultiAgentTaskEnv.

Policies:
- greedy: pick candidate slot 0 if valid, else NOOP
- random: uniformly sample among all valid actions
- unique: greedy with task deduplication using _last_cand_task_ids
"""
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
from train_ppo import load_generated_data


POLICIES = ["random", "greedy", "unique"]


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def greedy_nearest_action(mask: np.ndarray, R: int, NOOP: int) -> np.ndarray:
    """
    Greedy policy: select candidate slot 0 if valid, else NOOP.
    
    This assumes candidate slots are ordered by preference (nearest first).
    """
    a = np.full((R,), NOOP, dtype=np.int64)
    for r in range(R):
        if mask[r, 0] == 1:
            a[r] = 0
        else:
            a[r] = NOOP
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
        return greedy_nearest_action(mask, R, NOOP)

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
                    last_action = greedy_nearest_action(mask, R, NOOP)

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
        print(f"❌ Error loading data: {e}")
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
                print(f"  ❌ Error creating environment: {e}")
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
                print(f"  ✓ {policy_name}: {res['stats']['reward_mean']:.2f} ± {res['stats']['reward_std']:.2f}\n")

            except Exception as e:
                print(f"  ❌ Error evaluating {policy_name}: {e}")
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
    print("-" * 60)
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


if __name__ == "__main__":
    main()