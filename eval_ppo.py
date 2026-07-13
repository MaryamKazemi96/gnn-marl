# #!/usr/bin/env python3
# """
# Evaluate trained GNN-PPO model (deterministic + stochastic)
# Clean + robust + debug metrics
# """

# import argparse
# import json
# import random
# from pathlib import Path
# from typing import Any, Dict, List

# import numpy as np
# import torch
# import yaml
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# from src.environment.environment import MultiAgentTaskEnv
# from src.models.sb3_gnn_policy import RTGNNPolicy


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


# def find_latest_run(seed: int) -> Path:
#     base = Path("runs") / f"seed_{seed}"
#     if not base.exists():
#         raise FileNotFoundError(f"Seed folder not found: {base}")
#     runs = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime)
#     if not runs:
#         raise FileNotFoundError(f"No runs found in {base}")
#     return runs[-1]


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
#             K_max=config.get("K_max", 5),
#             N_max=config.get("N_max", 15),
#             E_max=config.get("E_max", 50),
#             use_xy_pickup=config.get("use_xy_pickup", False),
#             normalize_features=config.get("normalize_features", True),
#             use_node_type=config.get("use_node_type", True),
#             use_ego_robot=config.get("use_ego_robot", True),
#             use_edge_rt=config.get("use_edge_rt", False),
#             two_hop=config.get("two_hop", False),
#             two_hop_directed=config.get("two_hop_directed", False),
#             vicinity_m=config.get("vicinity_m", 20.0),
#             max_steps=config.get("max_steps", 1000),
#             max_robot_capacity=config.get("max_robot_capacity", 2),
#             max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
#             max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
#         )
#         env.reset(seed=seed)
#         return env

#     return DummyVecEnv([_init])


# # =========================================================
# # Model
# # =========================================================

# def pick_model(run_dir: Path) -> Path:
#     # prefer ppo_final
#     final = run_dir / "ppo_final.zip"
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
#                 # total_sum += int(info.get("total_action_count", 0))
#                 total_sum += int(info.get("decisions_total", 0))
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
#         print(actions_arr, 'actions_arr')
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

#     r = np.array(rewards, dtype=float)

#     total_actions_all = int(np.sum(ep_totals))
#     invalid_all = int(np.sum(ep_invalids))
#     conflict_all = int(np.sum(ep_conflicts))
#     caprej_all = int(np.sum(ep_capacity_rej))

#     invalid_rate = (invalid_all / total_actions_all) if total_actions_all > 0 else 0.0
#     conflict_rate = (conflict_all / total_actions_all) if total_actions_all > 0 else 0.0
#     caprej_rate = (caprej_all / total_actions_all) if total_actions_all > 0 else 0.0


#     decisions_all      = int(np.sum(ep_decisions))
#     noop_forced_all    = int(np.sum(ep_noop_forced))
#     noop_chosen_all    = int(np.sum(ep_noop_chosen))
#     had_candidates_all = int(np.sum(ep_had_candidates))
 

#     noop_frac_forced_rate = (noop_forced_all / decisions_all) if decisions_all > 0 else 0.0
#     noop_frac_chosen_rate = (noop_chosen_all / decisions_all) if decisions_all > 0 else 0.0
#     chosen_noop_rate_when_available = (
#         noop_chosen_all / had_candidates_all if had_candidates_all > 0 else 0.0
#     )
#     return {
#         "rewards": rewards,
#         "lengths": lengths,
#         "ticks": ticks,
#         "completed": completed,
#         "obsolete": obsolete,
#         "noop_fractions": noop_fractions,
#         "action_hists": action_hists,

#         # new debug arrays (per episode)
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

#         "stats": {
#             "reward_mean": float(r.mean()),
#             "reward_std": float(r.std()),
#             "min": float(r.min()),
#             "max": float(r.max()),
#             "completed": float(np.mean(completed)),
#             "obsolete": float(np.mean(obsolete)),
#             "noop_frac_mean": float(np.mean(noop_fractions)) if noop_fractions else 0.0,
#             "ticks_mean": float(np.mean(ticks)) if ticks else 0.0,

#             # debug summary
#             "invalid_action_total": invalid_all,
#             "total_action_count": total_actions_all,
#             "invalid_action_rate": float(invalid_rate),
#             "conflict_drop_rate": float(conflict_rate),
#             "capacity_reject_rate": float(caprej_rate),
#             "mask_zero_mean": float(np.mean(ep_mask_zeros)) if ep_mask_zeros else 0.0,

#             "r_comp_mean": float(np.mean(ep_r_comp)) if ep_r_comp else 0.0,
#             "r_wait_mean": float(np.mean(ep_r_wait)) if ep_r_wait else 0.0,
#             "r_deadline_mean": float(np.mean(ep_r_deadline)) if ep_r_deadline else 0.0,
#             "r_obsolete_mean": float(np.mean(ep_r_obsolete)) if ep_r_obsolete else 0.0,

#             "noop_frac_forced": float(noop_frac_forced_rate),
#             "noop_frac_chosen": float(noop_frac_chosen_rate),
#             "chosen_noop_rate_when_available": float(chosen_noop_rate_when_available),
#             "decisions_total": decisions_all,
#             "had_candidates_total": had_candidates_all,
#         }
#     }


# # =========================================================
# # Main
# # =========================================================

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", default="configs/training_config.yaml")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--run-dir", type=str, default=None)
#     ap.add_argument("--data-dir", default="data")
#     ap.add_argument("--episodes", type=int, default=50)
#     ap.add_argument("--output", type=str, default=None,
#                     help="Optional override output dir (defaults to run_dir/eval_results)")
#     args = ap.parse_args()

#     config = load_config(args.config)
#     set_seed(args.seed)

#     print("\n============================")
#     print(" PPO EVALUATION")
#     print("============================\n")

#     agents, tasks = load_data(args.data_dir)

#     if args.run_dir:
#         run_dir = Path(args.run_dir)
#     else:
#         run_dir = find_latest_run(args.seed)

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

#     print("Running deterministic...")
#     env_det = make_env(agents, tasks, config, args.seed)
#     det = run_eval(model, env_det, args.episodes, True)
#     env_det.close()

#     print("Running stochastic...")
#     env_sto = make_env(agents, tasks, config, args.seed + 1)
#     sto = run_eval(model, env_sto, args.episodes, False)
#     env_sto.close()

#     if args.output is not None:
#         out = Path(args.output)
#     else:
#         out = run_dir / "eval_results"

#     out.mkdir(parents=True, exist_ok=True)

#     save_json(det, out / "deterministic.json")
#     save_json(sto, out / "stochastic.json")

#     # extra compact debug summary file
#     debug_summary = {
#         "deterministic": det["stats"],
#         "stochastic": sto["stats"],
#     }
#     save_json(debug_summary, out / "debug_summary.json")

#     print("\n============================")
#     print("RESULTS")
#     print("============================")
#     print(f"Deterministic: {det['stats']['reward_mean']:.2f} ± {det['stats']['reward_std']:.2f}")
#     print(f"Stochastic:    {sto['stats']['reward_mean']:.2f} ± {sto['stats']['reward_std']:.2f}")
#     print(f"Det invalid rate: {det['stats']['invalid_action_rate']:.4f} | caprej: {det['stats']['capacity_reject_rate']:.4f} | conflict: {det['stats']['conflict_drop_rate']:.4f}")
#     print(f"Sto invalid rate: {sto['stats']['invalid_action_rate']:.4f} | caprej: {sto['stats']['capacity_reject_rate']:.4f} | conflict: {sto['stats']['conflict_drop_rate']:.4f}")
#     print(f"\nSaved → {out}\n")


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
            K_max=config.get("K_max", 5),
            N_max=config.get("N_max", 15),
            E_max=config.get("E_max", 50),
            use_xy_pickup=config.get("use_xy_pickup", False),
            normalize_features=config.get("normalize_features", True),
            use_node_type=config.get("use_node_type", True),
            use_ego_robot=config.get("use_ego_robot", True),
            use_edge_rt=config.get("use_edge_rt", False),
            two_hop=config.get("two_hop", False),
            two_hop_directed=config.get("two_hop_directed", False),
            vicinity_m=config.get("vicinity_m", 20.0),
            max_steps=config.get("max_steps", 1000),
            max_robot_capacity=config.get("max_robot_capacity", 2),
            max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
            max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
        )
        env.reset(seed=seed)
        return env
 
    return DummyVecEnv([_init])
 
 
# =========================================================
# Model
# =========================================================
 
def pick_model(run_dir: Path) -> Path:
    # prefer ppo_final
    final = run_dir / "ppo_final.zip"
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

    r = np.array(rewards, dtype=float)
 
    total_actions_all = int(np.sum(ep_totals))
    invalid_all = int(np.sum(ep_invalids))
    conflict_all = int(np.sum(ep_conflicts))
    caprej_all = int(np.sum(ep_capacity_rej))
 
    invalid_rate = (invalid_all / total_actions_all) if total_actions_all > 0 else 0.0
    conflict_rate = (conflict_all / total_actions_all) if total_actions_all > 0 else 0.0
    caprej_rate = (caprej_all / total_actions_all) if total_actions_all > 0 else 0.0
 
    decisions_all      = int(np.sum(ep_decisions))
    noop_forced_all    = int(np.sum(ep_noop_forced))
    noop_chosen_all    = int(np.sum(ep_noop_chosen))
    had_candidates_all = int(np.sum(ep_had_candidates))
 

    noop_frac_forced_rate = (noop_forced_all / decisions_all) if decisions_all > 0 else 0.0
    noop_frac_chosen_rate = (noop_chosen_all / decisions_all) if decisions_all > 0 else 0.0
    chosen_noop_rate_when_available = (
        noop_chosen_all / had_candidates_all if had_candidates_all > 0 else 0.0
    )
    return {
        "rewards": rewards,
        "lengths": lengths,
        "ticks": ticks,
        "completed": completed,
        "obsolete": obsolete,
        "noop_fractions": noop_fractions,
        "action_hists": action_hists,
 
        # new debug arrays (per episode)
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
 
        "stats": {
            "reward_mean": float(r.mean()),
            "reward_std": float(r.std()),
            "min": float(r.min()),
            "max": float(r.max()),
            "completed": float(np.mean(completed)),
            "obsolete": float(np.mean(obsolete)),
            "noop_frac_mean": float(np.mean(noop_fractions)) if noop_fractions else 0.0,
            "ticks_mean": float(np.mean(ticks)) if ticks else 0.0,
 
            # debug summary
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
    }
 
 
# =========================================================
# Main
# =========================================================
 
# =========================================================
# Main
# =========================================================
 
def evaluate_one_seed(seed: int, run_dir: Path, config: Dict, agents, tasks,
                       episodes: int, output_override: str = None):
    print(f"\n---- seed {seed} ----")
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
 
    print("Running deterministic...")
    env_det = make_env(agents, tasks, config, seed)
    det = run_eval(model, env_det, episodes, True)
    env_det.close()
 
    print("Running stochastic...")
    env_sto = make_env(agents, tasks, config, seed + 1)
    sto = run_eval(model, env_sto, episodes, False)
    env_sto.close()
 
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
    print(f"RESULTS — seed {seed}")
    print("============================")
    print(f"Deterministic: {det['stats']['reward_mean']:.2f} ± {det['stats']['reward_std']:.2f}")
    print(f"Stochastic:    {sto['stats']['reward_mean']:.2f} ± {sto['stats']['reward_std']:.2f}")
    print(f"Det invalid rate: {det['stats']['invalid_action_rate']:.4f} | caprej: {det['stats']['capacity_reject_rate']:.4f} | conflict: {det['stats']['conflict_drop_rate']:.4f}")
    print(f"Sto invalid rate: {sto['stats']['invalid_action_rate']:.4f} | caprej: {sto['stats']['capacity_reject_rate']:.4f} | conflict: {sto['stats']['conflict_drop_rate']:.4f}")
    print(f"\nSaved → {out}\n")
 
    return det, sto
 
 
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
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--output", type=str, default=None,
                    help="Optional override output dir (defaults to run_dir/eval_results). "
                         "Ignored (per-seed subfolders used instead) when --all-seeds is set.")
    args = ap.parse_args()
 
    config = load_config(args.config)
 
    print("\n============================")
    print(" PPO EVALUATION")
    print("============================\n")
 
    agents, tasks = load_data(args.data_dir)
 
    if args.all_seeds:
        seed_dirs = all_seed_dirs_in_run(args.run_id)
        print(f"Evaluating {len(seed_dirs)} seeds: {[d.name for d in seed_dirs]}")
        results = {}
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.replace("seed_", ""))
            set_seed(seed)
            det, sto = evaluate_one_seed(seed, seed_dir, config, agents, tasks, args.episodes)
            results[seed] = {"det_mean": det["stats"]["reward_mean"], "sto_mean": sto["stats"]["reward_mean"]}
 
        print("\n============================")
        print("SWEEP SUMMARY (all seeds)")
        print("============================")
        for seed, r in results.items():
            print(f"seed {seed:>4}: det={r['det_mean']:8.2f}  sto={r['sto_mean']:8.2f}")
        det_means = [r["det_mean"] for r in results.values()]
        sto_means = [r["sto_mean"] for r in results.values()]
        print(f"\nAcross seeds — det: {np.mean(det_means):.2f} ± {np.std(det_means):.2f} | "
              f"sto: {np.mean(sto_means):.2f} ± {np.std(sto_means):.2f}")
        return
 
    set_seed(args.seed)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run(args.seed, args.run_id)
 
    evaluate_one_seed(args.seed, run_dir, config, agents, tasks, args.episodes, args.output)
 
 
if __name__ == "__main__":
    main()