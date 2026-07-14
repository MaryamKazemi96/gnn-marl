import os
import sys
import json
import platform
import datetime
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch as th
import yaml
import random 

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.environment import MultiAgentTaskEnv
from src.models.sb3_gnn_policy import RTGNNPolicy


# ============================================================================
# Tee — mirror stdout to file
# ============================================================================

class Tee:
    def __init__(self, filename: str, mode: str = "w", base_stdout=None):
        self.file   = open(filename, mode)
        self.stdout = base_stdout if base_stdout is not None else sys.stdout
        sys.stdout  = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


# ============================================================================
# Utils
# ============================================================================

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_config(config_path: str) -> Optional[Dict]:
    if not Path(config_path).exists():
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_generated_data(data_dir: str, n_batches: Optional[int] = None):
    print("Loading data...")
    data_path = Path(data_dir)

    agents_file = data_path / "agents.npy"
    if not agents_file.exists():
        raise FileNotFoundError(f"Agents file not found: {agents_file}")

    agents = np.load(agents_file, allow_pickle=True)
    print(f"  ✓ Agents: {agents.shape}")

    tasks_batches, idx = [], 0
    while True:
        if n_batches is not None and idx >= n_batches:
            break
        f = data_path / f"tasks_batch_{idx}.npy"
        if not f.exists():
            break
        batch = np.load(f, allow_pickle=True)
        tasks_batches.append(batch)
        print(f"  ✓ Batch {idx}: {batch.shape}")
        idx += 1

    if not tasks_batches:
        raise FileNotFoundError(f"No task batches found in {data_path}")

    print(f"  ✓ Total: {len(agents)} robots, {len(tasks_batches)} batches")
    return agents, tasks_batches


def _latest_model_path(model_dir: str):
    """Find latest saved model by timestep number."""
    import re, glob
    pattern   = os.path.join(model_dir, "model_episode*_ts*.zip")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"model_episode(\d+)_ts(\d+)\.zip$", os.path.basename(path))
        if m:
            candidates.append((int(m.group(2)), int(m.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No saved models in {model_dir}")
    ts, ep, path = sorted(candidates)[-1]
    return path, ep, ts

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    # Keep minimal changes; no forced deterministic flags unless you want strict reproducibility.


def get_seed_list(config: Dict) -> List[int]:
    # Preferred: seeds: [1,2,3]
    if "seeds" in config and config["seeds"] is not None:
        seeds = [int(s) for s in config["seeds"]]
        if len(seeds) == 0:
            raise ValueError("Config key 'seeds' is empty.")
        return seeds
    # Fallback: seed: 42
    return [int(config.get("seed", 42))]

# ============================================================================
# Callbacks
# ============================================================================
class TrainingLogCallback(BaseCallback):
    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_freq != 0:
            return True

        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]

            ep_total = max(1, int(info.get("ep_total_action_count", 0)))
            inv_rate = float(info.get("ep_invalid_action_count", 0)) / ep_total
            cap_rate = float(info.get("ep_capacity_rejected_count", 0)) / ep_total
            cfl_rate = float(info.get("ep_conflict_dropped_count", 0)) / ep_total

            decisions_total = max(1, int(info.get("ep_decisions_total", 0)))
            had_candidates  = max(1, int(info.get("ep_had_candidates_count", 0)))
            noop_forced_rate = int(info.get("ep_noop_forced_count", 0)) / decisions_total
            noop_chosen_rate = int(info.get("ep_noop_chosen_count", 0)) / decisions_total
            chosen_noop_rate_when_available = int(info.get("ep_noop_chosen_count", 0)) / had_candidates

            print(
                f"TS:{self.num_timesteps:7d} | "
                f"R:{ep.get('r', float('nan')):8.2f} | L:{ep.get('l', -1):5d} | "
                f"C:{info.get('completed_count', '?'):3} O:{info.get('obsolete_count', '?'):3} | "
                f"Inv:{inv_rate:6.3f} CapRej:{cap_rate:6.3f} Cfl:{cfl_rate:6.3f} | "
                f"NoopFrc:{noop_forced_rate:6.3f} NoopChs:{noop_chosen_rate:6.3f} "
                f"ChsWhenAvail:{chosen_noop_rate_when_available:6.3f} | "
                f"r_comp:{info.get('ep_r_comp', 0.0):7.2f} "
                f"r_wait:{info.get('ep_r_wait', 0.0):7.2f} "
                f"r_dead:{info.get('ep_r_deadline', 0.0):7.2f} "
                f"r_obs:{info.get('ep_r_obsolete', 0.0):7.2f}"
            )
        return True

class CheckpointCallback(BaseCallback):
    """Save model checkpoints — mirrors colleague's model_episode{ep}_ts{ts}.zip naming."""

    def __init__(self, save_freq: int = 10000, save_path: str = "./models"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.ep_idx = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_idx += 1

        if self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"model_episode{self.ep_idx}_ts{self.num_timesteps}.zip"
            self.model.save(str(path))
            print(f"  ✓ Checkpoint: {path.name}")
        return True


# ============================================================================
# Main
# ============================================================================

def run_single_seed(seed: int, config: Dict, continue_training: bool, run_id: str, base_stdout=None):
    set_global_seed(seed)

    two_hop = config.get("two_hop", False)

    # New layout: runs/run_{run_id}/seed_{seed}/  — one run_id per training
    # session/sweep, with every seed trained in that sweep grouped under it.
    # (Old layout was runs/seed_{seed}/run_{run_id}/, which scattered a
    # sweep's seeds across separate seed-keyed folders instead of keeping
    # the whole sweep together.)
    run_dir   = Path("runs") / f"run_{run_id}" / f"seed_{seed}"
    model_dir = run_dir / "models"
    tb_dir    = run_dir / "tensorboard"
    log_dir   = run_dir / "logs"
    for d in [run_dir, model_dir, tb_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "train_output.txt"
    tee = Tee(str(output_file), mode="a" if continue_training else "w", base_stdout=base_stdout)
    print(f" Seed      : {seed}")
    print(f" Run dir   : {run_dir}")
    print(f" Model dir : {model_dir}")

    try:
        agents, tasks_batches = load_generated_data(
            config.get("data_dir", "data"),
            n_batches=config.get("n_batches"),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return

    print("\nCreating environment...")
    print('two hop flag',two_hop)
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
            two_hop=two_hop,
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
        )

        feature_dim = base_env.F
        print(f"   Feature dim  : {feature_dim}")
        print(f"   Action space : {base_env.action_space}")
        print(f"   two_hop      : {two_hop}")
        print(f"   vicinity_m   : {config.get('vicinity_m', 40.0)}")

        # Monitor with episode_reward as tracked keyword
        # action_mask stays in info as-is — RTGNNPolicy handles masking internally
        env = Monitor(
            base_env,
            filename=str(log_dir / "monitor.csv"),
            info_keywords=("completed_count", "obsolete_count"),
        )

    except Exception as e:
        print(f"Error creating environment: {e}")
        traceback.print_exc()
        return

    # ── Policy kwargs ─────────────────────────────────────────────────────────
    policy_kwargs = dict(
        in_dim=feature_dim,
        hidden=int(config.get("hidden", 64)),
        k_max=config["K_max"],
        logit_temperature=float(config.get("logit_temperature", 1.0)),
        noop_init=float(config.get("noop_init", -1.0)),
        freeze_noop_logit=bool(config.get("freeze_noop_logit", False)),
        edge_dim=0,
        use_competitor_fusion=bool(two_hop and config.get("two_hop_arch", "comp_corr") == "comp_corr"),
        use_two_hop_actor=bool(two_hop and config.get("two_hop_arch", "") == "plain"),
        use_two_hop_critic=bool(two_hop and config.get("two_hop_critic", False)),
        backbone=config.get("backbone", "sage"),
        critic_aggregation=config.get("critic_aggregation", "joint_mean"),
        gnn_kwargs={"layers": int(config.get("gnn_layers", 2))},
    )

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    try:
        if continue_training:
            latest_path, last_ep, last_ts = _latest_model_path(str(model_dir))
            print(f"[CONTINUE] Loading: {latest_path}")
            model = PPO.load(latest_path, env=env, device=device)
            model.num_timesteps = int(last_ts)
            print(f"  ✓ Resuming from episode {last_ep}, timestep {last_ts}")
        else:
            model = PPO(
                RTGNNPolicy,
                env,
                policy_kwargs=policy_kwargs,
                n_steps=config["ppo_steps"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                gamma=config.get("gamma", 0.99),
                clip_range=config.get("clip_range", 0.2),
                clip_range_vf=config.get("clip_range_vf", None),
                vf_coef=config.get("vf_coef", 0.5),
                ent_coef=config["ent_coef"],
                gae_lambda=config.get("gae_lambda", 0.95),
                n_epochs=config.get("n_epochs", 5),
                max_grad_norm=config.get("max_grad_norm", 0.5),
                target_kl=config.get("target_kl", None),
                seed=seed,
                verbose=1,
                device=device,
                tensorboard_log=str(tb_dir),
            )
            # ------------------------optimizer check
            actor_ids = {id(p) for p in model.policy.gnn_ac.parameters()} | {id(model.policy.noop_logit)}
            opt_ids = {id(p) for g in model.policy.optimizer.param_groups for p in g["params"]}
            missing = actor_ids - opt_ids
            assert not missing, f"{len(missing)} actor params NOT in optimizer — this is the bug"
            print(f"actor params: {len(actor_ids)}, in optimizer: {len(actor_ids & opt_ids)}")

            print("\n=== ACTOR PARAMETERS ===")
            for name, p in model.policy.gnn_ac.named_parameters():
                print(f"{name:40s} requires_grad={p.requires_grad} shape={tuple(p.shape)}")
            print("========================\n")


            # ------------------------
            # Force noop_logit to config value (matches colleague's pattern)
            model.policy.noop_logit.data.fill_(float(config.get("noop_init", -1.0)))
            print(f"   noop_logit set to: {model.policy.noop_logit.item():.3f}")

            # Save init model (matches colleague's pattern)
            init_path = model_dir / "model_episode0_ts0.zip"
            model.save(str(init_path))
            print(f"  ✓ Init model saved: {init_path.name}")

    except Exception as e:
        print(f"Error creating model: {e}")
        traceback.print_exc()
        return

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 10000),
        save_path=str(model_dir),
    )

    final_path = run_dir / "ppo_final.zip"
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[
                TrainingLogCallback(log_freq=100),
                checkpoint_cb,
            ],
            reset_num_timesteps=not continue_training,
        )
        print("\n✓ Training complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
    finally:
        try:
            model.save(str(final_path))
            print(f"✓ Final model saved: {final_path}")
        except Exception as e:
            print(f"Save failed: {e}")


    #metadata
    # Metadata
    metadata = {
    "seed": seed,
    "run_id": run_id,
    "config": config,
    "policy_kwargs": policy_kwargs,
    "locals": {
        k: str(v)
        for k, v in locals().items()
        if k not in ["agents", "tasks_batches"]
    }
}
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  Seed       : {seed}")
    print(f"  Run dir    : {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"  Finished   : {datetime.datetime.now().isoformat()}")
    print("=" * 80 + "\n")

    env.close()
    tee.close()  # restore sys.stdout so the next seed's Tee doesn't nest

    return run_dir


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            type=str,  default="configs/training_config.yaml")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--seed",              type=int,  default=None,
                         help="Train a single seed only, overriding config's seed/seeds list.")
    parser.add_argument("--run-id",            type=str,  default=None,
                         help="Reuse an existing runs/run_{id}/ folder (e.g. to add seeds to "
                              "a previous sweep or resume with --continue-training). "
                              "Defaults to a fresh UTC timestamp.")
    args = parser.parse_args()

    continue_training = args.continue_training

    print("=" * 80)
    print("GNN-PPO Training")
    print("=" * 80 + "\n")

    config = load_config(args.config)
    if config is None:
        raise FileNotFoundError(f"Could not load config: {args.config}")

    seeds = [args.seed] if args.seed is not None else get_seed_list(config)
    run_id = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    print(f"Seeds to run: {seeds}\n")

    base_stdout = sys.stdout
    run_dirs = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*80}\n# SEED {seed}  ({i+1}/{len(seeds)})\n{'#'*80}\n")
        try:
            run_dir = run_single_seed(seed, config, continue_training, run_id, base_stdout=base_stdout)
            run_dirs.append(run_dir)
        except Exception as e:
            # One bad seed shouldn't abort the rest of the sweep.
            sys.stdout = base_stdout
            print(f"⚠️  Seed {seed} failed: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"All seeds finished. Runs:")
    for rd in run_dirs:
        print(f"  {rd}")
    print("=" * 80)


if __name__ == "__main__":
    main()