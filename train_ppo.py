import os
import sys
import json
import platform
import datetime
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch as th
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.environment import MultiAgentTaskEnv
from src.models.sb3_gnn_policy import RTGNNPolicy


# ============================================================================
# Tee — mirror stdout to file
# ============================================================================

class Tee:
    def __init__(self, filename: str, mode: str = "w"):
        self.file   = open(filename, mode)
        self.stdout = sys.stdout
        sys.stdout  = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


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


# ============================================================================
# Callbacks
# ============================================================================

class TrainingLogCallback(BaseCallback):
    """Console logging of episode metrics."""

    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_freq != 0:
            return True

        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                print(
                    f"  Timesteps: {self.num_timesteps:7d} | "
                    f"Reward: {ep.get('r', float('nan')):8.2f} | "
                    f"Length: {ep.get('l', -1):6d} | "
                    f"Completed: {info.get('completed_count', '?')} | "
                    f"Obsolete: {info.get('obsolete_count', '?')}"
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

def main():
    # ── Args / continue flag ─────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            type=str,  default="configs/training_config.yaml")
    parser.add_argument("--continue-training", action="store_true")
    args = parser.parse_args()

    continue_training = args.continue_training

    # Mirror stdout to file (append if continuing)
    # Tee("train_output.txt", mode="a" if continue_training else "w")

    print("=" * 80)
    print("GNN-PPO Training")
    print("=" * 80 + "\n")

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config(args.config) or {
        "data_dir":           "data",
        "n_batches":          None,
        "K_max":              5,
        "N_max":              15,
        "E_max":              50,
        "use_xy_pickup":      False,
        "normalize_features": True,
        "use_node_type":      True,
        "use_ego_robot":      True,
        "use_edge_rt":        False,
        "two_hop":            False,
        "two_hop_directed":   False,
        "vicinity_m":         40.0,
        "max_steps":          1000,
        "max_robot_capacity": 2,
        "max_wait_delay_s":   600.0,
        "max_travel_delay_s": 3600.0,
        "ppo_steps":          2048,
        "batch_size":         64,
        "learning_rate":      3e-4,
        "total_timesteps":    100000,
        "checkpoint_freq":    10000,
        "seed":               42,
        "noop_init":          -1.0,
        "logit_temperature":  1.0,
        "hidden":             64,
        "gnn_layers":         2,
        "model_save_dir":     "models",
    }

    seed   = int(config.get("seed", 42))
    run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    two_hop = config.get("two_hop", False)

    # ── Directories ───────────────────────────────────────────────────────────
    run_dir   = Path("runs") / f"seed_{seed}" / f"run_{run_id}"
    model_dir = Path(config.get("model_save_dir", "models"))
    tb_dir    = run_dir / "tensorboard"
    log_dir   = run_dir / "logs"
    for d in [run_dir, model_dir, tb_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    output_file = run_dir / "train_output.txt"
    Tee(str(output_file), mode="a" if continue_training else "w")
    print(f"✓ Run dir   : {run_dir}")
    print(f"✓ Model dir : {model_dir}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "run_id":    run_id,
        "seed":      seed,
        "continue":  continue_training,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "git":       get_git_commit(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "cuda":   th.cuda.is_available(),
        },
        "config": config,
    }, indent=2))

    # ── Data ─────────────────────────────────────────────────────────────────
    try:
        agents, tasks_batches = load_generated_data(
            config.get("data_dir", "data"),
            n_batches=config.get("n_batches"),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return

    # ── Environment ───────────────────────────────────────────────────────────
    print("\nCreating environment...")
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
            vicinity_m=config.get("vicinity_m", 40.0),
            max_steps=config.get("max_steps", 1000),
            max_robot_capacity=config.get("max_robot_capacity", 2),
            max_wait_delay_s=config.get("max_wait_delay_s", 600.0),
            max_travel_delay_s=config.get("max_travel_delay_s", 3600.0),
        )

        feature_dim = base_env.F
        print(f"  ✓ Feature dim  : {feature_dim}")
        print(f"  ✓ Action space : {base_env.action_space}")
        print(f"  ✓ two_hop      : {two_hop}")
        print(f"  ✓ vicinity_m   : {config.get('vicinity_m', 40.0)}")

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

    # ── Model — new or continue ───────────────────────────────────────────────
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
                gamma=0.99,
                clip_range=0.2,
                vf_coef=0.5,
                ent_coef=0.01,
                gae_lambda=0.95,
                n_epochs=10,
                verbose=1,
                device=device,
                tensorboard_log=str(tb_dir),
            )

            # Force noop_logit to config value (matches colleague's pattern)
            model.policy.noop_logit.data.fill_(float(config.get("noop_init", -1.0)))
            print(f"  ✓ noop_logit set to: {model.policy.noop_logit.item():.3f}")

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

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  Run dir    : {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"  Finished   : {datetime.datetime.now().isoformat()}")
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()