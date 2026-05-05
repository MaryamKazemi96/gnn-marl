# """
# Training script for GNN-PPO with MultiAgentTaskEnv + TensorBoard + run tracking
# """

# import os
# import json
# import yaml
# import platform
# import datetime
# import subprocess
# from pathlib import Path

# import numpy as np
# import torch as th

# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback

# from src.environment.environment import MultiAgentTaskEnv
# from src.models.sb3_gnn_policy import RTGNNPolicy


# # =========================================================================
# # Utils
# # =========================================================================

# def get_git_commit() -> str:
#     try:
#         return subprocess.check_output(
#             ["git", "rev-parse", "HEAD"],
#             stderr=subprocess.DEVNULL
#         ).decode().strip()
#     except Exception:
#         return "unknown"


# def load_config(config_path: str):
#     if not Path(config_path).exists():
#         return None
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)


# # =========================================================================
# # Callback
# # =========================================================================

# class TrainingLogCallback(BaseCallback):
#     def __init__(self, log_freq: int = 100, verbose: int = 1):
#         super().__init__(verbose)
#         self.log_freq = log_freq

#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos", [])

#         for info in infos:
#             if "episode" in info:
#                 ep = info["episode"]
#                 print(
#                     f"Timesteps: {self.num_timesteps} | "
#                     f"Reward: {ep['r']:.2f} | "
#                     f"Length: {ep['l']}"
#                 )
#         return True


# # =========================================================================
# # Data loading
# # =========================================================================

# def load_generated_data(data_dir: str, n_batches=None):
#     data_path = Path(data_dir)

#     agents = np.load(data_path / "agents.npy", allow_pickle=True)

#     tasks_batches = []
#     idx = 0

#     while True:
#         if n_batches is not None and idx >= n_batches:
#             break

#         file = data_path / f"tasks_batch_{idx}.npy"
#         if not file.exists():
#             break

#         tasks_batches.append(np.load(file, allow_pickle=True))
#         idx += 1

#     if not tasks_batches:
#         raise FileNotFoundError("No task batches found")

#     return agents, tasks_batches


# # =========================================================================
# # Main
# # =========================================================================

# def main():
#     print("=" * 80)
#     print("GNN-PPO Training (TensorBoard + Run Tracking)")
#     print("=" * 80)

#     # ---------------------------------------------------------
#     # Config
#     # ---------------------------------------------------------
#     config_path = "configs/training_config.yaml"
#     config = load_config(config_path)

#     if config is None:
#         config = {
#             "data_dir": "../data",
#             "K_max": 5,
#             "N_max": 15,
#             "E_max": 50,
#             "use_xy_pickup": False,
#             "normalize_features": True,
#             "use_node_type": True,
#             "use_ego_robot": True,
#             "use_edge_rt": False,
#             "two_hop": False,
#             "vicinity_m": 20.0,
#             "max_steps": 1000,
#             "ppo_steps": 2048,
#             "batch_size": 64,
#             "learning_rate": 3e-4,
#             "total_timesteps": 100000,
#             "seed": 42,
#         }

#     seed = config.get("seed", 42)
#     run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

#     # ---------------------------------------------------------
#     # Run directories
#     # ---------------------------------------------------------
#     run_dir = Path("runs") / f"seed_{seed}" / f"run_{run_id}"
#     model_dir = run_dir / "models"
#     tb_dir = run_dir / "tensorboard"
#     log_dir = run_dir / "logs"

#     model_dir.mkdir(parents=True, exist_ok=True)
#     tb_dir.mkdir(parents=True, exist_ok=True)
#     log_dir.mkdir(parents=True, exist_ok=True)

#     # ---------------------------------------------------------
#     # Save metadata
#     # ---------------------------------------------------------
#     run_meta = {
#         "run_id": run_id,
#         "seed": seed,
#         "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
#         "git_commit": get_git_commit(),
#         "platform": {
#             "python": platform.python_version(),
#             "system": platform.platform(),
#             "cuda": str(th.cuda.is_available()),
#         },
#         "config": config,
#     }

#     (run_dir / "run_metadata.json").write_text(
#         json.dumps(run_meta, indent=2),
#         encoding="utf-8"
#     )

#     # ---------------------------------------------------------
#     # Load data
#     # ---------------------------------------------------------
#     agents, tasks_batches = load_generated_data("data")

#     # ---------------------------------------------------------
#     # Environment
#     # ---------------------------------------------------------
#     base_env = MultiAgentTaskEnv(
#         agents=agents,
#         tasks_batches=tasks_batches,
#         K_max=config["K_max"],
#         N_max=config["N_max"],
#         E_max=config["E_max"],
#         use_xy_pickup=config["use_xy_pickup"],
#         normalize_features=config["normalize_features"],
#         use_node_type=config["use_node_type"],
#         use_ego_robot=config["use_ego_robot"],
#         use_edge_rt=config["use_edge_rt"],
#         two_hop=config["two_hop"],
#         vicinity_m=config["vicinity_m"],
#         max_steps=config["max_steps"],
#     )

#     feature_dim = base_env.F

#     env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))

#     # ---------------------------------------------------------
#     # Model
#     # ---------------------------------------------------------
#     policy_kwargs = dict(
#         in_dim=feature_dim,
#         hidden=64,
#         k_max=config["K_max"],
#         logit_temperature=5.0,
#         noop_init=-1.0,
#         freeze_noop_logit=False,
#         edge_dim=0,
#         backbone="sage",
#         gnn_kwargs={"layers": 2},
#     )

#     model = PPO(
#         RTGNNPolicy,
#         env,
#         policy_kwargs=policy_kwargs,
#         n_steps=config["ppo_steps"],
#         batch_size=config["batch_size"],
#         learning_rate=config["learning_rate"],
#         gamma=0.99,
#         clip_range=0.2,
#         vf_coef=0.5,
#         ent_coef=0.01,
#         gae_lambda=0.95,
#         n_epochs=10,
#         verbose=1,
#         device="cuda" if th.cuda.is_available() else "cpu",

#         # ✅ TensorBoard
#         tensorboard_log=str(tb_dir),
#     )

#     print("\n✓ Model created with TensorBoard logging")

#     # ---------------------------------------------------------
#     # Train
#     # ---------------------------------------------------------
#     callback = TrainingLogCallback(log_freq=100)

#     model.learn(
#         total_timesteps=config["total_timesteps"],
#         callback=callback,
#     )

#     # ---------------------------------------------------------
#     # Save
#     # ---------------------------------------------------------
#     model.save(str(model_dir / "final_model"))

#     print("\n" + "=" * 80)
#     print("TRAINING COMPLETE")
#     print("=" * 80)
#     print(f"Run directory: {run_dir}")
#     print(f"Model saved: {model_dir}")
#     print(f"TensorBoard logs: {tb_dir}")


# if __name__ == "__main__":
#     main()
"""
Training script for GNN-PPO with MultiAgentTaskEnv.

Features:
- TensorBoard logging
- Run tracking with metadata
- Configurable training
- Checkpoint management
- Complete logging
"""

import os
import json
import yaml
import platform
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch as th

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from src.environment.environment import MultiAgentTaskEnv
from src.models.sb3_gnn_policy import RTGNNPolicy


# ============================================================================
# Utils
# ============================================================================

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_config(config_path: str) -> Optional[Dict]:
    """Load YAML config file."""
    if not Path(config_path).exists():
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_generated_data(data_dir: str, n_batches: Optional[int] = None):
    print("*************loading data*************")
    """Load agents and tasks from generated .npy files."""
    data_path = Path(data_dir)
    print(data_path, 'exists:', data_path.exists())
    # Load agents
    agents_file = data_path / "agents.npy"
    if not agents_file.exists():
        raise FileNotFoundError(f"Agents file not found: {agents_file}")
    
    agents = np.load(agents_file, allow_pickle=True)
    print(f"Loaded agents: {agents.shape}")

    # Load task batches
    tasks_batches = []
    idx = 0

    while True:
        if n_batches is not None and idx >= n_batches:
            break

        file = data_path / f"tasks_batch_{idx}.npy"
        if not file.exists():
            break

        batch = np.load(file, allow_pickle=True)
        tasks_batches.append(batch)
        print(f"✓ Loaded batch {idx}: {batch.shape}")
        idx += 1

    if not tasks_batches:
        raise FileNotFoundError(f"No task batches found in {data_path}")

    print(f"✓ Total: {len(agents)} robots, {len(tasks_batches)} batches")
    return agents, tasks_batches


# ============================================================================
# Callbacks
# ============================================================================

class TrainingLogCallback(BaseCallback):
    """Log training metrics."""
    
    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        
        if self.step_count % self.log_freq != 0:
            return True
        
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                print(
                    f"  Timesteps: {self.num_timesteps:7d} | "
                    f"Episode Reward: {ep['r']:8.2f} | "
                    f"Episode Length: {ep['l']:6d}"
                )
        
        return True


class CheckpointCallback(BaseCallback):
    """Save checkpoints at intervals."""
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./checkpoints",
        name_prefix: str = "model"
    ):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_timestep_{self.num_timesteps}.zip"
            self.model.save(str(path))
            print(f"  ✓ Checkpoint saved: {path.name}")
        
        return True


# ============================================================================
# Main Training
# ============================================================================

def main():
    print("=" * 80)
    print("GNN-PPO Training with TensorBoard & Run Tracking")
    print("=" * 80 + "\n")

    # =========================================================================
    # Configuration
    # =========================================================================
    print("Loading configuration...")
    config_path = "configs/training_config.yaml"
    config = load_config(config_path)

    if config is None:
        print(" Config not found, using defaults")
        config = {
            "data_dir": "data",
            "n_batches": None,
            "K_max": 5,
            "N_max": 15,
            "E_max": 50,
            "use_xy_pickup": False,
            "normalize_features": True,
            "use_node_type": True,
            "use_ego_robot": True,
            "use_edge_rt": False,
            "two_hop": False,
            "vicinity_m": 20.0,
            "max_steps": 1000,
            "max_robot_capacity": 2,
            "ppo_steps": 2048,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "total_timesteps": 100000,
            "checkpoint_freq": 10000,
            "seed": 42,
        }
    else:
        print(f"  ✓ Loaded from {config_path}")

    seed = int(config.get("seed", 42))
    run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(" Configuration:")
    for k, v in config.items():
        if k != "seed":
            print(f"  {k}: {v}")
    print(f"  seed: {seed}")
    print(f"  run_id: {run_id}")

    # =========================================================================
    # Run Directories
    # =========================================================================
    print(" Setting up directories...")
    run_dir = Path("runs") / f"seed_{seed}" / f"run_{run_id}"
    model_dir = run_dir / "models"
    tb_dir = run_dir / "tensorboard"
    log_dir = run_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"  ✓ Run directory: {run_dir}")

    # =========================================================================
    # Save Metadata
    # =========================================================================
    print("\n💾 Saving run metadata...")
    run_meta = {
        "run_id": run_id,
        "seed": seed,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "cuda_available": th.cuda.is_available(),
            "cuda_version": th.version.cuda if th.cuda.is_available() else None,
        },
        "config": config,
    }

    meta_file = run_dir / "run_metadata.json"
    meta_file.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"  ✓ Metadata saved to {meta_file.name}")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("Loading generated data...")
    try:
        agents, tasks_batches = load_generated_data(
            "data",
            n_batches=config.get("n_batches")
        )
    except Exception as e:
        print(f"  Error loading data: {e}")
        return

    # =========================================================================
    # Create Environment
    # =========================================================================
    print("\n🔧 Creating environment...")
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
        )

        feature_dim = base_env.F
        print(f"  ✓ Environment created")
        print(f"    Feature dim: {feature_dim}")
        print(f"    Action space: {base_env.action_space}")
        print(f"    Observation space keys: {list(base_env.observation_space.spaces.keys())}")

        # Wrap with Monitor
        env = Monitor(base_env, filename=str(log_dir / "monitor.csv"))
        print(f"  ✓ Monitor enabled: {log_dir / 'monitor.csv'}")

    except Exception as e:
        print(f"  ❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Create Model
    # =========================================================================
    print("\n🧠 Creating PPO model...")
    try:
        policy_kwargs = dict(
            in_dim=feature_dim,
            hidden=64,
            k_max=config["K_max"],
            logit_temperature=5.0,
            noop_init=-1.0,
            freeze_noop_logit=False,
            edge_dim=0,
            use_competitor_fusion=False,
            use_two_hop_actor=False,
            use_two_hop_critic=False,
            backbone="sage",
            critic_aggregation="joint_mean",
            gnn_kwargs={"layers": 2},
        )

        print("  Policy kwargs:")
        for k, v in policy_kwargs.items():
            print(f"    {k}: {v}")

        device = "cuda" if th.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

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

        print(f"  ✓ Model created with TensorBoard logging")

    except Exception as e:
        print(f"  ❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Training
    # =========================================================================
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    try:
        callbacks = [
            TrainingLogCallback(log_freq=100),
            CheckpointCallback(
                save_freq=config.get("checkpoint_freq", 10000),
                save_path=str(model_dir),
                name_prefix="model"
            ),
        ]

        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
        )

        print("\n✓ Training complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Save Final Model
    # =========================================================================
    print("\n💾 Saving final model...")
    try:
        final_path = run_dir / "ppo_final.zip"
        model.save(str(final_path))
        print(f"  ✓ Model saved: {final_path}")
    except Exception as e:
        print(f"  ❌ Error saving model: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\n📂 Run directory: {run_dir}")
    print(f"📊 Model: {final_path}")
    print(f"📈 TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"📋 Logs: {log_dir}")
    print(f"✅ Training finished at {datetime.datetime.now().isoformat()}\n")

    env.close()


if __name__ == "__main__":
    main()