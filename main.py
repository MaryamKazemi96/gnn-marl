import torch
import numpy as np
import gym
from pathlib import Path
import matplotlib.pyplot as plt

from src.environment.environment import MultiTaskAllocationEnv
from models.actor_critic_old import ActorGNN, CriticGNN
from src.training.train_actor_critic import train
from gym.envs.registration import register
import time

# # Register environment
# register(
#     id="MultiTaskAlloc-v1",
#     entry_point="src.environment.environment:MultiTaskAllocationEnv",
#     kwargs={"radius": 20, "feature_size": 9, "use_true_id": False},
# )

# # Load agents and tasks
# agents_file = Path(__file__).resolve().parent / "data" / "agents.npy"
# tasks_file = Path(__file__).resolve().parent / "data" / "tasks_batch_0.npy"
# agents = np.load(agents_file, allow_pickle=True)
# tasks = np.load(tasks_file, allow_pickle=True)


# # ... your existing imports and env initialization above ...

# # Initialize the environment
# env = gym.make("MultiTaskAlloc-v1",
#                agents_cont_coord_array=agents,
#                task_cont_coord_array=tasks,
#                use_true_id=False)


import argparse
import json
from pathlib import Path
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.environment.environment import MultiTaskAllocationEnv
from models.actor_critic_old import ActorGNN, CriticGNN
from src.training.train_actor_critic import train

# --- Utilities ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_models(save_dir: Path, actors: dict, critic: torch.nn.Module):
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save each actor separately
    for rid, actor in actors.items():
        torch.save(actor.state_dict(), save_dir / f"actor_{rid}.pt")
    torch.save(critic.state_dict(), save_dir / "critic.pt")

def plot_rewards(save_dir: Path, episode_rewards):
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (sum over robots)")
        plt.title("Training Rewards")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / "rewards.png")
        plt.close()
    except Exception as e:
        print("Warning: failed to plot rewards:", e)

# --- Main ---
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Device:", device)

    # Load agents and tasks
    agents_file = Path(args.agents)
    tasks_file = Path(args.tasks)
    agents = np.load(agents_file, allow_pickle=True)
    tasks = np.load(tasks_file, allow_pickle=True)

    # Create environment
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=tasks,
        radius=args.radius,
        feature_size=args.feature_size,
        use_true_id=args.use_true_id,
    )

    # Model hyperparams
    input_dim = args.feature_size
    hidden_dim = args.hidden_dim
    critic_aggregation = args.critic_agg  # "per_robot" or "joint_mean" etc.

    # Create actors (one per robot) and optimizers
    num_robots = env.n_robots
    actors = {rid: ActorGNN(input_dim, hidden_dim).to(device) for rid in range(num_robots)}
    optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
                         for rid, actor in actors.items()}

    # Shared critic
    critic = CriticGNN(input_dim, hidden_dim, critic_aggregation).to(device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # Train
    t0 = time.time()
    episode_rewards = train(
        env,
        num_episodes=args.episodes,
        actors=actors,
        critic=critic,
        optimizers_actors=optimizers_actors,
        optimizer_critic=optimizer_critic,
        gamma=args.gamma,
        max_steps_per_episode=args.max_steps,
        device=device
    )
    t1 = time.time()
    print(f"Training finished in {t1 - t0:.1f}s")

    # Save results and models
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Save models
    save_models(save_dir, actors, critic)
    # Save rewards
    with open(save_dir / "episode_rewards.json", "w") as f:
        json.dump([float(x) for x in episode_rewards], f)
    # Plot rewards
    plot_rewards(save_dir, episode_rewards)

    print(f"Saved models and rewards to {save_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=str, default="data/agents.npy")
    parser.add_argument("--tasks", type=str, default="data/tasks_batch_0.npy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--feature-size", type=int, default=9)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=1e-3)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--radius", type=int, default=20)
    parser.add_argument("--critic-agg", type=str, default="per_robot", choices=["per_robot", "joint_mean", "joint_attn"])
    parser.add_argument("--use-true-id", action="store_true", dest="use_true_id")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    main(args)