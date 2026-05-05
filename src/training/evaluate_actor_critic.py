import torch
from src.environment.environment import MultiTaskAllocationEnv
from models.actor_critic_old import ActorCritic
import numpy as np

def evaluate(env, model, num_episodes=10):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            ego_graphs, node_features = obs
            edge_index = ego_graphs.edge_index
            x = node_features

            # Forward pass
            policy, _ = model(x, edge_index)

            # Select action
            action = torch.argmax(policy, dim=-1)

            # Step the environment
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # Load agents and tasks
    agents = np.load("data/agents.npy", allow_pickle=True)
    tasks = np.load("data/tasks.npy", allow_pickle=True)

    # Initialize the environment
    env = MultiTaskAllocationEnv(agents, tasks)

    # Load the trained model
    node_feature_dim = 9  # Adjust based on your environment
    hidden_dim = 128
    num_actions = env.action_space.n
    model = ActorCritic(node_feature_dim, hidden_dim, num_actions)
    model.load_state_dict(torch.load("actor_critic_model.pth"))

    # Evaluate the model
    evaluate(env, model)