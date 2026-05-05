import pytest
import numpy as np
import torch
from pathlib import Path
from src.environment.environment import MultiTaskAllocationEnv
from models.actor_critic_old import ActorGNN, CriticGNN
from src.training.train_actor_critic import train
from main_all_batches import load_all_batches, set_seed, save_models, plot_rewards, plot_task_stats, plot_values

@pytest.fixture
def setup_environment(tmp_path):
    """Fixture to set up agents, batches, and environment."""
    # Create mock agents
    agents = np.random.rand(5, 2)  # 5 agents with 2D coordinates
    agents_file = tmp_path / "agents.npy"
    np.save(agents_file, agents)

    # Create mock task batches
    n_batches = 3
    batches = []
    for i in range(n_batches):
        batch = np.random.rand(10, 9)  # 10 tasks with 9 features
        batch[:, 7] = i * 50  # Set release times
        batch_file = tmp_path / f"tasks_batch_{i}.npy"
        np.save(batch_file, batch)
        batches.append(batch)

    return agents_file, tmp_path, batches

def test_load_all_batches(setup_environment):
    """Test that task batches are loaded correctly."""
    _, data_dir, batches = setup_environment
    loaded_batches = load_all_batches(data_dir, len(batches))
    assert len(loaded_batches) == len(batches), "Mismatch in number of loaded batches"
    for i, batch in enumerate(batches):
        assert np.array_equal(loaded_batches[i], batch), f"Batch {i} does not match"

def test_environment_creation(setup_environment):
    """Test that the environment is created correctly."""
    agents_file, data_dir, batches = setup_environment
    agents = np.load(agents_file, allow_pickle=True)
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=30,
        feature_size=9,
        use_true_id=False,
        all_batches=True
    )
    assert env.n_tasks == sum(len(batch) for batch in batches), "Mismatch in total tasks"
    assert env.n_robots == len(agents), "Mismatch in number of robots"

def test_training_process(setup_environment):
    """Test that the training process runs without errors."""
    agents_file, data_dir, batches = setup_environment
    agents = np.load(agents_file, allow_pickle=True)
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=30,
        feature_size=9,
        use_true_id=False,
        all_batches=True
    )

    # Create actors and critic
    input_dim = 9
    hidden_dim = 64
    num_robots = env.n_robots
    actors = {rid: ActorGNN(input_dim, hidden_dim) for rid in range(num_robots)}
    optimizers_actors = {rid: torch.optim.Adam(actor.parameters(), lr=1e-4) for rid, actor in actors.items()}
    critic = CriticGNN(input_dim, hidden_dim, "per_robot")
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=5e-4)

    # Train
    episode_rewards, episode_task_stats, episode_value_means = train(
        env,
        num_episodes=5,
        actors=actors,
        critic=critic,
        optimizers_actors=optimizers_actors,
        optimizer_critic=optimizer_critic,
        gamma=0.99,
        max_steps_per_episode=200,
        device="cpu",
        verbose=False,
        save_dir=None,
        save_every=0,
        plot_rewards_fn=None,
        plot_task_stats=None,
        plot_values_fn=None,
        save_models_fn=None
    )

    assert len(episode_rewards) == 5, "Mismatch in number of episodes"
    assert len(episode_task_stats) == 5, "Mismatch in task stats"
    assert len(episode_value_means) == 5, "Mismatch in value means"

def test_loss_plotting(tmp_path):
    """Test that loss plots are generated."""
    save_dir = tmp_path / "plots"
    save_dir.mkdir()

    # Mock data
    episode_rewards = [10, 20, 30, 40, 50]
    episode_task_stats = [{"episode": i, "obsolete": i, "never_picked": i, "completed": i} for i in range(5)]
    episode_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Generate plots
    plot_rewards(save_dir, episode_rewards)
    plot_task_stats(save_dir, episode_task_stats)
    plot_values(save_dir, episode_values)

    # Check that plots are saved
    assert (save_dir / "rewards.png").exists(), "Rewards plot not generated"
    assert (save_dir / "task_stats.png").exists(), "Task stats plot not generated"
    assert (save_dir / "values.png").exists(), "Values plot not generated"