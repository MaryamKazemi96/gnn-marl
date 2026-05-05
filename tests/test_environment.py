from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.environment.environment import MultiTaskAllocationEnv

def test_environment():
    # Paths to the agents and tasks files
    agents_file = Path(__file__).resolve().parent.parent / "data" / "agents.npy"
    tasks_file = Path(__file__).resolve().parent.parent / "data" / "tasks.npy"

    # Load agents and tasks
    agents = np.load(agents_file, allow_pickle=True)
    tasks = np.load(tasks_file, allow_pickle=True)

    # Initialize the environment
    env = MultiTaskAllocationEnv(agents, tasks)

    # Reset the environment
    observation, _ = env.reset()
    print("Initial Observation:")
    print(observation)

    # Take a few steps in the environment
    print("\nTaking steps in the environment...")
    for step in range(5):
        # Generate random task-to-robot assignments for testing
        list_t2r_assignments = [(task[0], agents[np.random.randint(len(agents))][0]) for task in tasks]

        # Step the environment
        observation, reward, done, truncated, info = env.step(list_t2r_assignments)
        print(f"Step {step + 1}:")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        if done or truncated:
            print("Environment terminated.")
            break

    # Close the environment
    env.close()

if __name__ == "__main__":
    test_environment()