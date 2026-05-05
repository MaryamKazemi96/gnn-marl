import gym
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.environment.environment import MultiTaskAllocationEnv

agents_file = Path(__file__).resolve().parent.parent / "data" / "agents.npy"
tasks_file = Path(__file__).resolve().parent.parent / "data" / "tasks_batch_0.npy"

    # Load agents and tasks
agents = np.load(agents_file, allow_pickle=True)
tasks = np.load(tasks_file, allow_pickle=True)
env = MultiTaskAllocationEnv(agents, tasks, radius=5, use_true_id=False )
obs = env.reset()
ego_edge_indices, attribute_matrix = obs
print(obs)
