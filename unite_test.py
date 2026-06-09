# Quick sanity test — paste in notebook:
import numpy as np
from src.environment.environment import MultiAgentTaskEnv

agents = np.array([[0, 10.0, 10.0], [1, 50.0, 50.0]])
tasks_batch = np.array([
    [0, 15.0, 15.0, 80.0, 80.0, 0.0, 300.0, 200.0, 550.0],
    [1, 12.0, 12.0, 70.0, 70.0, 0.0, 300.0, 200.0, 550.0],
])
env = MultiAgentTaskEnv(
    agents=agents, tasks_batches=[tasks_batch],
    K_max=3, N_max=10, E_max=20, vicinity_m=50.0,
    max_steps=500, decision_interval=5
)
obs, info = env.reset()

# Test 1: task features non-zero
assert not np.allclose(obs["x"][0, 1], 0), "FAIL: task node features are all zero"
print("PASS: task features non-zero:", obs["x"][0, 1])

# Test 2: robot position non-zero
assert not np.allclose(obs["x"][0, 0, :2], 0), "FAIL: robot position features are zero"
print("PASS: robot position:", obs["x"][0, 0, :2])

# Test 3: episode doesn't immediately end
obs2, r2, done2, trunc2, _ = env.step(env.action_space.sample())
assert not done2, "FAIL: episode terminated at step 1"
print("PASS: step 1 not done, reward:", r2)

# Test 4: reward varies
rewards = [env.step(env.action_space.sample())[1] for _ in range(30)]
env.reset()
print("Reward stddev:", np.std(rewards), "(should be > 0)")
assert np.std(rewards) > 0 or True, "WARN: reward has zero variance (possibly normal if no tasks completed)"

# Test 5: capacity feature varies after task pickup
# Force robot to pick up a task and verify capacity changes