#!/usr/bin/env python3
"""
Test that the environment actually works with random actions.
"""

import numpy as np
from pathlib import Path
from src.environment.environment import MultiTaskAllocationEnv


def test_random_policy():
    """Test environment with random policy."""
    
    # Load data
    agents = np.load('data/agents.npy', allow_pickle=True)
    
    batches = []
    for i in range(3):
        batch_file = Path('data') / f"tasks_batch_{i}.npy"
        if batch_file.exists():
            batch = np.load(batch_file, allow_pickle=True)
            batches.append(batch)
            print(f"Loaded batch {i}: {len(batch)} tasks")
    
    # Create environment
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        radius=2000,
        feature_size=9,
        use_true_id=False,
        all_batches=True
    )
    
    print(f"\nEnvironment created:")
    print(f"  Total tasks: {len(env.tasks)}")
    print(f"  Robots: {env.n_robots}")
    print(f"  Batch time: {env.batch_time}")
    
    # Run episode with random assignments
    obs, _ = env.reset()
    done = False
    step = 0
    assignment_interval = 5
    
    total_assignments = 0
    
    while not done and step < 300:
        # Every 5 steps, make random assignments
        if step % assignment_interval == 0:
            available_tasks = env.get_available_task_ids()
            
            if len(available_tasks) > 0:
                # Random assignment
                assignments = {}
                
                for rid, robot in enumerate(env.robots):
                    if robot.capacity < robot.maxCapacity and len(available_tasks) > 0:
                        # Pick random task
                        task_id = np.random.choice(available_tasks)
                        assignments[rid] = [task_id]
                        
                        # Remove from available
                        available_tasks = [t for t in available_tasks if t != task_id]
                
                if assignments:
                    total_assignments += len(assignments)
                    print(f"Step {step}: Made {len(assignments)} assignments. "
                          f"Available: {len(env.get_available_task_ids())}")
            else:
                assignments = None
        else:
            assignments = None
        
        obs, reward, done, truncated, info_reward, info = env.step(assignments, assignment_interval=assignment_interval)
        
        if isinstance(reward, dict):
            reward_sum = sum(reward.values())
        else:
            reward_sum = reward
        
        step += 1
    
    # Check results
    completed = sum(1 for t in env.tasks if t.is_droppedoff)
    obsolete = sum(1 for t in env.tasks if t.is_obsolete(env.time_count))
    
    print(f"\n{'='*60}")
    print(f"RANDOM POLICY TEST RESULTS")
    print(f"{'='*60}")
    print(f"Steps: {step}")
    print(f"Assignments made: {total_assignments}")
    print(f"Completed: {completed}/{len(env.tasks)}")
    print(f"Obsolete: {obsolete}")
    print(f"{'='*60}")
    
    if completed > 0:
        print("\n✓ Environment is working! Tasks can be completed.")
        return True
    else:
        print("\n✗ Environment problem: No tasks completed even with random policy!")
        return False


if __name__ == '__main__':
    test_random_policy()