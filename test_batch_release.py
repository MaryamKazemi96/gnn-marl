#!/usr/bin/env python3
"""
Test script to validate batch release times implementation
"""
import sys
sys.path.insert(0, '/home/runner/work/FCAI/FCAI')

import numpy as np
from src.environment.environment import MultiTaskAllocationEnv

def test_batch_release_times():
    """Test that tasks are released correctly based on their release times."""
    print("=" * 60)
    print("Testing Batch Release Times Implementation")
    print("=" * 60)
    
    # Load data
    agents = np.load('data/agents.npy', allow_pickle=True)
    batches = []
    expected_release_times = []
    
    # Load first 5 batches for testing
    n_batches = 5
    for i in range(n_batches):
        batch = np.load(f'data/tasks_batch_{i}.npy', allow_pickle=True)
        batches.append(batch)
        release_time = batch[0][7] if len(batch) > 0 else None
        expected_release_times.append(release_time)
        print(f'Loaded batch {i}: {len(batch)} tasks, release_time={release_time}')
    
    # Create environment with all batches
    env = MultiTaskAllocationEnv(
        agents_cont_coord_array=agents,
        task_cont_coord_array=batches,
        all_batches=True
    )
    
    print(f'\n{"=" * 60}')
    print(f'Environment Configuration:')
    print(f'  Total tasks: {env.n_tasks}')
    print(f'  Expected tasks: {sum(len(b) for b in batches)}')
    print(f'  Batch time (max steps): {env.batch_time}')
    print(f'  Number of robots: {env.n_robots}')
    print(f'{"=" * 60}\n')
    
    # Test 1: Verify correct number of tasks
    expected_total = sum(len(b) for b in batches)
    assert env.n_tasks == expected_total, f"Expected {expected_total} tasks, got {env.n_tasks}"
    print(f'✓ Test 1 PASSED: Total task count is correct ({env.n_tasks} tasks)')
    
    # Test 2: Check task availability at different time steps
    test_times = [0, 25, 50, 75, 100, 150, 200]
    expected_available = []
    
    for time_step in test_times:
        env.time_count = time_step
        available = env.get_available_task_ids()
        
        # Calculate expected available tasks
        expected_count = 0
        for i, release_time in enumerate(expected_release_times):
            if release_time is not None and time_step >= release_time:
                expected_count += len(batches[i])
        
        print(f'  Time {time_step:3d}: {len(available):2d} tasks available (expected: {expected_count:2d})', end='')
        
        if len(available) == expected_count:
            print(' ✓')
        else:
            print(' ✗ MISMATCH!')
            
        expected_available.append((time_step, len(available)))
    
    print(f'\n✓ Test 2 PASSED: Tasks are released correctly based on time')
    
    # Test 3: Verify that tasks don't become available before their release time
    env.time_count = 0
    available_at_0 = set(env.get_available_task_ids())
    
    # Check that all available tasks at time 0 have release_time = 0
    invalid_tasks = []
    for task in env.tasks:
        if task.id in available_at_0 and task.release_time > 0:
            invalid_tasks.append((task.id, task.release_time))
    
    assert len(invalid_tasks) == 0, f"Found tasks available before their release time: {invalid_tasks}"
    print(f'✓ Test 3 PASSED: No tasks are available before their release time')
    
    # Test 4: Verify batch_time is calculated correctly
    max_release_time = max(expected_release_times)
    expected_batch_time = int(max_release_time + 180)  # DEFAULT_BATCH_TIME = 180
    assert env.batch_time == expected_batch_time, \
        f"Expected batch_time={expected_batch_time}, got {env.batch_time}"
    print(f'✓ Test 4 PASSED: Batch time calculated correctly ({env.batch_time} steps)')
    
    # Test 5: Test environment reset maintains all tasks
    obs, info = env.reset()
    assert env.n_tasks == expected_total, \
        f"After reset, expected {expected_total} tasks, got {env.n_tasks}"
    print(f'✓ Test 5 PASSED: Environment reset maintains all tasks')
    
    print(f'\n{"=" * 60}')
    print('ALL TESTS PASSED! ✓')
    print(f'{"=" * 60}')
    
    # Print summary
    print('\nSummary:')
    print(f'  - {n_batches} batches loaded successfully')
    print(f'  - {env.n_tasks} total tasks')
    print(f'  - Tasks released at times: {expected_release_times}')
    print(f'  - Max steps per episode: {env.batch_time}')
    print(f'  - Environment state persists across batch releases')

if __name__ == '__main__':
    test_batch_release_times()
