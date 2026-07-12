import random
import math
import sys
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import astar, discritized_path  # Ensure astar is implemented in utils.py
from environment.environment import Planner

class DataGenerator:
    def __init__(self, x_min, x_max, y_min, y_max, max_waiting_time,
                  max_travel_delay_percentage, planning_resolution,
                    planner, origin_x, origin_y ):
        self.planning_resolution = planning_resolution
        self.h_min = 0
        # ensure integer max indices
        self.h_max = int(abs(y_min - y_max) / planning_resolution)
        self.w_min = 0
        self.w_max = int(abs(x_min - x_max) / planning_resolution)
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.max_waiting_time = max_waiting_time
        self.max_travel_delay_percentage = max_travel_delay_percentage
        self.planner = planner  
        

    def generate_agents(self, num_agents):
        agents = []
        for i in range(num_agents):
            while True:
                h = random.randint(self.h_min, self.h_max)
                w = random.randint(self.w_min, self.w_max)
                if self.planner.is_point_valid((h, w)) == True:
                    unique_id =  int(f"2{i:02d}") # Hex-like ID (e.g. 200, 201...)
                    # yaw = random.random() * 2 * math.pi - math.pi
                    agents.append([unique_id, w, h])
                    break
        return np.array(agents)

    def generate_tasks(self, n_batches, n_points, release_time_interval=15):
        """
        Generate tasks for multiple batches with release times.
        
        Args:
            n_batches: Number of batches to generate
            n_points: Number of tasks per batch
            release_time_interval: Time interval between batch releases (default: 30)
            
        Returns:
            List of task batches, where each batch has tasks with batch-specific release times
        """
        all_batches = []  # Store all batches of tasks
        for i in range(n_batches):
            tasks = []
            task_num = 0  # Initialize task number for the batch
            batch_release_time = i * release_time_interval  # Calculate release time for this batch
            
            while len(tasks) < n_points:
                # origin
                while True:
                    h_origin = random.randint(self.h_min, self.h_max)
                    w_origin = random.randint(self.w_min, self.w_max)
                    if self.planner.is_point_valid((h_origin, w_origin)):
                        break 
                # destination
                while True:
                    h_destination = random.randint(self.h_min, self.h_max)
                    w_destination = random.randint(self.w_min, self.w_max)
                    if self.planner.is_point_valid((h_destination, w_destination)):
                        break
                # yaw_origin = random.random() * 2 * math.pi - math.pi
                # yaw_destination = random.random() * 2 * math.pi - math.pi
                t_release = batch_release_time  # Use batch-specific release time

                # Calculate estimated travel time using Planner (A*)
                start = (h_origin, w_origin)
                goal = (h_destination, w_destination)
                found, path = self.planner.get_plan(start, goal)
                if not found:
                    continue  # Skip if no path found

                # Use path length as estimated travel time
                estimated_travel_time = len(path) if found else float('inf')

                # Calculate deadlines
                pickup_deadline = (t_release * i) + self.max_waiting_time
                drop_off_deadline = pickup_deadline + (estimated_travel_time * (1 + self.max_travel_delay_percentage))

                # Generate a unique ID (task)
                unique_id = int(f"1{i:02d}{task_num:02d}")

                tasks.append([
                    unique_id,
                    w_origin, h_origin,
                    w_destination, h_destination,
                    t_release,
                    pickup_deadline,
                    estimated_travel_time,   # estimated travel time BEFORE dropoff_deadline
                    drop_off_deadline
                ])
                task_num += 1

            all_batches.append(tasks)

        return all_batches
    
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate batch data with release times for multi-robot task allocation"
    )
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of batches to generate (default: 10)")
    parser.add_argument("--n-tasks", type=int, default=8,
                        help="Number of tasks per batch (default: 4)")
    parser.add_argument("--n-robots", type=int, default=6,
                        help="Number of robots/agents (default: 6)")
    parser.add_argument("--release-interval", type=int, default=15,
                        help="Time interval between batch releases (default: 30)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for data files (default: data/)")
    args = parser.parse_args()
    
    # Load configuration from ATC_wed.yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "env" / "ATC_wed.yaml"
    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)

    # Extract grid bounds and other parameters from the YAML file
    x_min, x_max = params['x_min'], params['x_max']
    y_min, y_max = params['y_min'], params['y_max']
    map_resolution = params['map_resolution']
    Planning_resolution = params['Planning_resolution']
    max_waiting_time = 200
    max_travel_delay_percentage = 2

    # Initialize planner (expects Planner to load map from env/ATC_wed.yaml)
    planner = Planner()

    # Initialize the data generator
    generator = DataGenerator(x_min, x_max, y_min, y_max, max_waiting_time, 
                              max_travel_delay_percentage, Planning_resolution, 
                              planner, origin_x=-60, origin_y=20)

    # Generate tasks with batch release times
    # Batch 0: release_time = 0, Batch 1: release_time = 30, Batch 2: release_time = 60, etc.
    print(f"Generating data:")
    print(f"  - {args.n_batches} batches")
    print(f"  - {args.n_tasks} tasks per batch")
    print(f"  - {args.n_robots} robots")
    print(f"  - Release time interval: {args.release_interval}")
    
    agents = generator.generate_agents(args.n_robots)
    tasks = generator.generate_tasks(args.n_batches, args.n_tasks, args.release_interval)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Save agents
    agents_file = output_dir / "agents.npy"
    np.save(agents_file, agents)
    print(f"Agents saved to {agents_file}")

    # Save tasks batches
    for i, batch in enumerate(tasks):
        tasks_file = output_dir / f"tasks_batch_{i}.npy"
        np.save(tasks_file, batch)
        print(f"Tasks for batch {i} (release_time={i * args.release_interval}) saved to {tasks_file}")