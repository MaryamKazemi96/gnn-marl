
# """
# Updated environment.py with MultiAgentTaskEnv that supports both
# old format (for backward compatibility) and new format (with generated data).
# """
# import gymnasium as gym
# import numpy as np
# from typing import Dict, Any, Tuple, Optional, List
# from pathlib import Path
# import sys
# import yaml
# from PIL import Image

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from src.utils.ego_graph_builder import build_padded_ego_batch
# from src.utils.feature_fn import make_feature_fn, compute_feature_dim
# from utils import utils as ut


# class Planner:
#     """Path planning with A* algorithm."""
#     def __init__(self):
#         root_path = Path(__file__).resolve().parent.parent.parent / "env"
#         config_path = root_path / "ATC_wed.yaml"
#         with open(config_path, 'r') as file:
#             params = yaml.safe_load(file)
#         map_path = root_path / params['map_filename']
#         self.map_img = Image.open(map_path).convert('L')
#         self.map_resolution = params['map_resolution']
#         self.Planning_resolution = params['Planning_resolution']
#         self.threshold = params['obstacle_threshold']
#         self.origin_x = params['origin_x']
#         self.origin_y = params['origin_y']
#         self.average_velocity = params['average_velocity']

#     def get_obstacle_grid(self):
#         img_w, img_h = self.map_img.size
#         scale = self.map_resolution / self.Planning_resolution
#         grid_height, grid_width = int(img_h * scale), int(img_w * scale)
#         grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
#         for row in range(grid_height):
#             for col in range(grid_width):
#                 px = int((col + 0.5) * img_w / grid_width)
#                 py = int((row + 0.5) * img_h / grid_height)
#                 grid[row, col] = 1 if self.map_img.getpixel((px, py)) < (self.threshold * 255) else 0
#         return grid

#     def is_point_valid(self, point):
#         grid = self.get_obstacle_grid()
#         w = point[1]
#         h = point[0]
#         if 0 <= h < grid.shape[0] and 0 <= w < grid.shape[1]:
#             return grid[h, w] == 0
#         return False

#     def get_plan(self, start, end):
#         grid = self.get_obstacle_grid()
#         found, path = ut.astar(grid, start, end)
#         return found, path


# class MultiAgentTaskEnv(gym.Env):
#     """
#     Multi-agent task allocation environment with GNN support.
    
#     Supports two initialization modes:
#     1. New mode: Pass agents and tasks_batches arrays (from generated data)
#     2. Old mode: Pass agents_cont_coord_array and task_cont_coord_array (legacy)
#     """

#     def __init__(
#         self,
#         # New mode parameters
#         agents: np.ndarray = None,
#         tasks_batches: list = None,
#         # Old mode parameters (legacy)
#         agents_cont_coord_array: np.ndarray = None,
#         task_cont_coord_array: np.ndarray = None,
#         # Feature extraction params
#         use_xy_pickup: bool = False,
#         normalize_features: bool = True,
#         use_node_type: bool = True,
#         use_ego_robot: bool = True,
#         use_edge_rt: bool = False,
#         edge_features=None,
#         # Graph params
#         N_max: int = 15,
#         E_max: int = 50,
#         K_max: int = 5,
#         # Scaling params
#         max_robot_capacity: int = 2,
#         max_wait_delay_s: float = 600.0,
#         max_travel_delay_s: float = 3600.0,
#         max_steps: int = 1000,
#         # 2-hop params
#         two_hop: bool = False,
#         two_hop_directed: bool = False,
#         vicinity_m: float = 50.0,
#         # Legacy parameters
#         radius: int = 20,
#         feature_size: int = 9,
#         use_true_id: bool = False,
#         reward_mode: str = "new",
#     ):
#         super().__init__()

#         # Determine initialization mode
#         if agents is not None and tasks_batches is not None:
#             # New mode: using generated data
#             self.init_mode = "new"
#             self.agents_data = agents
#             self.tasks_batches = tasks_batches
#             self.current_batch_idx = 0
#             self.num_robots = len(agents)
#         elif agents_cont_coord_array is not None and task_cont_coord_array is not None:
#             # Old mode: legacy format
#             self.init_mode = "old"
#             self.agents_cont_coord_array = agents_cont_coord_array
#             self.task_cont_coord_array = task_cont_coord_array
#             self.num_robots = len(agents_cont_coord_array)
#             self.radius = radius
#             self.feature_size = feature_size
#             self.use_true_id = use_true_id
#             self.reward_mode = reward_mode
#             self.planner = Planner()
#         else:
#             raise ValueError(
#                 "Must provide either (agents, tasks_batches) or "
#                 "(agents_cont_coord_array, task_cont_coord_array)"
#             )

#         # Common parameters
#         self.N_max = N_max
#         self.E_max = E_max
#         self.K_max = K_max
#         self.max_robot_capacity = max_robot_capacity
#         self.vicinity_m = vicinity_m
#         self.two_hop = two_hop
#         self.two_hop_directed = two_hop_directed
#         self.max_steps = max_steps

#         # Initialize environment state
#         self.robots = {}
#         self.tasks = {}
#         self.current_time = 0.0
#         self.current_step = 0

#         # Get map bounds
#         if self.init_mode == "new":
#             self.max_position = max(
#                 np.max(agents[:, 1]),  # max w
#                 np.max(agents[:, 2]),  # max h
#             )
#         else:
#             self.max_position = 100.0  # Default for old mode

#         # ====================================================================
#         # FEATURE EXTRACTION & GRAPH BUILDING
#         # ====================================================================

#         # Compute feature dimension
#         self.F = compute_feature_dim(
#             use_xy_pickup=use_xy_pickup,
#             use_node_type=use_node_type,
#             use_edge_rt=use_edge_rt,
#             use_ego_robot=use_ego_robot,
#         )

#         # Initialize feature function
#         self.feature_fn = make_feature_fn(
#             env_state=self,
#             use_xy_pickup=use_xy_pickup,
#             normalize_features=normalize_features,
#             use_node_type=use_node_type,
#             use_edge_rt=use_edge_rt,
#             edge_features=edge_features or [],
#             use_ego_robot=use_ego_robot,
#             max_position=self.max_position,
#             max_robot_capacity=max_robot_capacity,
#             max_wait_delay_s=max_wait_delay_s,
#             max_travel_delay_s=max_travel_delay_s,
#             max_steps=max_steps,
#         )

#         # Determine edge feature dimension
#         if use_edge_rt:
#             self.edge_features = edge_features or ["dx", "dy", "eta"]
#             self.edge_feat_dim = len(self.edge_features)
#         else:
#             self.edge_features = []
#             self.edge_feat_dim = 0

#         # Define observation space
#         self.observation_space = gym.spaces.Dict({
#             "x": gym.spaces.Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(self.num_robots, N_max, self.F),
#                 dtype=np.float32,
#             ),
#             "node_mask": gym.spaces.Box(
#                 low=0,
#                 high=1,
#                 shape=(self.num_robots, N_max),
#                 dtype=np.uint8,
#             ),
#             "edge_index": gym.spaces.Box(
#                 low=0,
#                 high=N_max,
#                 shape=(self.num_robots, 2, E_max),
#                 dtype=np.int64,
#             ),
#             "edge_mask": gym.spaces.Box(
#                 low=0,
#                 high=1,
#                 shape=(self.num_robots, E_max),
#                 dtype=np.uint8,
#             ),
#             "cand_idx": gym.spaces.Box(
#                 low=0,
#                 high=N_max,
#                 shape=(self.num_robots, K_max),
#                 dtype=np.int64,
#             ),
#             "cand_mask": gym.spaces.Box(
#                 low=0,
#                 high=1,
#                 shape=(self.num_robots, K_max),
#                 dtype=np.uint8,
#             ),
#         })

#         # Add edge_attr if needed
#         if self.edge_feat_dim > 0:
#             self.observation_space.spaces["edge_attr"] = gym.spaces.Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(self.num_robots, E_max, self.edge_feat_dim),
#                 dtype=np.float32,
#             )

#         # Action space
#         self.action_space = gym.spaces.MultiDiscrete([K_max + 1] * self.num_robots)
        
#         # NO-OP index
#         self._noop_index = K_max
        
#         # Tracking
#         self._last_cand_task_ids = [[] for _ in range(self.num_robots)]

#     def reset(self, seed=None):
#         """Reset environment and return initial observation."""
#         super().reset(seed=seed)

#         self.current_time = 0.0
#         self.current_step = 0

#         if self.init_mode == "new":
#             self._reset_new_mode()
#         else:
#             self._reset_old_mode()

#         obs = self._build_observation()
#         return obs, {"action_mask": self.action_mask()}

#     def _reset_new_mode(self):
#         """Reset for new mode (generated data)."""
#         # Initialize robots from agents data
#         self.robots = {}
#         for i, agent in enumerate(self.agents_data):
#             robot_id = int(agent[0])
#             self.robots[robot_id] = {
#                 "id": robot_id,
#                 "x": float(agent[1]),
#                 "y": float(agent[2]),
#                 "capacity": 0,
#                 "assigned_tasks": [],
#             }

#         # Initialize tasks from current batch
#         self.tasks = {}
#         batch = self.tasks_batches[self.current_batch_idx]
        
#         for task_data in batch:
#             task_id = int(task_data[0])
#             self.tasks[task_id] = {
#                 "id": task_id,
#                 "pickup_x": float(task_data[1]),
#                 "pickup_y": float(task_data[2]),
#                 "dropoff_x": float(task_data[3]),
#                 "dropoff_y": float(task_data[4]),
#                 "release_time": float(task_data[5]),
#                 "pickup_deadline": float(task_data[6]),
#                 "est_travel_time": float(task_data[7]),
#                 "dropoff_deadline": float(task_data[8]),
#                 "is_assigned": False,
#                 "is_obsolete": False,
#                 "is_completed": False,
#             }

#     def _reset_old_mode(self):
#         """Reset for old mode (legacy format)."""
#         # For backward compatibility with old mode
#         # This would use the old environment logic
#         pass

#     def step(self, actions: np.ndarray):
#         """Execute one environment step."""
        
#         # 1. Process actions
#         action_info = self._process_actions(actions)
        
#         # 2. Update robot positions
#         self._update_robot_positions()
        
#         # 3. Update task assignments
#         self._update_task_assignments(action_info)
        
#         # 4. Advance time
#         self.current_time += 1.0
#         self.current_step += 1
        
#         # 5. Compute rewards
#         rewards = self._compute_rewards(action_info)
        
#         # 6. Check termination
#         terminated = self.current_step >= self.max_steps
#         truncated = False
        
#         # 7. Build observation
#         obs = self._build_observation()
        
#         # 8. Aggregate reward
#         if isinstance(rewards, dict):
#             reward = float(sum(rewards.values()))
#         else:
#             reward = float(np.mean(rewards))
        
#         info = {
#             "action_mask": self.action_mask(),
#             "step": self.current_step,
#             "time": self.current_time,
#         }
        
#         if self.init_mode == "new":
#             info["batch"] = self.current_batch_idx
        
#         return obs, reward, terminated, truncated, info

#     def _build_observation(self):
#         """Build GNN observation using build_padded_ego_batch."""
        
#         # Robot list
#         robot_ids = sorted(list(self.robots.keys()))
#         if len(robot_ids) < self.num_robots:
#             robot_ids.extend([None] * (self.num_robots - len(robot_ids)))
#         robot_ids = robot_ids[:self.num_robots]
        
#         # Get candidates for each robot
#         candidate_lists = []
#         for robot_id in robot_ids:
#             if robot_id is None:
#                 candidate_lists.append([])
#             else:
#                 cands = self._get_candidate_tasks(robot_id)
#                 candidate_lists.append(cands)
        
#         # Call graph builder
#         obs_dict, cand_task_ids = build_padded_ego_batch(
#             robots=robot_ids,
#             tasks=self.tasks,
#             candidate_lists=candidate_lists,
#             N_max=self.N_max,
#             E_max=self.E_max,
#             K_max=self.K_max,
#             F=self.F,
#             G=0,
#             feature_fn=self.feature_fn,
#             two_hop=self.two_hop,
#             two_hop_directed=self.two_hop_directed,
#             normalize_features=True,
#             vicinity_m=self.vicinity_m,
#             use_edge_rt=(self.edge_feat_dim > 0),
#             edge_feat_dim=self.edge_feat_dim,
#             edge_features=self.edge_features,
#         )
        
#         # Store candidate IDs
#         self._last_cand_task_ids = cand_task_ids
        
#         return obs_dict

#     def _get_candidate_tasks(self, robot_id):
#         """Get candidate tasks for a robot."""
#         if robot_id not in self.robots:
#             return []
        
#         robot = self.robots[robot_id]
#         candidates = []
        
#         for task_id, task in self.tasks.items():
#             if task.get("is_assigned", False) or task.get("is_obsolete", False) or task.get("is_completed", False):
#                 continue
            
#             if task.get("release_time", 0) > self.current_time:
#                 continue
            
#             if task.get("pickup_deadline", float('inf')) < self.current_time:
#                 task["is_obsolete"] = True
#                 continue
            
#             dist = np.sqrt(
#                 (robot["x"] - task["pickup_x"]) ** 2 +
#                 (robot["y"] - task["pickup_y"]) ** 2
#             )
#             if dist <= self.vicinity_m:
#                 candidates.append(task_id)
        
#         if candidates:
#             candidates.sort(
#                 key=lambda tid: np.sqrt(
#                     (robot["x"] - self.tasks[tid]["pickup_x"]) ** 2 +
#                     (robot["y"] - self.tasks[tid]["pickup_y"]) ** 2
#                 )
#             )
        
#         return candidates

#     def _process_actions(self, actions: np.ndarray):
#         """Parse policy actions into task assignments."""
#         robot_ids = sorted(list(self.robots.keys()))
#         action_info = {}
        
#         for r_idx, action in enumerate(actions):
#             if r_idx >= len(robot_ids):
#                 break
            
#             robot_id = robot_ids[r_idx]
#             if robot_id is None:
#                 continue
            
#             action_info[robot_id] = {
#                 "action": int(action),
#                 "assigned_task": None,
#             }
            
#             if action == self._noop_index:
#                 continue
            
#             cands = self._get_candidate_tasks(robot_id)
#             if action < len(cands):
#                 task_id = cands[action]
#                 action_info[robot_id]["assigned_task"] = task_id
        
#         return action_info

#     def _update_robot_positions(self):
#         """Update robot positions."""
#         for robot_id, robot in self.robots.items():
#             if robot["assigned_tasks"]:
#                 next_task_id = robot["assigned_tasks"][0]
#                 next_task = self.tasks[next_task_id]
                
#                 dx = next_task["pickup_x"] - robot["x"]
#                 dy = next_task["pickup_y"] - robot["y"]
#                 dist = np.sqrt(dx**2 + dy**2)
                
#                 if dist > 1.0:
#                     speed = 1.0
#                     robot["x"] += (dx / dist) * speed
#                     robot["y"] += (dy / dist) * speed
#                 else:
#                     robot["x"] = next_task["dropoff_x"]
#                     robot["y"] = next_task["dropoff_y"]
#                     next_task["is_completed"] = True

#     def _update_task_assignments(self, action_info):
#         """Update task assignments."""
#         for robot_id, info in action_info.items():
#             task_id = info["assigned_task"]
#             if task_id:
#                 robot = self.robots[robot_id]
#                 if task_id not in robot["assigned_tasks"]:
#                     robot["assigned_tasks"].append(task_id)
#                     self.tasks[task_id]["is_assigned"] = True
#                     robot["capacity"] = min(
#                         robot["capacity"] + 1,
#                         self.max_robot_capacity
#                     )

#     def _compute_rewards(self, action_info):
#         """Compute rewards."""
#         robot_ids = sorted(list(self.robots.keys()))
#         rewards = {}
        
#         for r_idx, robot_id in enumerate(robot_ids[:self.num_robots]):
#             if robot_id is None:
#                 rewards[r_idx] = 0.0
#                 continue
            
#             reward = 0.0
#             reward -= 0.01
            
#             if robot_id in action_info:
#                 if action_info[robot_id]["assigned_task"] is not None:
#                     reward += 1.0
            
#             robot = self.robots[robot_id]
#             reward += 0.1 * robot["capacity"]
            
#             for task_id in robot["assigned_tasks"]:
#                 if self.tasks[task_id].get("is_completed", False):
#                     reward += 5.0
            
#             rewards[r_idx] = reward
        
#         return rewards

#     def action_mask(self):
#         """Get valid action mask."""
#         mask = np.zeros((self.num_robots, self.K_max + 1), dtype=np.uint8)
        
#         for r in range(self.num_robots):
#             for k in range(min(self.K_max, len(self._last_cand_task_ids[r]))):
#                 if self._last_cand_task_ids[r][k] is not None:
#                     mask[r, k] = 1
            
#             mask[r, self._noop_index] = 1
        
#         return mask

"""
Comprehensive multi-agent task allocation environment with full simulator logic.

Features:
- Robot movement & navigation
- Task pickup and dropoff handling
- Capacity management
- Deadline tracking and obsolescence
- Reward computation with multiple components
- GNN observation building
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import sys
import yaml
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.ego_graph_builder import build_padded_ego_batch
from src.utils.feature_fn import make_feature_fn, compute_feature_dim
from utils import utils as ut


class Planner:
    """Path planning with A* algorithm."""
    def __init__(self):
        root_path = Path(__file__).resolve().parent.parent.parent / "env"
        config_path = root_path / "ATC_wed.yaml"
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)
        map_path = root_path / params['map_filename']
        self.map_img = Image.open(map_path).convert('L')
        self.map_resolution = params['map_resolution']
        self.Planning_resolution = params['Planning_resolution']
        self.threshold = params['obstacle_threshold']
        self.origin_x = params['origin_x']
        self.origin_y = params['origin_y']
        self.average_velocity = params['average_velocity']

    def get_obstacle_grid(self):
        img_w, img_h = self.map_img.size
        scale = self.map_resolution / self.Planning_resolution
        grid_height, grid_width = int(img_h * scale), int(img_w * scale)
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        for row in range(grid_height):
            for col in range(grid_width):
                px = int((col + 0.5) * img_w / grid_width)
                py = int((row + 0.5) * img_h / grid_height)
                grid[row, col] = 1 if self.map_img.getpixel((px, py)) < (self.threshold * 255) else 0
        return grid

    def is_point_valid(self, point):
        grid = self.get_obstacle_grid()
        w = point[1]
        h = point[0]
        if 0 <= h < grid.shape[0] and 0 <= w < grid.shape[1]:
            return grid[h, w] == 0
        return False

    def get_plan(self, start, end):
        grid = self.get_obstacle_grid()
        found, path = ut.astar(grid, start, end)
        return found, path


class MultiAgentTaskEnv(gym.Env):
    """
    Multi-agent task allocation environment with full simulator.
    
    Handles:
    - Robot movement and navigation
    - Task pickup and dropoff
    - Capacity management
    - Deadline tracking
    - Reward computation
    - Observation generation with GNN graph
    """

    def __init__(
        self,
        # New mode parameters
        agents: np.ndarray = None,
        tasks_batches: list = None,
        # Old mode parameters (legacy)
        agents_cont_coord_array: np.ndarray = None,
        task_cont_coord_array: np.ndarray = None,
        # Feature extraction params
        use_xy_pickup: bool = False,
        normalize_features: bool = True,
        use_node_type: bool = True,
        use_ego_robot: bool = True,
        use_edge_rt: bool = False,
        edge_features=None,
        # Graph params
        N_max: int = 15,
        E_max: int = 50,
        K_max: int = 5,
        # Scaling params
        max_robot_capacity: int = 2,
        max_wait_delay_s: float = 600.0,
        max_travel_delay_s: float = 3600.0,
        max_steps: int = 1000,
        # 2-hop params
        two_hop: bool = False,
        two_hop_directed: bool = False,
        vicinity_m: float = 50.0,
        # Simulation params
        movement_speed: float = 1.0,  # units per step
        decision_interval: int = 1,   # how often to make decisions
        # Legacy parameters
        radius: int = 20,
        feature_size: int = 9,
        use_true_id: bool = False,
        reward_mode: str = "new",
    ):
        super().__init__()

        # Determine initialization mode
        if agents is not None and tasks_batches is not None:
            self.init_mode = "new"
            self.agents_data = agents
            self.tasks_batches = tasks_batches
            self.current_batch_idx = 0
            self.num_robots = len(agents)
        elif agents_cont_coord_array is not None and task_cont_coord_array is not None:
            self.init_mode = "old"
            self.agents_cont_coord_array = agents_cont_coord_array
            self.task_cont_coord_array = task_cont_coord_array
            self.num_robots = len(agents_cont_coord_array)
            self.radius = radius
            self.feature_size = feature_size
            self.use_true_id = use_true_id
            self.reward_mode = reward_mode
            self.planner = Planner()
        else:
            raise ValueError(
                "Must provide either (agents, tasks_batches) or "
                "(agents_cont_coord_array, task_cont_coord_array)"
            )

        # Common parameters
        self.N_max = N_max
        self.E_max = E_max
        self.K_max = K_max
        self.max_robot_capacity = max_robot_capacity
        self.vicinity_m = vicinity_m
        self.two_hop = two_hop
        self.two_hop_directed = two_hop_directed
        self.max_steps = max_steps
        self.movement_speed = movement_speed
        self.decision_interval = decision_interval

        # Initialize environment state
        self.robots = {}
        self.tasks = {}
        self.current_time = 0.0
        self.current_step = 0
        
        # Tracking for episode statistics
        self.episode_completed_count = 0
        self.episode_obsolete_count = 0
        self.episode_pickup_count = 0
        self.episode_dropoff_count = 0

        # Get map bounds
        if self.init_mode == "new":
            self.max_position = max(
                np.max(agents[:, 1]),  # max w
                np.max(agents[:, 2]),  # max h
            )
        else:
            self.max_position = 100.0

        # ====================================================================
        # FEATURE EXTRACTION & GRAPH BUILDING
        # ====================================================================

        # Compute feature dimension
        self.F = compute_feature_dim(
            use_xy_pickup=use_xy_pickup,
            use_node_type=use_node_type,
            use_edge_rt=use_edge_rt,
            use_ego_robot=use_ego_robot,
        )

        # Initialize feature function
        self.feature_fn = make_feature_fn(
            env_state=self,
            use_xy_pickup=use_xy_pickup,
            normalize_features=normalize_features,
            use_node_type=use_node_type,
            use_edge_rt=use_edge_rt,
            edge_features=edge_features or [],
            use_ego_robot=use_ego_robot,
            max_position=self.max_position,
            max_robot_capacity=max_robot_capacity,
            max_wait_delay_s=max_wait_delay_s,
            max_travel_delay_s=max_travel_delay_s,
            max_steps=max_steps,
        )

        # Determine edge feature dimension
        if use_edge_rt:
            self.edge_features = edge_features or ["dx", "dy", "eta"]
            self.edge_feat_dim = len(self.edge_features)
        else:
            self.edge_features = []
            self.edge_feat_dim = 0

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "x": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_robots, N_max, self.F),
                dtype=np.float32,
            ),
            "node_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.num_robots, N_max),
                dtype=np.uint8,
            ),
            "edge_index": gym.spaces.Box(
                low=0,
                high=N_max,
                shape=(self.num_robots, 2, E_max),
                dtype=np.int64,
            ),
            "edge_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.num_robots, E_max),
                dtype=np.uint8,
            ),
            "cand_idx": gym.spaces.Box(
                low=0,
                high=N_max,
                shape=(self.num_robots, K_max),
                dtype=np.int64,
            ),
            "cand_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.num_robots, K_max),
                dtype=np.uint8,
            ),
        })

        # Add edge_attr if needed
        if self.edge_feat_dim > 0:
            self.observation_space.spaces["edge_attr"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_robots, E_max, self.edge_feat_dim),
                dtype=np.float32,
            )

        # Action space
        self.action_space = gym.spaces.MultiDiscrete([K_max + 1] * self.num_robots)
        
        # NO-OP index
        self._noop_index = K_max
        
        # Tracking
        self._last_cand_task_ids = [[] for _ in range(self.num_robots)]

    def reset(self, seed=None):
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        self.current_time = 0.0
        self.current_step = 0
        self.episode_completed_count = 0
        self.episode_obsolete_count = 0
        self.episode_pickup_count = 0
        self.episode_dropoff_count = 0

        if self.init_mode == "new":
            self._reset_new_mode()
        else:
            self._reset_old_mode()

        obs = self._build_observation()
        return obs, {"action_mask": self.action_mask()}

    def _reset_new_mode(self):
        """Reset for new mode (generated data)."""
        # Initialize robots from agents data
        self.robots = {}
        for i, agent in enumerate(self.agents_data):
            robot_id = int(agent[0])
            self.robots[robot_id] = {
                "id": robot_id,
                "x": float(agent[1]),
                "y": float(agent[2]),
                "current_capacity": 0,  # Number of tasks currently carrying
                "assigned_tasks": [],   # Queue of tasks to complete
                "current_task": None,   # Currently executing task (pickup or dropoff phase)
                "task_phase": None,     # "pickup", "travel_to_dropoff", or "dropoff"
                "target_location": None,  # Current movement target
                "path": [],             # Current path
            }

        # Initialize tasks from current batch
        self.tasks = {}
        batch = self.tasks_batches[self.current_batch_idx]
        
        for task_data in batch:
            task_id = int(task_data[0])
            self.tasks[task_id] = {
                "id": task_id,
                "pickup_x": float(task_data[1]),
                "pickup_y": float(task_data[2]),
                "dropoff_x": float(task_data[3]),
                "dropoff_y": float(task_data[4]),
                "release_time": float(task_data[5]),
                "pickup_deadline": float(task_data[6]),
                "est_travel_time": float(task_data[7]),
                "dropoff_deadline": float(task_data[8]),
                "is_assigned": False,
                "is_obsolete": False,
                "is_picked_up": False,
                "is_completed": False,
                "assigned_robot": None,
            }

    def _reset_old_mode(self):
        """Reset for old mode (legacy format)."""
        pass

    def step(self, actions: np.ndarray):
        """Execute one environment step with full simulation."""
        
        # 1. Process actions and assign tasks to robots
        action_info = self._process_actions(actions)
        
        # 2. Update task deadlines and mark obsolete
        self._update_task_deadlines()
        
        # 3. Execute robot movements and task phases
        self._execute_robot_movements_and_tasks()
        
        # 4. Advance time
        self.current_time += 1.0
        self.current_step += 1
        
        # 5. Compute rewards
        reward = self._compute_rewards(action_info)
        
        # 6. Check termination
        terminated = self._check_episode_done()
        truncated = self.current_step >= self.max_steps
        
        # 7. Build observation
        obs = self._build_observation()
        
        # 8. Prepare info
        info = {
            "action_mask": self.action_mask(),
            "step": self.current_step,
            "time": self.current_time,
            "completed_count": self.episode_completed_count,
            "obsolete_count": self.episode_obsolete_count,
            "pickup_count": self.episode_pickup_count,
            "dropoff_count": self.episode_dropoff_count,
        }
        
        if self.init_mode == "new":
            info["batch"] = self.current_batch_idx
        
        return obs, reward, terminated, truncated, info

    # ========================================================================
    # TASK ASSIGNMENT & ACTION PROCESSING
    # ========================================================================

    def _process_actions(self, actions: np.ndarray) -> Dict[int, Dict]:
        """
        Process policy actions and assign tasks to robots.
        
        Returns:
            action_info: Dict mapping robot_id to action details
        """
        robot_ids = sorted(list(self.robots.keys()))
        action_info = {}
        
        for r_idx, action in enumerate(actions):
            if r_idx >= len(robot_ids):
                break
            
            robot_id = robot_ids[r_idx]
            if robot_id is None:
                continue
            
            action_info[robot_id] = {
                "action": int(action),
                "assigned_task": None,
            }
            
            # NO-OP: do nothing
            if action == self._noop_index:
                continue
            
            # Try to assign candidate task
            cands = self._get_candidate_tasks(robot_id)
            if action < len(cands):
                task_id = cands[action]
                robot = self.robots[robot_id]
                
                # Check if robot has capacity
                if robot["current_capacity"] < self.max_robot_capacity:
                    task = self.tasks[task_id]
                    
                    # Assign task
                    action_info[robot_id]["assigned_task"] = task_id
                    robot["assigned_tasks"].append(task_id)
                    task["is_assigned"] = True
                    task["assigned_robot"] = robot_id
        
        return action_info

    def _get_candidate_tasks(self, robot_id) -> List[int]:
        """Get list of available candidate tasks for a robot."""
        if robot_id not in self.robots:
            return []
        
        robot = self.robots[robot_id]
        candidates = []
        
        for task_id, task in self.tasks.items():
            # Skip if already assigned or completed
            if task.get("is_assigned", False) or task.get("is_completed", False):
                continue
            
            # Skip if obsolete
            if task.get("is_obsolete", False):
                continue
            
            # Skip if not released yet
            if task.get("release_time", 0) > self.current_time:
                continue
            
            # Skip if pickup deadline passed
            if task.get("pickup_deadline", float('inf')) <= self.current_time:
                continue
            
            # Check distance to pickup
            dist = np.sqrt(
                (robot["x"] - task["pickup_x"]) ** 2 +
                (robot["y"] - task["pickup_y"]) ** 2
            )
            if dist <= self.vicinity_m:
                candidates.append(task_id)
        
        # Sort by distance (nearest first)
        if candidates:
            candidates.sort(
                key=lambda tid: np.sqrt(
                    (robot["x"] - self.tasks[tid]["pickup_x"]) ** 2 +
                    (robot["y"] - self.tasks[tid]["pickup_y"]) ** 2
                )
            )
        
        return candidates

    # ========================================================================
    # TASK LIFECYCLE MANAGEMENT
    # ========================================================================

    def _update_task_deadlines(self):
        """Mark tasks as obsolete if deadlines passed."""
        for task_id, task in self.tasks.items():
            if task.get("is_obsolete", False) or task.get("is_completed", False):
                continue
            
            # Mark as obsolete if pickup deadline passed and not picked up
            if not task.get("is_picked_up", False):
                if task.get("pickup_deadline", float('inf')) <= self.current_time:
                    task["is_obsolete"] = True
                    self.episode_obsolete_count += 1
            
            # Mark as obsolete if dropoff deadline passed and not completed
            else:
                if task.get("dropoff_deadline", float('inf')) <= self.current_time:
                    task["is_obsolete"] = True
                    self.episode_obsolete_count += 1

    def _execute_robot_movements_and_tasks(self):
        """
        Execute full robot state machine:
        1. If no current task: pick up next task from queue
        2. Move toward current destination
        3. Handle pickup/dropoff actions
        4. Complete task and move to next
        """
        for robot_id, robot in self.robots.items():
            # If no current task, try to pick up next one
            if robot["current_task"] is None and robot["assigned_tasks"]:
                next_task_id = robot["assigned_tasks"].pop(0)
                robot["current_task"] = next_task_id
                robot["task_phase"] = "pickup"
                robot["target_location"] = (
                    self.tasks[next_task_id]["pickup_x"],
                    self.tasks[next_task_id]["pickup_y"],
                )
            
            # If we have a current task, execute it
            if robot["current_task"] is not None:
                self._move_robot_toward_target(robot_id)
            else:
                # Idle: no tasks to do
                pass

    def _move_robot_toward_target(self, robot_id):
        """
        Move robot toward target. Handle pickup/dropoff when reaching destination.
        """
        robot = self.robots[robot_id]
        target_x, target_y = robot["target_location"]
        current_x, current_y = robot["x"], robot["y"]
        
        # Calculate distance to target
        dx = target_x - current_x
        dy = target_y - current_y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0.1:  # Not yet at target
            # Move toward target
            move_dist = min(self.movement_speed, dist)
            robot["x"] += (dx / dist) * move_dist
            robot["y"] += (dy / dist) * move_dist
        else:
            # Reached target, execute task action
            task_id = robot["current_task"]
            task = self.tasks[task_id]
            
            if robot["task_phase"] == "pickup":
                # Execute pickup
                robot["current_capacity"] += 1
                task["is_picked_up"] = True
                self.episode_pickup_count += 1
                
                # Move to dropoff
                robot["task_phase"] = "travel_to_dropoff"
                robot["target_location"] = (task["dropoff_x"], task["dropoff_y"])
            
            elif robot["task_phase"] == "travel_to_dropoff":
                # At dropoff location, execute dropoff
                robot["current_capacity"] = max(0, robot["current_capacity"] - 1)
                task["is_completed"] = True
                self.episode_dropoff_count += 1
                self.episode_completed_count += 1
                
                # Clear current task and move to next
                robot["current_task"] = None
                robot["task_phase"] = None
                robot["target_location"] = None

    # ========================================================================
    # OBSERVATION & GRAPH BUILDING
    # ========================================================================

    def _build_observation(self) -> Dict:
        """Build GNN observation using build_padded_ego_batch."""
        
        # Robot list
        robot_ids = sorted(list(self.robots.keys()))
        if len(robot_ids) < self.num_robots:
            robot_ids.extend([None] * (self.num_robots - len(robot_ids)))
        robot_ids = robot_ids[:self.num_robots]
        
        # Get candidates for each robot
        candidate_lists = []
        for robot_id in robot_ids:
            if robot_id is None:
                candidate_lists.append([])
            else:
                cands = self._get_candidate_tasks(robot_id)
                candidate_lists.append(cands)
        
        # Call graph builder
        obs_dict, cand_task_ids = build_padded_ego_batch(
            robots=robot_ids,
            tasks=self.tasks,
            candidate_lists=candidate_lists,
            N_max=self.N_max,
            E_max=self.E_max,
            K_max=self.K_max,
            F=self.F,
            G=0,
            feature_fn=self.feature_fn,
            two_hop=self.two_hop,
            two_hop_directed=self.two_hop_directed,
            normalize_features=True,
            vicinity_m=self.vicinity_m,
            use_edge_rt=(self.edge_feat_dim > 0),
            edge_feat_dim=self.edge_feat_dim,
            edge_features=self.edge_features,
        )
        
        # Store candidate IDs
        self._last_cand_task_ids = cand_task_ids
        
        return obs_dict

    # ========================================================================
    # REWARD COMPUTATION
    # ========================================================================

    def _compute_rewards(self, action_info: Dict) -> float:
        """
        Compute composite reward:
        - Penalty for each time step
        - Reward for task assignments
        - Reward for task completions
        - Penalty for obsolete tasks
        - Bonus for efficient execution
        """
        robot_ids = sorted(list(self.robots.keys()))
        total_reward = 0.0
        
        # Per-robot rewards
        for r_idx, robot_id in enumerate(robot_ids[:self.num_robots]):
            if robot_id is None:
                continue
            
            robot = self.robots[robot_id]
            
            # Time penalty
            reward = -0.01
            
            # Assignment reward
            if robot_id in action_info:
                if action_info[robot_id]["assigned_task"] is not None:
                    reward += 1.0
            
            # Capacity bonus
            reward += 0.05 * robot["current_capacity"]
            
            # Pickup bonus (only first time)
            # (handled in episode tracking)
            
            total_reward += reward
        
        # Episode-level rewards
        # Completion bonus
        total_reward += 5.0 * self.episode_completed_count
        
        # Obsolete penalty
        total_reward -= 2.0 * self.episode_obsolete_count
        
        return float(total_reward)

    def _check_episode_done(self) -> bool:
        """
        Episode terminates when:
        - All available tasks are completed
        - All released tasks are either completed or obsolete
        """
        # Count active tasks (available, not completed, not obsolete)
        active_tasks = 0
        for task_id, task in self.tasks.items():
            if not task.get("is_completed", False) and not task.get("is_obsolete", False):
                if task.get("release_time", 0) <= self.current_time:
                    active_tasks += 1
        
        # Also check if all robots are idle
        robots_idle = all(
            len(robot["assigned_tasks"]) == 0 and robot["current_task"] is None
            for robot in self.robots.values()
        )
        
        # Episode done if no active tasks and all robots idle
        if active_tasks == 0 and robots_idle:
            return True
        
        return False

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def action_mask(self) -> np.ndarray:
        """Get valid action mask."""
        mask = np.zeros((self.num_robots, self.K_max + 1), dtype=np.uint8)
        
        for r in range(self.num_robots):
            # Mark valid candidates
            for k in range(min(self.K_max, len(self._last_cand_task_ids[r]))):
                if self._last_cand_task_ids[r][k] is not None:
                    mask[r, k] = 1
            
            # NO-OP always allowed
            mask[r, self._noop_index] = 1
        
        return mask

    def close(self):
        """Clean up resources."""
        pass
