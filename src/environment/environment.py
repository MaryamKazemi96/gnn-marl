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
import torch as th
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
        agents: np.ndarray = None,
        tasks_batches: list = None,
        agents_cont_coord_array: np.ndarray = None,
        task_cont_coord_array: np.ndarray = None,
        use_xy_pickup: bool = False,
        normalize_features: bool = True,
        use_node_type: bool = True,
        use_ego_robot: bool = True,
        use_edge_rt: bool = False,
        edge_features=None,
        N_max: int = 15,
        E_max: int = 50,
        K_max: int = 5,
        max_robot_capacity: int = 2,
        max_wait_delay_s: float = 60.0,
        max_travel_delay_s: float = 3600.0,
        max_steps: int = 1000,
        two_hop: bool = False,
        two_hop_directed: bool = False,
        vicinity_m: float = 40.0,
        movement_speed: float = 1.0,
        decision_interval: int = 7,
        radius: int = 20,
        feature_size: int = 9,
        use_true_id: bool = False,
        reward_mode: str = "new",
    ):
        super().__init__()

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
        self.max_wait_delay_s = max_wait_delay_s

        self.robots = {}
        self.tasks = {}
        self.current_time = 0.0
        self.current_step = 0

        self.episode_completed_count = 0
        self.episode_obsolete_count = 0
        self.episode_pickup_count = 0
        self.episode_dropoff_count = 0
        self._prev_completed_count = 0
        self._prev_obsolete_count = 0
        self._prev_pickup_count = 0
        self._prev_dropoff_count = 0

        if self.init_mode == "new":
            self.max_position = max(
                np.max(agents[:, 1]),
                np.max(agents[:, 2]),
            )
        else:
            self.max_position = 100.0

        self.F = compute_feature_dim(
            use_xy_pickup=use_xy_pickup,
            use_node_type=use_node_type,
            use_edge_rt=use_edge_rt,
            use_ego_robot=use_ego_robot,
        )

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

        if use_edge_rt:
            self.edge_features = edge_features or ["dx", "dy", "eta"]
            self.edge_feat_dim = len(self.edge_features)
        else:
            self.edge_features = []
            self.edge_feat_dim = 0

        self.observation_space = gym.spaces.Dict({
            "x": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_robots, N_max, self.F),
                dtype=np.float32,
            ),
            "node_mask": gym.spaces.Box(
                low=0, high=1,
                shape=(self.num_robots, N_max),
                dtype=np.uint8,
            ),
            "edge_index": gym.spaces.Box(
                low=0, high=N_max,
                shape=(self.num_robots, 2, E_max),
                dtype=np.int64,
            ),
            "edge_mask": gym.spaces.Box(
                low=0, high=1,
                shape=(self.num_robots, E_max),
                dtype=np.uint8,
            ),
            "cand_idx": gym.spaces.Box(
                low=0, high=N_max,
                shape=(self.num_robots, K_max),
                dtype=np.int64,
            ),
            "cand_mask": gym.spaces.Box(
                low=0, high=1,
                shape=(self.num_robots, K_max),
                dtype=np.uint8,
            ),
        })

        if self.edge_feat_dim > 0:
            self.observation_space.spaces["edge_attr"] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_robots, E_max, self.edge_feat_dim),
                dtype=np.float32,
            )

        self.action_space = gym.spaces.MultiDiscrete([K_max + 1] * self.num_robots)
        self._noop_index = K_max
        self._last_cand_task_ids = [[] for _ in range(self.num_robots)]

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self, seed=None):
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        self.current_time = 0.0
        self.current_step = 0
        self.episode_completed_count = 0
        self.episode_obsolete_count = 0
        self.episode_pickup_count = 0
        self.episode_dropoff_count = 0
        self._prev_completed_count   = 0
        self._prev_obsolete_count    = 0
        self._prev_pickup_count      = 0
        self._prev_dropoff_count     = 0
        if self.init_mode == "new":
            self._reset_new_mode()
            # Advance batch for next episode
            self.current_batch_idx = (self.current_batch_idx + 1) % len(self.tasks_batches)
        else:
            self._reset_old_mode()

        obs = self._build_observation()
        return obs, {"action_mask": self.action_mask()}

    def _reset_new_modeold(self):
        """Reset robots and tasks. Tasks injected via _release_pending_tasks."""
        self.robots = {}
        for agent in self.agents_data:
            robot_id = int(agent[0])
            self.robots[robot_id] = {
                "id": robot_id,
                "x": float(agent[1]),
                "y": float(agent[2]),
                "current_capacity": 0,
                "assigned_tasks": [],
                "current_task": None,
                "task_phase": None,
                "target_location": None,
                "path": [],
            }

        # Start with empty task pool — release_pending_tasks fills it
        self.tasks = {}
        self._release_pending_tasks()
    def _reset_new_mode(self):
        self.robots = {}
        for agent in self.agents_data:
            robot_id = str(int(agent[0]))   # <<< str key
            self.robots[robot_id] = {
                "id": robot_id,
                "x": float(agent[1]),
                "y": float(agent[2]),
                "max_capacity": self.max_robot_capacity,
                "current_capacity": 0,
                "assigned_tasks": [],
                "current_task": None,
                "task_phase": None,
                "target_location": None,
                "just_picked_up_task": None,  # <<< ADD: for shaped reward
            }
        self.tasks = {}
        batch = self.tasks_batches[self.current_batch_idx]
        for task_data in batch:
            task_id = str(int(task_data[0]))   # <<< str key
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
        """Reset for legacy mode."""
        pass

    # =========================================================================
    # TASK RELEASE
    # =========================================================================

    def _release_pending_tasks(self):
        """Inject tasks from all batches whose release_time <= current_time."""
        newly_released = 0
        for batch in self.tasks_batches:
            for task_data in batch:
                task_id = str(int(task_data[0]))
                if task_id in self.tasks:
                    continue
                if float(task_data[5]) <= self.current_time:
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
                    newly_released += 1
        # if newly_released:
        #     print(f"Step {self.current_step}: Released {newly_released} new tasks, "
        #           f"total={len(self.tasks)}")

    # =========================================================================
    # STEP
    # =========================================================================

    def stepold(self, actions: np.ndarray):
        """Execute one environment step."""
        prev_completed = self.episode_completed_count
        prev_obsolete  = self.episode_obsolete_count

        # 1. Assign tasks based on last observation's candidates
        action_info = self._process_actions(actions)

        # 2. Inject newly released tasks
        self._release_pending_tasks()

        # 3. Mark expired tasks as obsolete, free robot capacity
        self._update_task_deadlines()

        # 4. Move robots and execute pickup/dropoff
        self._execute_robot_movements_and_tasks()

        # 5. Advance time
        self.current_time += 1.0
        self.current_step += 1

        # 6. Compute reward using deltas
        reward = self._compute_rewards(
            action_info,
            completed_delta=self.episode_completed_count - prev_completed,
            obsolete_delta=self.episode_obsolete_count  - prev_obsolete,
        )

        # 7. Check termination
        terminated = self._check_episode_done()
        truncated  = self.current_step >= self.max_steps
        print(f"Step {self.current_step}: reward={reward:.2f}, ")
        print(f"terminated={terminated}, truncated={truncated}, completed={self.episode_completed_count}, ")
        # 8. Build observation
        obs = self._build_observation()

        info = {
            "action_mask":      self.action_mask(),
            "step":             self.current_step,
            "time":             self.current_time,
            "completed_count":  self.episode_completed_count,
            "obsolete_count":   self.episode_obsolete_count,
            "pickup_count":     self.episode_pickup_count,
            "dropoff_count":    self.episode_dropoff_count,
            "batch":            self.current_batch_idx,
        }

        return obs, reward, terminated, truncated, info
    def step(self, actions):
        action_info = self._process_actions(actions)
        self._update_task_deadlines()
        self._execute_robot_movements_and_tasks()
        self.current_time += 1.0
        self.current_step += 1
        reward = self._compute_rewards(action_info)
        
        # Inner no-op steps
        
        for _ in range(self.decision_interval - 1):
            if self.current_step >= self.max_steps:
                break
            self._update_task_deadlines()
            self._execute_robot_movements_and_tasks()
            self.current_time += 1.0
            self.current_step += 1
            reward += self._compute_rewards({})   # no-op actions
            if self._check_episode_done():
                break
        
        terminated = self._check_episode_done()
        truncated  = self.current_step >= self.max_steps
        # print(f"Step {self.current_step}: reward={reward:.2f}, ")
        # print(f"terminated={terminated}, truncated={truncated}, completed={self.episode_completed_count}, ")
        
        obs = self._build_observation()
        info = {
            "action_mask":      self.action_mask(),
            "step":             self.current_step,
            "time":             self.current_time,
            "completed_count":  self.episode_completed_count,
            "obsolete_count":   self.episode_obsolete_count,
            "pickup_count":     self.episode_pickup_count,
            "dropoff_count":    self.episode_dropoff_count,
            "batch":            self.current_batch_idx,
        }
        return obs, reward, terminated, truncated, info
    # =========================================================================
    # ACTION PROCESSING
    # =========================================================================

    def _process_actionsold(self, actions: np.ndarray) -> Dict[int, Dict]:
        """
        Map policy actions to task assignments using last observation's candidates.
        Reuses _last_cand_task_ids to avoid double-computing candidates.
        """
        robot_ids   = sorted(self.robots.keys())
        action_info = {}

        for r_idx, action in enumerate(actions):
            if r_idx >= len(robot_ids):
                break

            robot_id = robot_ids[r_idx]
            action_info[robot_id] = {"action": int(action), "assigned_task": None}

            if action == self._noop_index:
                continue

            # Reuse candidates from last _build_observation call
            cands = [tid for tid in self._last_cand_task_ids[r_idx] if tid is not None]

            if action >= len(cands):
                continue

            task_id = cands[action]
            robot   = self.robots[robot_id]

            # Capacity check: counts assigned-but-not-dropped-off tasks
            if robot["current_capacity"] < self.max_robot_capacity:
                task = self.tasks.get(task_id)
                if task is None or task.get("is_assigned") or task.get("is_obsolete"):
                    continue  # task gone between obs and action

                action_info[robot_id]["assigned_task"] = task_id
                robot["assigned_tasks"].append(task_id)
                task["is_assigned"]    = True
                task["assigned_robot"] = robot_id
                robot["current_capacity"] += 1  # increment at assignment, decrement at dropoff

        return action_info
    def _process_actions(self, actions):
        # Collect all (robot, task, distance) assignment requests
        robot_ids   = sorted(self.robots.keys())
        action_info = {}
        requests = []
        for r_idx, action in enumerate(actions):
            if action == self._noop_index:
                continue
            robot_id = robot_ids[r_idx]
            cands = self._get_candidate_tasks(robot_id)
            if action < len(cands):
                task_id = cands[action]
                robot = self.robots[robot_id]
                dist = np.sqrt((robot["x"] - self.tasks[task_id]["pickup_x"])**2 +
                            (robot["y"] - self.tasks[task_id]["pickup_y"])**2)
                requests.append((dist, robot_id, task_id))
        
        # Sort by distance, assign greedily (closest wins conflicts)
        requests.sort()
        assigned_tasks = set()
        action_info = {}
        # for dist, robot_id, task_id in requests:
        #     if task_id in assigned_tasks:
        #         action_info[robot_id] = {"action": ..., "assigned_task": None}
        #         continue
        #     robot = self.robots[robot_id]
        #     if robot["current_capacity"] < self.max_robot_capacity:
        #         robot["assigned_tasks"].append(task_id)
        #         self.tasks[task_id]["is_assigned"] = True
        #         assigned_tasks.add(task_id)
        #         action_info[robot_id] = {"action": ..., "assigned_task": task_id}
        for dist, robot_id, task_id in requests:
            if task_id in assigned_tasks:
                continue
            robot = self.robots[robot_id]
            # Total claimed = physically onboard + queued for pickup
            total_claimed = robot["current_capacity"] + len(robot["assigned_tasks"])
            if total_claimed < self.max_robot_capacity:
                robot["assigned_tasks"].append(task_id)
                self.tasks[task_id]["is_assigned"]    = True
                self.tasks[task_id]["assigned_robot"] = robot_id
                assigned_tasks.add(task_id)
                action_info[robot_id] = {"action": ..., "assigned_task": task_id}
        return action_info
    # =========================================================================
    # CANDIDATE TASKS
    # =========================================================================

    def _get_candidate_tasksold(self, robot_id) -> List[int]:
        """
        Return available tasks within vicinity of robot, sorted by distance.
        Fallback to K_max nearest tasks if none are within vicinity.
        """
        if robot_id not in self.robots:
            return []

        robot    = self.robots[robot_id]
        eligible = []

        for task_id, task in self.tasks.items():
            if task.get("is_assigned") or task.get("is_completed"):
                continue
            if task.get("is_obsolete"):
                continue
            if task.get("release_time", 0) > self.current_time:
                continue
            if task.get("pickup_deadline", float("inf")) <= self.current_time:
                continue
            eligible.append(task_id)

        if not eligible:
            return []

        # Sort by distance to pickup
        eligible.sort(key=lambda tid: np.sqrt(
            (robot["x"] - self.tasks[tid]["pickup_x"]) ** 2 +
            (robot["y"] - self.tasks[tid]["pickup_y"]) ** 2
        ))

        # Tasks within vicinity
        nearby = [
            tid for tid in eligible
            if np.sqrt(
                (robot["x"] - self.tasks[tid]["pickup_x"]) ** 2 +
                (robot["y"] - self.tasks[tid]["pickup_y"]) ** 2
            ) <= self.vicinity_m
        ]

        # Return nearby if any, else fallback to K_max nearest
        return nearby[:self.K_max] if nearby else eligible[:self.K_max]
    def _get_candidate_tasks(self, robot_id) -> List[str]:
        robot = self.robots.get(str(robot_id))
        if robot is None:
            return []
        candidates = []
        for task_id, task in self.tasks.items():
            if task.get("is_assigned") or task.get("is_completed") or task.get("is_obsolete"):
                continue
            if task.get("release_time", 0) > self.current_time:
                continue
            if task.get("pickup_deadline", float("inf")) <= self.current_time:
                continue
            dist = np.sqrt((robot["x"] - task["pickup_x"])**2 +
                        (robot["y"] - task["pickup_y"])**2)
            if dist <= self.vicinity_m:
                candidates.append(task_id)   # <<< already str
        candidates.sort(key=lambda tid: np.sqrt(
            (robot["x"] - self.tasks[tid]["pickup_x"])**2 +
            (robot["y"] - self.tasks[tid]["pickup_y"])**2
        ))
        return candidates[:self.K_max]
    # =========================================================================
    # TASK LIFECYCLE
    # =========================================================================

    def _update_task_deadlines(self):
        """Mark expired tasks obsolete and free robot capacity."""
        for task_id, task in self.tasks.items():
            if task.get("is_obsolete") or task.get("is_completed"):
                continue

            became_obsolete = False
            if not task.get("is_picked_up"):
                if task.get("pickup_deadline", float("inf")) <= self.current_time:
                    became_obsolete = True
            else:
                if task.get("dropoff_deadline", float("inf")) <= self.current_time:
                    became_obsolete = True

            if not became_obsolete:
                continue

            task["is_obsolete"] = True
            self.episode_obsolete_count += 1

            # Free the assigned robot
            assigned_robot_id = task.get("assigned_robot")
            if assigned_robot_id and assigned_robot_id in self.robots:
                robot = self.robots[assigned_robot_id]
                robot["current_capacity"] = max(0, robot["current_capacity"] - 1)
                if task_id in robot["assigned_tasks"]:
                    robot["assigned_tasks"].remove(task_id)
                if robot["current_task"] == task_id:
                    robot["current_task"]      = None
                    robot["task_phase"]        = None
                    robot["target_location"]   = None

    def _execute_robot_movements_and_tasks(self):
        """Advance robot state machine: dequeue tasks, move, pickup, dropoff."""
        for robot_id, robot in self.robots.items():
            # Dequeue next task if idle
            if robot["current_task"] is None and robot["assigned_tasks"]:
                next_task_id = robot["assigned_tasks"].pop(0)
                # Guard: task may have gone obsolete while queued
                if self.tasks.get(next_task_id, {}).get("is_obsolete"):
                    continue
                robot["current_task"]    = next_task_id
                robot["task_phase"]      = "pickup"
                robot["target_location"] = (
                    self.tasks[next_task_id]["pickup_x"],
                    self.tasks[next_task_id]["pickup_y"],
                )

            if robot["current_task"] is not None:
                self._move_robot_toward_target(robot_id)

    def _move_robot_toward_target(self, robot_id):
        """Move robot one step toward target; execute pickup/dropoff on arrival."""
        robot    = self.robots[robot_id]
        target_x, target_y = robot["target_location"]
        dx = target_x - robot["x"]
        dy = target_y - robot["y"]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist > 0.1:
            move_dist  = min(self.movement_speed, dist)
            robot["x"] += (dx / dist) * move_dist
            robot["y"] += (dy / dist) * move_dist
            return

        # Arrived at target
        task_id = robot["current_task"]
        task    = self.tasks[task_id]

        if robot["task_phase"] == "pickup":
            robot["current_capacity"] += 1
            task["is_picked_up"] = True
            self.episode_pickup_count += 1
            robot["just_picked_up_task"] = task_id   # <<< ADD for reward shaping
            robot["task_phase"] = "travel_to_dropoff"
            robot["target_location"] = (task["dropoff_x"], task["dropoff_y"])
        elif robot["task_phase"] == "travel_to_dropoff":
            robot["current_capacity"]   = max(0, robot["current_capacity"] - 1)
            task["is_completed"]        = True
            self.episode_dropoff_count  += 1
            self.episode_completed_count += 1
            robot["current_task"]       = None
            robot["task_phase"]         = None
            robot["target_location"]    = None

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    def _build_observation(self) -> Dict:
        """Build GNN observation using build_padded_ego_batch."""
        robot_ids = sorted(self.robots.keys())
        # Pad to num_robots if needed
        if len(robot_ids) < self.num_robots:
            robot_ids += [None] * (self.num_robots - len(robot_ids))
        robot_ids = robot_ids[:self.num_robots]

        candidate_lists = [
            self._get_candidate_tasks(rid) if rid is not None else []
            for rid in robot_ids
        ]

        obs_dict, cand_task_ids = build_padded_ego_batch(
            robots=robot_ids,
            robots_dict=self.robots,          # ← robot positions for 2-hop
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

        self._last_cand_task_ids = cand_task_ids
        return obs_dict

    # =========================================================================
    # REWARD
    # =========================================================================
    def _compute_rewards(self, action_info):
        new_complete = self.episode_completed_count - self._prev_completed_count
        new_obsolete = self.episode_obsolete_count  - self._prev_obsolete_count
        new_pickups  = self.episode_pickup_count    - self._prev_pickup_count
        self._prev_completed_count = self.episode_completed_count
        self._prev_obsolete_count  = self.episode_obsolete_count
        self._prev_pickup_count    = self.episode_pickup_count

        W_COMP     = 1.0
        W_WAIT     = 1.5
        W_DEADLINE = 5.0
        WAIT_CAP   = max(1.0, float(self.max_wait_delay_s))

        reward = float(new_complete) * W_COMP

        # Wait penalty at pickup time
        for robot in self.robots.values():
            picked_id = robot.get("just_picked_up_task")
            if picked_id:
                t = self.tasks.get(picked_id)
                if t is not None:
                    wait_s = max(0.0, self.current_time - t["release_time"])
                    reward += -(min(wait_s, WAIT_CAP) / WAIT_CAP) * W_WAIT
                robot["just_picked_up_task"] = None

        # Obsolete penalty
        reward -= float(new_obsolete) * (W_DEADLINE * 0.5)

        return reward
    def _compute_rewardsold(
        self,
        action_info: Dict,
        completed_delta: int = 0,
        obsolete_delta: int = 0,
    ) -> float:
        """
        Composite reward:
          - Small time penalty per robot per step
          - Bonus for each new task assignment
          - Capacity utilisation bonus
          - Completion bonus (delta only)
          - Obsolete penalty (delta only)
        """
        total = 0.0

        for robot_id in sorted(self.robots.keys())[:self.num_robots]:
            robot  = self.robots[robot_id]
            r      = -0.01                                          # time penalty

            if action_info.get(robot_id, {}).get("assigned_task"):
                r += 1.0                                            # assignment bonus

            r     += 0.05 * robot["current_capacity"]              # utilisation bonus
            total += r

        total += 5.0 * completed_delta                              # completion bonus
        total -= 2.0 * obsolete_delta                               # obsolete penalty

        return float(total)

    # =========================================================================
    # TERMINATION
    # =========================================================================

    # def _check_episode_doneold(self) -> bool:
    #     """
    #     Done when:
    #     - No future tasks remain unreleased across all batches, AND
    #     - All released tasks are completed or obsolete, AND
    #     - All robots are idle
    #     """
    #     # Any task not yet injected into self.tasks?
    #     for batch in self.tasks_batches:
    #         for task_data in batch:
    #             if int(task_data[0]) not in self.tasks:
    #                 print(f"Episode not done: task {int(task_data[0])} not yet released")
    #                 return False

    #     active_tasks = sum(
    #         1 for task in self.tasks.values()
    #         if not task.get("is_completed") and not task.get("is_obsolete")
    #     )

    #     robots_idle = all(
    #         len(r["assigned_tasks"]) == 0 and r["current_task"] is None
    #         for r in self.robots.values()s
    #     )
    #     # print(f"Checking episode done: active_tasks={active_tasks}, robots_idle={robots_idle}")
    #     return active_tasks == 0 and robots_idle
    def _check_episode_done(self):
        any_pending = any(
            not t.get("is_completed") and not t.get("is_obsolete")
            for t in self.tasks.values()
        )
        robots_idle = all(
            len(r["assigned_tasks"]) == 0 and r["current_task"] is None
            for r in self.robots.values()
        )
        # Only done when nothing is left to do AND no unreleased future tasks
        return (not any_pending) and robots_idle
    # =========================================================================
    # UTILITIES
    # =========================================================================

    def action_mask(self) -> np.ndarray:
        """Return valid action mask (R, K_max+1). NO-OP always valid."""
        mask = np.zeros((self.num_robots, self.K_max + 1), dtype=np.uint8)

        for r in range(self.num_robots):
            for k in range(min(self.K_max, len(self._last_cand_task_ids[r]))):
                if self._last_cand_task_ids[r][k] is not None:
                    mask[r, k] = 1
            mask[r, self._noop_index] = 1  # NO-OP always allowed

        return mask

    def close(self):
        pass