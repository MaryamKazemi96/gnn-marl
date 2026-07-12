"""
Comprehensive multi-agent task allocation environment with full simulator logic.

Features:
- Robot movement & navigation via A* path planning
- Task pickup and dropoff handling
- Capacity management
- Deadline tracking and obsolescence
- Reward computation with multiple components
- GNN observation building

Patched updates:
- Added diagnostics for action validity/masking/reward components.
- Action decoding now uses observation-time candidate snapshot (_last_cand_task_ids).
- Dropoff time is set at dropoff event (before reward read).
- If task is already picked up, it is NOT obsoleted on deadline; delivery continues and lateness is penalized in reward.
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


# =============================================================================
# Planner — A* path planning with cached obstacle grid
# =============================================================================

class Planner:
    """
    A* path planner over the ATC obstacle grid.

    The obstacle grid is built once from the map image and cached on the
    instance so that every call to get_plan() / is_point_valid() reuses the
    same array instead of re-sampling the PNG pixel-by-pixel each time.
    """

    def __init__(self):
        root_path = Path(__file__).resolve().parent.parent.parent / "env"
        config_path = root_path / "ATC_wed.yaml"
        with open(config_path, "r") as fh:
            params = yaml.safe_load(fh)

        map_path = root_path / params["map_filename"]
        self.map_img = Image.open(map_path).convert("L")
        self.map_resolution = params["map_resolution"]
        self.Planning_resolution = params["Planning_resolution"]
        self.threshold = params["obstacle_threshold"]
        self.origin_x = params["origin_x"]
        self.origin_y = params["origin_y"]
        self.average_velocity = params["average_velocity"]

        self._obstacle_grid: Optional[np.ndarray] = None

    def get_obstacle_grid(self) -> np.ndarray:
        """Return cached obstacle grid, building it on first call."""
        if self._obstacle_grid is not None:
            return self._obstacle_grid

        img_w, img_h = self.map_img.size
        scale = self.map_resolution / self.Planning_resolution
        grid_height = int(img_h * scale)
        grid_width = int(img_w * scale)
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        for row in range(grid_height):
            for col in range(grid_width):
                px = int((col + 0.5) * img_w / grid_width)
                py = int((row + 0.5) * img_h / grid_height)
                if self.map_img.getpixel((px, py)) < (self.threshold * 255):
                    grid[row, col] = 1

        self._obstacle_grid = grid
        return grid

    def is_point_valid(self, point: Tuple[int, int]) -> bool:
        grid = self.get_obstacle_grid()
        h, w = point
        if 0 <= h < grid.shape[0] and 0 <= w < grid.shape[1]:
            return grid[h, w] == 0
        return False

    def get_plan(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Return (found: bool, path: list[(row,col)]) via A*."""
        grid = self.get_obstacle_grid()
        return ut.astar(grid, start, end)


# =============================================================================
# MultiAgentTaskEnv
# =============================================================================

class MultiAgentTaskEnv(gym.Env):
    """
    Multi-agent task allocation environment.
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
        max_wait_delay_s: float = 600.0,
        max_travel_delay_s: float = 3600.0,
        max_steps: int = 2000,
        two_hop: bool = False,
        two_hop_directed: bool = False,
        vicinity_m: float = 100.0,
        movement_speed: float = 1.0,
        decision_interval: int = 8,
        radius: int = 100,
        feature_size: int = 9,
        use_true_id: bool = False,
        reward_mode: str = "new",
        capacity_method: str = "assigned"
    ):
        super().__init__()

        if agents is not None and tasks_batches is not None:
            self.init_mode = "new"
            self.agents_data = agents
            self.tasks_batches = tasks_batches
            self.num_robots = len(agents)
            self.planner = Planner()
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
        self.max_travel_delay_s = max_travel_delay_s

        self.robots = {}
        self.tasks = {}
        self.current_time = 0.0
        self.current_step = 0
        self.total_task_count = 0

        self.episode_completed_count = 0
        self.episode_obsolete_count = 0
        self.episode_pickup_count = 0
        self.episode_dropoff_count = 0
        self._prev_completed_count = 0
        self._prev_obsolete_count = 0
        self._prev_pickup_count = 0
        self._prev_dropoff_count = 0

        # diagnostics (episode-level)
        self.debug_invalid_action_count = 0
        self.debug_total_action_count = 0
        self.debug_valid_action_count = 0
        self.debug_conflict_dropped_count = 0
        self.debug_capacity_rejected_count = 0

        # diagnostics (last-step)
        self.debug_last_invalid_action_count = 0
        self.debug_last_total_action_count = 0
        self.debug_last_valid_action_count = 0
        self.debug_last_conflict_dropped_count = 0
        self.debug_last_capacity_rejected_count = 0
        self.debug_last_mask_zero_count = 0

        self.debug_last_r_comp = 0.0
        self.debug_last_r_wait = 0.0
        self.debug_last_r_deadline = 0.0
        self.debug_last_r_obsolete = 0.0

        self.debug_ep_r_comp = 0.0
        self.debug_ep_r_wait = 0.0
        self.debug_ep_r_deadline = 0.0
        self.debug_ep_r_obsolete = 0.0

        if self.init_mode == "new":
            self.max_position = max(np.max(agents[:, 1]), np.max(agents[:, 2]))
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
            "x": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_robots, N_max, self.F), dtype=np.float32),
            "node_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_robots, N_max), dtype=np.uint8),
            "edge_index": gym.spaces.Box(low=0, high=N_max, shape=(self.num_robots, 2, E_max), dtype=np.int64),
            "edge_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_robots, E_max), dtype=np.uint8),
            "cand_idx": gym.spaces.Box(low=0, high=N_max, shape=(self.num_robots, K_max), dtype=np.int64),
            "cand_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_robots, K_max), dtype=np.uint8),
        })

        if self.edge_feat_dim > 0:
            self.observation_space.spaces["edge_attr"] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_robots, E_max, self.edge_feat_dim), dtype=np.float32
            )

        self.action_space = gym.spaces.MultiDiscrete([K_max + 1] * self.num_robots)
        self._noop_index = K_max
        self._last_cand_task_ids = [[] for _ in range(self.num_robots)]

        self.capacity_method = capacity_method.lower()
        if self.capacity_method not in ("assigned", "pickup"):
            raise ValueError(
                "capacity_method must be 'assigned' or 'pickup'"
            )
    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

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

        self.debug_invalid_action_count = 0
        self.debug_total_action_count = 0
        self.debug_valid_action_count = 0
        self.debug_conflict_dropped_count = 0
        self.debug_capacity_rejected_count = 0

        self.debug_last_invalid_action_count = 0
        self.debug_last_total_action_count = 0
        self.debug_last_valid_action_count = 0
        self.debug_last_conflict_dropped_count = 0
        self.debug_last_capacity_rejected_count = 0
        self.debug_last_mask_zero_count = 0

        self.debug_last_r_comp = 0.0
        self.debug_last_r_wait = 0.0
        self.debug_last_r_deadline = 0.0
        self.debug_last_r_obsolete = 0.0

        self.debug_ep_r_comp = 0.0
        self.debug_ep_r_wait = 0.0
        self.debug_ep_r_deadline = 0.0
        self.debug_ep_r_obsolete = 0.0

        if self.init_mode == "new":
            self._reset_new_mode()
        else:
            self._reset_old_mode()

        obs = self._build_observation()
        return obs, {"action_mask": self.action_mask()}

    def _reset_new_mode(self):
        self.robots = {}
        for agent in self.agents_data:
            robot_id = str(int(agent[0]))
            self.robots[robot_id] = {
                "id": robot_id,
                "x": float(agent[1]),
                "y": float(agent[2]),
                "max_capacity": self.max_robot_capacity,
                "current_capacity": 0,          # == len(onboard_tasks), kept for back-compat
                "assigned_tasks": [],           # task_ids assigned, not yet picked up
                "onboard_tasks": [],            # task_ids picked up, not yet dropped off
                "current_stop": None,           # {"task_id":, "kind": "pickup"|"dropoff"} or None
                "target_location": None,
                "path": [],
                "just_picked_up_task": None,
            }

        self.total_task_count = sum(len(b) for b in self.tasks_batches)
        self.tasks = {}
        self._release_pending_tasks()

    def _reset_old_mode(self):
        pass

    # =========================================================================
    # TASK RELEASE
    # =========================================================================

    def _release_pending_tasks(self):
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

    # =========================================================================
    # STEP
    # =========================================================================
    def _debug_robot_state(self):
        print("\n================ ROBOT STATE ================")
        print(f"time={self.current_time:.1f} step={self.current_step}")

        for robot_id in sorted(self.robots.keys()):
            r = self.robots[robot_id]

            print(
                f"Robot {robot_id} | "
                f"cap={r['current_capacity']}/{r['max_capacity']} | "
                f"onboard={r['onboard_tasks']} | "
                f"stop={r['current_stop']} | "
                f"queue={r['assigned_tasks']}"
            )

            # onboard tasks
            for tid in r["onboard_tasks"]:
                t = self.tasks[tid]
                print(
                    f"   ONBOARD {tid}: "
                    f"picked={t['is_picked_up']} "
                    f"completed={t['is_completed']} "
                    f"obsolete={t['is_obsolete']}"
                )

            # queued tasks
            for tid in r["assigned_tasks"]:
                t = self.tasks[tid]
                print(
                    f"   QUEUED {tid}: "
                    f"assigned={t['is_assigned']} "
                    f"picked={t['is_picked_up']} "
                    f"completed={t['is_completed']} "
                    f"obsolete={t['is_obsolete']}"
                )

        print("=============================================\n")
    def step(self, actions):
        # print(f"Step {self.current_step}: actions={actions}")
        action_info = self._process_actions(actions)

        # macro-step component accumulators
        macro_r_comp = 0.0
        macro_r_wait = 0.0
        macro_r_deadline = 0.0
        macro_r_obsolete = 0.0

        self._release_pending_tasks()
        self._update_task_deadlines()
        self._execute_robot_movements_and_tasks()
        self.current_time += 1.0
        self.current_step += 1
        reward = self._compute_rewards(action_info)

        macro_r_comp += self.debug_last_r_comp
        macro_r_wait += self.debug_last_r_wait
        macro_r_deadline += self.debug_last_r_deadline
        macro_r_obsolete += self.debug_last_r_obsolete

        for _ in range(self.decision_interval - 1):
            if self.current_step >= self.max_steps:
                break
            self._release_pending_tasks()
            self._update_task_deadlines()
            self._execute_robot_movements_and_tasks()
            self.current_time += 1.0
            self.current_step += 1
            reward += self._compute_rewards({})

            macro_r_comp += self.debug_last_r_comp
            macro_r_wait += self.debug_last_r_wait
            macro_r_deadline += self.debug_last_r_deadline
            macro_r_obsolete += self.debug_last_r_obsolete

            if self._check_episode_done():
                break

        terminated = self._check_episode_done()
        truncated = self.current_step >= self.max_steps
        # self._debug_robot_state()
        
        obs = self._build_observation()
        mask = self.action_mask()

        info = {
            "action_mask": mask,
            "step": self.current_step,
            "time": self.current_time,
            "completed_count": self.episode_completed_count,
            "obsolete_count": self.episode_obsolete_count,
            "pickup_count": self.episode_pickup_count,
            "dropoff_count": self.episode_dropoff_count,

            "invalid_action_count": self.debug_last_invalid_action_count,
            "total_action_count": self.debug_last_total_action_count,
            "valid_action_count": self.debug_last_valid_action_count,
            "conflict_dropped_count": self.debug_last_conflict_dropped_count,
            "capacity_rejected_count": self.debug_last_capacity_rejected_count,
            "mask_zero_count": self.debug_last_mask_zero_count,

            # IMPORTANT: macro-step sums (not last micro-step)
            "r_comp": float(macro_r_comp),
            "r_wait": float(macro_r_wait),
            "r_deadline": float(macro_r_deadline),
            "r_obsolete": float(macro_r_obsolete),

            "ep_r_comp": self.debug_ep_r_comp,
            "ep_r_wait": self.debug_ep_r_wait,
            "ep_r_deadline": self.debug_ep_r_deadline,
            "ep_r_obsolete": self.debug_ep_r_obsolete,
        }
        # print(f'completed_count:=',self.episode_completed_count, 'obsolete_count:=',self.episode_obsolete_count, 'pickup_count:=',
        #       self.episode_pickup_count, 'dropoff_count:=',self.episode_dropoff_count)
        return obs, reward, terminated, truncated, info
   
    # =========================================================================
    # ACTION PROCESSING
    # =========================================================================

    def _process_actions(self, actions) -> Dict:
        """
        Uses _last_cand_task_ids from observation-time snapshot to prevent
        index mismatch between policy output and candidate list.
        """
        robot_ids = sorted(self.robots.keys())
        requests = []

        invalid_action_count = 0
        conflict_dropped_count = 0
        total_action_count = 0
        capacity_rejected_count = 0

        act_arr = np.asarray(actions).flatten()

        # for r_idx, action in enumerate(act_arr):
        #     if r_idx >= len(robot_ids):
        #         break

        #     total_action_count += 1
        #     a = int(action)

        #     if a == self._noop_index:
        #         continue

        #     robot_id = robot_ids[r_idx]
        #     cands = self._last_cand_task_ids[r_idx] if r_idx < len(self._last_cand_task_ids) else []

        #     if a < 0 or a >= len(cands):
        #         invalid_action_count += 1
        #         continue

        #     task_id = cands[a]
        for r_idx, action in enumerate(actions):
            # total_action_count += 1
            if int(action) == self._noop_index:
                continue
            if r_idx >= len(robot_ids):
                break
            robot_id = robot_ids[r_idx]
            cands = self._last_cand_task_ids[r_idx]     # <-- use cached list, not recomputed
            if int(action) >= len(cands) or cands[int(action)] is None:
                continue
            task_id = cands[int(action)]
            task = self.tasks.get(task_id)

            if task is None or task.get("is_assigned") or task.get("is_obsolete") or task.get("is_completed"):
                invalid_action_count += 1
                continue

            robot = self.robots[robot_id]
            dist = np.sqrt(
                (robot["x"] - task["pickup_x"]) ** 2 +
                (robot["y"] - task["pickup_y"]) ** 2
            )
            requests.append((dist, robot_id, task_id))

        requests.sort()
        assigned_this_step = set()
        action_info = {}

        for _dist, robot_id, task_id in requests:
            if task_id in assigned_this_step:
                conflict_dropped_count += 1
                continue

            robot = self.robots[robot_id]
            task = self.tasks.get(task_id)
            if task is None or task.get("is_assigned") or task.get("is_obsolete") or task.get("is_completed"):
                invalid_action_count += 1
                continue

            # capacity_method="assigned": count onboard + queued-not-yet-picked
            #   (conservative — reserves a seat for every task promised, even
            #   ones not yet physically onboard)
            # capacity_method="pickup": count onboard only (len(onboard_tasks))
            #   (permissive — allows queuing more pickups than max_capacity as
            #   long as physical onboard load never exceeds it; relies on
            #   _assign_next_stop's room_to_pickup check to enforce the real
            #   physical limit at pickup time)
            if self.capacity_method == "assigned":
                total_committed = (
                    len(robot["onboard_tasks"])
                    + len(robot["assigned_tasks"])
                )
            else:
                total_committed = len(robot["onboard_tasks"])

            if total_committed >= self.max_robot_capacity:
                capacity_rejected_count += 1
                continue

            robot["assigned_tasks"].append(task_id)
            task["is_assigned"] = True
            task["assigned_robot"] = robot_id
            assigned_this_step.add(task_id)
            action_info[robot_id] = {"assigned_task": task_id}

        valid_action_count = len(action_info)

        self.debug_last_invalid_action_count = int(invalid_action_count)
        self.debug_last_total_action_count = int(total_action_count)
        self.debug_last_valid_action_count = int(valid_action_count)
        self.debug_last_conflict_dropped_count = int(conflict_dropped_count)
        self.debug_last_capacity_rejected_count = int(capacity_rejected_count)

        self.debug_invalid_action_count += int(invalid_action_count)
        self.debug_total_action_count += int(total_action_count)
        self.debug_valid_action_count += int(valid_action_count)
        self.debug_conflict_dropped_count += int(conflict_dropped_count)
        self.debug_capacity_rejected_count += int(capacity_rejected_count)

        action_info["_diag"] = {
            "invalid_action_count": int(invalid_action_count),
            "total_action_count": int(total_action_count),
            "valid_action_count": int(valid_action_count),
            "conflict_dropped_count": int(conflict_dropped_count),
            "capacity_rejected_count": int(capacity_rejected_count),
        }
        # print(f"Step {self.current_step}: invalid={invalid_action_count}, total={total_action_count}, valid={valid_action_count}, conflict_dropped={conflict_dropped_count}, capacity_rejected={capacity_rejected_count}")
        return action_info

    # =========================================================================
    # CANDIDATE TASKS
    # =========================================================================
# src/environment/environment.py

    def _remaining_capacity(self, robot_id) -> int:
        """Free 'seats' on this robot right now.
        capacity_method='assigned': onboard + queued-not-yet-picked both count
            (conservative — matches candidate gating with _process_actions).
        capacity_method='pickup': onboard only counts (permissive)."""
        robot = self.robots.get(str(robot_id))
        if robot is None:
            return 0

        if self.capacity_method == "assigned":
            committed = len(robot["onboard_tasks"]) + len(robot["assigned_tasks"])
        else:   # pickup
            committed = len(robot["onboard_tasks"])

        return max(0, robot["max_capacity"] - committed)


    def _get_candidate_tasks(self, robot_id) -> List[str]:
        """Return up to K_max available tasks within vicinity_m of the robot,
        sorted by ascending Euclidean distance to pickup location.
        Gated by the robot's own remaining capacity — a full robot gets an
        empty candidate list (forced no-op), matching the reference adapter.
        """
        assigned = 0
        completed = 0
        obsolete = 0
        future = 0
        deadline = 0
        far = 0
        accepted = 0
        robot = self.robots.get(str(robot_id))
        if robot is None:
            return []

        if self._remaining_capacity(robot_id) <= 0:
            return []

        candidates = []
        for task_id, task in self.tasks.items():
            if task["is_assigned"]:
                assigned += 1
                

            if task["is_completed"]:
                completed += 1
                

            if task["is_obsolete"]:
                obsolete += 1
                

            if task["release_time"] > self.current_time:
                future += 1
                continue

            if task["pickup_deadline"] <= self.current_time:
                deadline += 1
                continue
            dist = np.sqrt(
                (robot["x"] - task["pickup_x"]) ** 2 +
                (robot["y"] - task["pickup_y"]) ** 2
            )
            if task["is_assigned"] or task["is_completed"] or task["is_obsolete"]:
                continue
            if dist > self.vicinity_m:
                far += 1
                continue
            if dist <= self.vicinity_m:
                accepted += 1
                candidates.append((dist, task_id))

        candidates.sort()
        # print(
        #     robot_id,
        #     "accepted", accepted,
        #     "assigned", assigned,s
        #     "completed", completed,
        #     "obsolete", obsolete,
        #     "future", future,
        #     "deadline", deadline,
        #     "far", far,
        #     "candidates", len(candidates)
        # )
        return [tid for _, tid in candidates[: self.K_max]]
    def _get_candidate_tasks_no_capacity_check(self, robot_id) -> List[str]:
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
            dist = np.sqrt(
                (robot["x"] - task["pickup_x"]) ** 2 +
                (robot["y"] - task["pickup_y"]) ** 2
            )
            if dist <= self.vicinity_m:
                candidates.append((dist, task_id))

        candidates.sort()
        return [tid for _, tid in candidates[: self.K_max]]

    # =========================================================================
    # TASK LIFECYCLE — deadlines
    # =========================================================================

    def _update_task_deadlines(self):
        """
        Deadline handling policy:
        - Not picked up yet: pickup deadline expiry => obsolete.
        - Already picked up: NEVER obsolete; keep delivery, penalize lateness in reward.
        """
        for task_id, task in list(self.tasks.items()):
            if task.get("is_completed") or task.get("is_obsolete"):
                continue

            if not task.get("is_picked_up"):
                expired_pickup = task.get("pickup_deadline", float("inf")) <= self.current_time
                if not expired_pickup:
                    continue

                task["is_obsolete"] = True
                self.episode_obsolete_count += 1

                assigned_id = task.get("assigned_robot")
                if assigned_id and assigned_id in self.robots:
                    robot = self.robots[assigned_id]

                    if task_id in robot["assigned_tasks"]:
                        robot["assigned_tasks"].remove(task_id)

                    stop = robot["current_stop"]
                    if stop is not None and stop["task_id"] == task_id and stop["kind"] == "pickup":
                        robot["current_stop"]    = None
                        robot["target_location"] = None
                        robot["path"]            = []

            else:
                # Picked up tasks are kept alive; lateness handled at dropoff reward.
                pass

    # =========================================================================
    # ROBOT MOVEMENT — A* path following
    # =========================================================================

    def _execute_robot_movements_and_tasks(self):
        for robot_id, robot in self.robots.items():
            if robot["current_stop"] is None:
                self._assign_next_stop(robot)

            if robot["current_stop"] is not None:
                self._move_robot_toward_target(robot_id)

    def _assign_next_stop(self, robot):
        """
        Nearest-stop routing policy for multi-capacity robots.

        Candidate next stops:
          - a pickup for any task in assigned_tasks, but only if the robot
            currently has room to carry another (len(onboard_tasks) < max_capacity)
          - a dropoff for any task in onboard_tasks

        The nearest candidate (Euclidean distance from current position) is
        chosen, letting a robot interleave pickups and dropoffs instead of
        finishing one task before starting the next.
        """
        candidates = []  # (dist, kind, task_id, location)

        room_to_pickup = len(robot["onboard_tasks"]) < robot["max_capacity"]
        if room_to_pickup:
            for task_id in robot["assigned_tasks"]:
                task = self.tasks.get(task_id)
                if task is None or task.get("is_obsolete"):
                    continue
                loc  = (task["pickup_x"], task["pickup_y"])
                dist = np.sqrt((robot["x"] - loc[0]) ** 2 + (robot["y"] - loc[1]) ** 2)
                candidates.append((dist, "pickup", task_id, loc))

        for task_id in robot["onboard_tasks"]:
            task = self.tasks.get(task_id)
            if task is None:
                continue
            loc  = (task["dropoff_x"], task["dropoff_y"])
            dist = np.sqrt((robot["x"] - loc[0]) ** 2 + (robot["y"] - loc[1]) ** 2)
            candidates.append((dist, "dropoff", task_id, loc))

        if not candidates:
            robot["current_stop"]    = None
            robot["target_location"] = None
            robot["path"]            = []
            return

        candidates.sort(key=lambda c: c[0])
        _, kind, task_id, loc = candidates[0]
        robot["current_stop"]    = {"task_id": task_id, "kind": kind}
        robot["target_location"] = loc
        robot["path"]            = []

    def _move_robot_toward_target(self, robot_id: str):
        robot = self.robots[robot_id]
        target_x, target_y = robot["target_location"]

        if not robot["path"]:
            start = (int(round(robot["y"])), int(round(robot["x"])))
            goal = (int(round(target_y)), int(round(target_x)))
            if start != goal:
                found, path = self.planner.get_plan(start, goal)
                if found and path and len(path) > 1:
                    robot["path"] = list(path[1:])
                else:
                    robot["path"] = []

        if robot["path"]:
            next_row, next_col = robot["path"][0]
            dx = float(next_col) - robot["x"]
            dy = float(next_row) - robot["y"]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist <= self.movement_speed:
                robot["x"] = float(next_col)
                robot["y"] = float(next_row)
                robot["path"].pop(0)
            else:
                robot["x"] += (dx / dist) * self.movement_speed
                robot["y"] += (dy / dist) * self.movement_speed
            return

        dx = target_x - robot["x"]
        dy = target_y - robot["y"]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > 0.5:
            move = min(self.movement_speed, dist)
            robot["x"] += (dx / dist) * move
            robot["y"] += (dy / dist) * move
            return

        # ── Arrival ───────────────────────────────────────────────────────
        stop = robot["current_stop"]
        if stop is None:
            return
        task_id = stop["task_id"]
        task    = self.tasks.get(task_id)
        if task is None:
            robot["current_stop"]    = None
            robot["target_location"] = None
            robot["path"]            = []
            return

        if stop["kind"] == "pickup":
            if task_id in robot["assigned_tasks"]:
                robot["assigned_tasks"].remove(task_id)
            robot["onboard_tasks"].append(task_id)
            robot["current_capacity"]  = len(robot["onboard_tasks"])
            task["is_picked_up"]       = True
            self.episode_pickup_count += 1
            task["pickup_time"]        = self.current_time
            robot["just_picked_up_task"] = task_id
            # Stop is cleared; _assign_next_stop() picks the next pickup or
            # dropoff (whichever is nearest) next tick — this is what lets
            # onboard_tasks hold more than one task at a time.
            robot["current_stop"]      = None
            robot["target_location"]   = None
            robot["path"]              = []

        elif stop["kind"] == "dropoff":
            if task_id in robot["onboard_tasks"]:
                robot["onboard_tasks"].remove(task_id)
            robot["current_capacity"]     = len(robot["onboard_tasks"])
            task["dropoff_time"]          = self.current_time
            task["is_completed"]          = True
            self.episode_dropoff_count   += 1
            self.episode_completed_count += 1
            robot["current_stop"]         = None
            robot["target_location"]      = None
            robot["path"]                 = []

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    def _build_observation(self) -> Dict:
        robot_ids = sorted(self.robots.keys())
        if len(robot_ids) < self.num_robots:
            robot_ids += [None] * (self.num_robots - len(robot_ids))
        robot_ids = robot_ids[: self.num_robots]

        candidate_lists = [
            self._get_candidate_tasks(rid) if rid is not None else []
            for rid in robot_ids
        ]

        obs_dict, cand_task_ids = build_padded_ego_batch(
            robots=robot_ids,
            robots_dict=self.robots,
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

    def _compute_rewards(self, action_info) -> float:
        W_COMP = 2.0
        W_WAIT = 1
        W_DEADLINE = 10.0
        W_OBS = 0.5

        WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
        DEADLINE_CAP = max(1.0, float(self.max_travel_delay_s))

        reward = 0.0
        r_comp = 0.0
        r_wait = 0.0
        r_deadline = 0.0
        r_obsolete = 0.0

        # 1) pickup wait penalty
        for r in self.robots.values():
            task_id = r.get("just_picked_up_task")
            if not task_id:
                continue
            task = self.tasks.get(task_id)
            if task is None:
                r["just_picked_up_task"] = None
                continue

            wait = max(0.0, self.current_time - task["release_time"])
            delta = W_WAIT * (-min(wait, WAIT_CAP) / WAIT_CAP)
            reward += delta
            r_wait += delta
            r["just_picked_up_task"] = None

        # 2) completion + lateness penalties
        for task in self.tasks.values():
            if not task.get("is_completed"):
                continue
            if task.get("_rewarded"):
                continue
            task["_rewarded"] = True

            reward += W_COMP
            r_comp += W_COMP

            pickup_time = task.get("pickup_time", self.current_time)
            dropoff_time = task.get("dropoff_time", self.current_time)

            if task.get("pickup_deadline") is not None:
                late_p = max(0.0, pickup_time - task["pickup_deadline"])
                delta = W_DEADLINE * (-min(late_p, DEADLINE_CAP) / DEADLINE_CAP)
                reward += delta
                r_deadline += delta

            if task.get("dropoff_deadline") is not None:
                late_d = max(0.0, dropoff_time - task["dropoff_deadline"])
                delta = W_DEADLINE * (-min(late_d, DEADLINE_CAP) / DEADLINE_CAP)
                reward += delta
                r_deadline += delta

        # 3) obsolete penalties (only not-picked tasks become obsolete by design)
        for task in self.tasks.values():
            if not task.get("is_obsolete"):
                continue
            if task.get("_obsolete_rewarded"):
                continue
            task["_obsolete_rewarded"] = True

            delta_obs = -W_OBS
            reward += delta_obs
            r_obsolete += delta_obs
            # print(r_obsolete,'r_obsolete')
            late = max(0.0, self.current_time - task.get("pickup_deadline", self.current_time))
            delta_dead = W_DEADLINE * (-min(late, DEADLINE_CAP) / DEADLINE_CAP)
            reward += delta_dead
            r_deadline += delta_dead

        self.debug_last_r_comp = float(r_comp)
        self.debug_last_r_wait = float(r_wait)
        self.debug_last_r_deadline = float(r_deadline)
        self.debug_last_r_obsolete = float(r_obsolete)

        self.debug_ep_r_comp += float(r_comp)
        self.debug_ep_r_wait += float(r_wait)
        self.debug_ep_r_deadline += float(r_deadline)
        self.debug_ep_r_obsolete += float(r_obsolete)
        # print(self.debug_ep_r_obsolete,'debug_ep_r_obsolete')
        return float(reward)

    # =========================================================================
    # TERMINATION
    # =========================================================================

    def _check_episode_done(self) -> bool:
        if len(self.tasks) < self.total_task_count:
            return False

        any_pending = any(
            not t.get("is_completed") and not t.get("is_obsolete")
            for t in self.tasks.values()
        )
        if any_pending:
            return False

        robots_idle = all(
            len(r["assigned_tasks"]) == 0 and len(r["onboard_tasks"]) == 0
            for r in self.robots.values()
        )
        return robots_idle

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_robots, self.K_max + 1), dtype=np.uint8)
        for r in range(self.num_robots):
            cand_list = self._last_cand_task_ids[r] if r < len(self._last_cand_task_ids) else []
            for k in range(min(self.K_max, len(cand_list))):
                if cand_list[k] is not None:
                    mask[r, k] = 1
            mask[r, self._noop_index] = 1

        self.debug_last_mask_zero_count = int(np.sum(mask[:, :self._noop_index] == 0))
        return mask

    def close(self):
        pass