"""
Comprehensive multi-agent task allocation environment with full simulator logic.

Features:
- Robot movement & navigation via A* path planning
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
        root_path   = Path(__file__).resolve().parent.parent.parent / "env"
        config_path = root_path / "ATC_wed.yaml"
        with open(config_path, "r") as fh:
            params = yaml.safe_load(fh)

        map_path = root_path / params["map_filename"]
        self.map_img           = Image.open(map_path).convert("L")
        self.map_resolution    = params["map_resolution"]
        self.Planning_resolution = params["Planning_resolution"]
        self.threshold         = params["obstacle_threshold"]
        self.origin_x          = params["origin_x"]
        self.origin_y          = params["origin_y"]
        self.average_velocity  = params["average_velocity"]

        # Cache the obstacle grid once — rebuilding it on every call was O(W*H)
        self._obstacle_grid: Optional[np.ndarray] = None

    def get_obstacle_grid(self) -> np.ndarray:
        """Return cached obstacle grid, building it on first call."""
        if self._obstacle_grid is not None:
            return self._obstacle_grid

        img_w, img_h = self.map_img.size
        scale        = self.map_resolution / self.Planning_resolution
        grid_height  = int(img_h * scale)
        grid_width   = int(img_w * scale)
        grid         = np.zeros((grid_height, grid_width), dtype=np.uint8)

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

    Coordinate system
    -----------------
    All positions (robot x/y, task pickup/dropoff) are stored as GRID INDICES
    (col, row) corresponding to the planning grid defined in ATC_wed.yaml
    with Planning_resolution=1.  movement_speed=1 therefore means 1 grid cell
    per simulation tick.  The A* planner also operates in (row, col) grid
    index space, so the conversion is:

        planner_start = (int(robot["y"]), int(robot["x"]))   # (row, col)
        planner_goal  = (int(target_y),  int(target_x))

    A* paths are stored as lists of (row, col) tuples; the robot follows them
    one cell at a time.

    Episode semantics
    -----------------
    One episode covers ALL task batches.  Tasks are released gradually as
    current_time reaches each batch's release_time.  The episode terminates
    naturally when every task from every batch has been either completed or
    made obsolete and all robots are idle.  It is truncated if current_step
    reaches max_steps.
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
    ):
        super().__init__()

        if agents is not None and tasks_batches is not None:
            self.init_mode    = "new"
            self.agents_data  = agents
            self.tasks_batches = tasks_batches
            self.num_robots   = len(agents)
            # Planner is needed in new mode too (for A* path following).
            self.planner      = Planner()
        elif agents_cont_coord_array is not None and task_cont_coord_array is not None:
            self.init_mode                  = "old"
            self.agents_cont_coord_array    = agents_cont_coord_array
            self.task_cont_coord_array      = task_cont_coord_array
            self.num_robots                 = len(agents_cont_coord_array)
            self.radius                     = radius
            self.feature_size               = feature_size
            self.use_true_id                = use_true_id
            self.reward_mode                = reward_mode
            self.planner                    = Planner()
        else:
            raise ValueError(
                "Must provide either (agents, tasks_batches) or "
                "(agents_cont_coord_array, task_cont_coord_array)"
            )

        self.N_max              = N_max
        self.E_max              = E_max
        self.K_max              = K_max
        self.max_robot_capacity = max_robot_capacity
        self.vicinity_m         = vicinity_m
        self.two_hop            = two_hop
        self.two_hop_directed   = two_hop_directed
        self.max_steps          = max_steps
        self.movement_speed     = movement_speed
        self.decision_interval  = decision_interval
        self.max_wait_delay_s   = max_wait_delay_s

        self.robots       = {}
        self.tasks        = {}
        self.current_time = 0.0
        self.current_step = 0
        # Total tasks across ALL batches — used by _check_episode_done.
        self.total_task_count = 0

        self.episode_completed_count = 0
        self.episode_obsolete_count  = 0
        self.episode_pickup_count    = 0
        self.episode_dropoff_count   = 0
        self._prev_completed_count   = 0
        self._prev_obsolete_count    = 0
        self._prev_pickup_count      = 0
        self._prev_dropoff_count     = 0

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
            self.edge_features  = edge_features or ["dx", "dy", "eta"]
            self.edge_feat_dim  = len(self.edge_features)
        else:
            self.edge_features  = []
            self.edge_feat_dim  = 0

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

        self.action_space          = gym.spaces.MultiDiscrete([K_max + 1] * self.num_robots)
        self._noop_index           = K_max
        self._last_cand_task_ids   = [[] for _ in range(self.num_robots)]

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self, seed=None):
        """Reset environment for a new episode covering ALL task batches."""
        super().reset(seed=seed)

        self.current_time            = 0.0
        self.current_step            = 0
        self.episode_completed_count = 0
        self.episode_obsolete_count  = 0
        self.episode_pickup_count    = 0
        self.episode_dropoff_count   = 0
        self._prev_completed_count   = 0
        self._prev_obsolete_count    = 0
        self._prev_pickup_count      = 0
        self._prev_dropoff_count     = 0

        if self.init_mode == "new":
            self._reset_new_mode()
        else:
            self._reset_old_mode()

        obs = self._build_observation()
        return obs, {"action_mask": self.action_mask()}

    def _reset_new_mode(self):
        """
        Initialise robots at their starting positions and seed the task pool.

        Every episode uses ALL batches: tasks are NOT loaded upfront but are
        injected gradually by _release_pending_tasks() as current_time reaches
        each batch's release_time.  Batch 0 has release_time=0 so those tasks
        are available immediately; later batches appear at t=15, 30, …
        """
        # ── Robots ────────────────────────────────────────────────────────────
        self.robots = {}
        for agent in self.agents_data:
            robot_id = str(int(agent[0]))
            self.robots[robot_id] = {
                "id":               robot_id,
                "x":                float(agent[1]),   # grid col index
                "y":                float(agent[2]),   # grid row index
                "max_capacity":     self.max_robot_capacity,
                "current_capacity": 0,                 # tasks physically onboard
                "assigned_tasks":   [],                # queue: assigned but not current
                "current_task":     None,              # task currently being executed
                "task_phase":       None,              # "pickup" | "travel_to_dropoff"
                "target_location":  None,              # (x, y) grid coords of next waypoint
                "path":             [],                # A* path: list of (row, col) cells
                "just_picked_up_task": None,           # set at pickup for reward shaping
            }

        # ── Tasks ─────────────────────────────────────────────────────────────
        # Precompute total task count once per episode for _check_episode_done.
        self.total_task_count = sum(len(b) for b in self.tasks_batches)
        self.tasks = {}
        # Seed tasks whose release_time <= 0 (i.e. batch 0 with release_time=0).
        self._release_pending_tasks()

    def _reset_old_mode(self):
        """Reset for legacy mode (not used in current training)."""
        pass

    # =========================================================================
    # TASK RELEASE
    # =========================================================================

    def _release_pending_tasks(self):
        """
        Inject tasks from ALL batches whose release_time <= current_time.

        Called once in reset() (seeds t=0 tasks) and once per simulation tick
        inside step() so later batches enter the pool at the right moment.
        """
        for batch in self.tasks_batches:
            for task_data in batch:
                task_id = str(int(task_data[0]))
                if task_id in self.tasks:
                    continue
                if float(task_data[5]) <= self.current_time:
                    self.tasks[task_id] = {
                        "id":              task_id,
                        "pickup_x":        float(task_data[1]),
                        "pickup_y":        float(task_data[2]),
                        "dropoff_x":       float(task_data[3]),
                        "dropoff_y":       float(task_data[4]),
                        "release_time":    float(task_data[5]),
                        "pickup_deadline": float(task_data[6]),
                        "est_travel_time": float(task_data[7]),
                        "dropoff_deadline":float(task_data[8]),
                        "is_assigned":     False,
                        "is_obsolete":     False,
                        "is_picked_up":    False,
                        "is_completed":    False,
                        "assigned_robot":  None,
                    }

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, actions):
        """
        Execute one policy step covering decision_interval simulation ticks.

        Tick order per simulation tick:
          1. Release newly-due tasks (_release_pending_tasks)
          2. Expire overdue tasks    (_update_task_deadlines)
          3. Move robots / pickup / dropoff (_execute_robot_movements_and_tasks)
          4. Advance time
          5. Compute incremental reward
        """
        # Outer decision tick (first of the decision_interval ticks)
        action_info = self._process_actions(actions)
        self._release_pending_tasks()
        self._update_task_deadlines()
        self._execute_robot_movements_and_tasks()
        self.current_time  += 1.0
        self.current_step  += 1
        reward = self._compute_rewards(action_info)

        # Inner no-op ticks (robots keep moving, no new policy decisions)
        for _ in range(self.decision_interval - 1):
            if self.current_step >= self.max_steps:
                break
            self._release_pending_tasks()
            self._update_task_deadlines()
            self._execute_robot_movements_and_tasks()
            self.current_time  += 1.0
            self.current_step  += 1
            reward += self._compute_rewards({})
            if self._check_episode_done():
                break

        terminated = self._check_episode_done()
        truncated  = self.current_step >= self.max_steps
        # print(f"Step {self.current_step}: reward={reward:.3f}, completed={self.episode_completed_count}, obsolete={self.episode_obsolete_count}, pickup={self.episode_pickup_count}, dropoff={self.episode_dropoff_count}, batches={len(self.tasks_batches)}, tasks={len(self.tasks)}")
        obs  = self._build_observation()
        info = {
            "action_mask":     self.action_mask(),
            "step":            self.current_step,
            "time":            self.current_time,
            "completed_count": self.episode_completed_count,
            "obsolete_count":  self.episode_obsolete_count,
            "pickup_count":    self.episode_pickup_count,
            "dropoff_count":   self.episode_dropoff_count,
        }
        return obs, reward, terminated, truncated, info

    # =========================================================================
    # ACTION PROCESSING
    # =========================================================================

    def _process_actions(self, actions) -> Dict:
        """
        Map policy actions to task assignments with conflict resolution.

        Capacity accounting
        -------------------
        A robot's committed capacity = tasks physically onboard
        (current_capacity) + tasks queued but not yet started (assigned_tasks)
        + the task it is currently travelling to pick up (current_task when
        phase == "pickup").  This prevents over-assignment when a robot is
        already en-route to a pickup location but hasn't arrived yet.

        Conflict resolution: if two robots select the same task, the closer
        robot wins (greedy by ascending distance).
        """
        robot_ids = sorted(self.robots.keys())
        requests  = []

        for r_idx, action in enumerate(actions):
            if int(action) == self._noop_index:
                continue
            if r_idx >= len(robot_ids):
                break
            robot_id = robot_ids[r_idx]
            cands    = self._get_candidate_tasks(robot_id)
            if int(action) >= len(cands):
                continue
            task_id = cands[int(action)]
            robot   = self.robots[robot_id]
            dist    = np.sqrt(
                (robot["x"] - self.tasks[task_id]["pickup_x"]) ** 2 +
                (robot["y"] - self.tasks[task_id]["pickup_y"]) ** 2
            )
            requests.append((dist, robot_id, task_id))

        # Closest robot wins any conflict on the same task
        requests.sort()
        assigned_this_step = set()
        action_info        = {}

        for _dist, robot_id, task_id in requests:
            if task_id in assigned_this_step:
                continue  # another robot already claimed this task

            robot = self.robots[robot_id]
            task  = self.tasks.get(task_id)

            # Skip if task disappeared (obsoleted between obs and action)
            if task is None or task.get("is_assigned") or task.get("is_obsolete"):
                continue

            # Total committed capacity:
            #   current_capacity  = physically onboard (picked up, not dropped off)
            #   len(assigned_tasks) = queued but not yet current_task
            #   +1 if current_task is in pickup phase (committed but not onboard yet)
            in_pickup_phase = (
                robot["current_task"] is not None and
                robot["task_phase"] == "pickup"
            )
            total_committed = (
                robot["current_capacity"] +
                len(robot["assigned_tasks"]) +
                (1 if in_pickup_phase else 0)
            )

            if total_committed >= self.max_robot_capacity:
                continue

            # Assign
            robot["assigned_tasks"].append(task_id)
            task["is_assigned"]    = True
            task["assigned_robot"] = robot_id
            assigned_this_step.add(task_id)
            action_info[robot_id] = {"assigned_task": task_id}

        return action_info

    # =========================================================================
    # CANDIDATE TASKS
    # =========================================================================

    def _get_candidate_tasks(self, robot_id) -> List[str]:
        """
        Return up to K_max available tasks within vicinity_m of the robot,
        sorted by ascending Euclidean distance to pickup location.
        """
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
        Mark expired tasks obsolete and clean up the owning robot's state.

        Capacity notes:
          - If the task was already picked up (is_picked_up=True), it
            occupies a slot in current_capacity → decrement.
          - If it was only assigned (not yet picked up), it is still in
            assigned_tasks or is current_task in pickup phase → no
            current_capacity decrement, but must be removed from the queue.
        """
        for task_id, task in list(self.tasks.items()):
            if task.get("is_obsolete") or task.get("is_completed"):
                continue

            # Determine if expired
            if not task.get("is_picked_up"):
                # print(f"Checking task {task_id} for expiration: pickup_deadline={task.get('pickup_deadline')}, current_time={self.current_time}")
                expired = task.get("pickup_deadline", float("inf")) <= self.current_time
            else:
                # print(f"Checking task {task_id} for expiration: dropoff_deadline={task.get('dropoff_deadline')}, current_time={self.current_time}")
                expired = task.get("dropoff_deadline", float("inf")) <= self.current_time

            if not expired:
                continue

            task["is_obsolete"]         = True
            self.episode_obsolete_count += 1

            # Clean up the assigned robot
            assigned_id = task.get("assigned_robot")
            if assigned_id and assigned_id in self.robots:
                robot = self.robots[assigned_id]

                # Decrement onboard count only if physically picked up
                if task.get("is_picked_up"):
                    robot["current_capacity"] = max(0, robot["current_capacity"] - 1)

                # Remove from queue if present
                if task_id in robot["assigned_tasks"]:
                    robot["assigned_tasks"].remove(task_id)

                # Clear current_task if robot was executing this task
                if robot["current_task"] == task_id:
                    robot["current_task"]     = None
                    robot["task_phase"]       = None
                    robot["target_location"]  = None
                    robot["path"]             = []   # discard stale A* path

    # =========================================================================
    # ROBOT MOVEMENT — A* path following
    # =========================================================================

    def _execute_robot_movements_and_tasks(self):
        """
        Advance robot state machines: dequeue tasks, follow A* path,
        execute pickup / dropoff on arrival.
        """
        for robot_id, robot in self.robots.items():
            # Dequeue next task if robot has nothing to do
            if robot["current_task"] is None and robot["assigned_tasks"]:
                next_task_id = robot["assigned_tasks"].pop(0)
                task = self.tasks.get(next_task_id)
                if task is None or task.get("is_obsolete"):
                    continue  # task expired while queued
                robot["current_task"]    = next_task_id
                robot["task_phase"]      = "pickup"
                robot["target_location"] = (task["pickup_x"], task["pickup_y"])
                robot["path"]            = []   # will be planned on first movement

            if robot["current_task"] is not None:
                self._move_robot_toward_target(robot_id)

    def _move_robot_toward_target(self, robot_id: str):
        """
        Move robot one step along its A* path toward the current target.

        Path planning
        -------------
        On the first call after a new target is set (robot["path"] == []),
        A* is invoked to compute a collision-free path from the robot's
        current grid cell to the target cell.  The resulting list of
        (row, col) waypoints is stored on the robot and consumed one cell
        per simulation tick.

        If A* cannot find a path (obstacle-blocked), the robot falls back
        to straight-line movement so it is never permanently stuck.

        Coordinate conventions
        ----------------------
        robot["x"] = grid column   robot["y"] = grid row
        A* takes (row, col) = (int(y), int(x))
        After following a waypoint the robot position is updated to the
        exact (col, row) of that waypoint (integer grid cell).
        """
        robot    = self.robots[robot_id]
        target_x, target_y = robot["target_location"]

        # ── Plan path if we don't have one ────────────────────────────────────
        if not robot["path"]:
            start = (int(round(robot["y"])), int(round(robot["x"])))
            goal  = (int(round(target_y)),   int(round(target_x)))

            if start == goal:
                # Already at target — let the arrival logic below handle it
                pass
            else:
                found, path = self.planner.get_plan(start, goal)
                if found and path and len(path) > 1:
                    # path[0] is the start cell; skip it (robot is already there)
                    robot["path"] = list(path[1:])
                else:
                    # A* failed (e.g. start/goal in obstacle) — straight-line fallback
                    robot["path"] = []

        # ── Follow next waypoint ───────────────────────────────────────────────
        if robot["path"]:
            next_row, next_col = robot["path"][0]

            # Move toward the next cell by up to movement_speed units
            dx   = float(next_col) - robot["x"]
            dy   = float(next_row) - robot["y"]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist <= self.movement_speed:
                # Snap to the waypoint exactly and consume it
                robot["x"] = float(next_col)
                robot["y"] = float(next_row)
                robot["path"].pop(0)
            else:
                robot["x"] += (dx / dist) * self.movement_speed
                robot["y"] += (dy / dist) * self.movement_speed
            return

        # ── No path remaining: check if we have arrived ───────────────────────
        # (handles both the "path exhausted" and "start==goal" cases)
        dx   = target_x - robot["x"]
        dy   = target_y - robot["y"]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > 0.5:
            # Path exhausted but not at target (A* fallback: straight-line nudge)
            move = min(self.movement_speed, dist)
            robot["x"] += (dx / dist) * move
            robot["y"] += (dy / dist) * move
            return

        # ── Arrival ───────────────────────────────────────────────────────────
        task_id = robot["current_task"]
        task    = self.tasks.get(task_id)
        if task is None:
            robot["current_task"]    = None
            robot["task_phase"]      = None
            robot["target_location"] = None
            robot["path"]            = []
            return

        if robot["task_phase"] == "pickup":
            robot["current_capacity"]  += 1
            task["is_picked_up"]        = True
            self.episode_pickup_count  += 1
            robot["just_picked_up_task"] = task_id      # used by reward shaping
            robot["task_phase"]         = "travel_to_dropoff"
            robot["target_location"]    = (task["dropoff_x"], task["dropoff_y"])
            robot["path"]               = []             # plan fresh for dropoff leg

        elif robot["task_phase"] == "travel_to_dropoff":
            robot["current_capacity"]    = max(0, robot["current_capacity"] - 1)
            task["is_completed"]         = True
            self.episode_dropoff_count  += 1
            self.episode_completed_count += 1
            robot["current_task"]        = None
            robot["task_phase"]          = None
            robot["target_location"]     = None
            robot["path"]                = []

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    def _build_observation(self) -> Dict:
        """Build GNN observation using build_padded_ego_batch."""
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
        """
        Shaped reward with incremental deltas:
          +W_COMP  per completed task
          -W_WAIT * (wait / WAIT_CAP)  per pickup (wait since release)
          -W_DEADLINE * 0.5  per obsolete task
        """
        new_complete = self.episode_completed_count - self._prev_completed_count
        new_obsolete = self.episode_obsolete_count  - self._prev_obsolete_count

        self._prev_completed_count = self.episode_completed_count
        self._prev_obsolete_count  = self.episode_obsolete_count
        self._prev_pickup_count    = self.episode_pickup_count
        self._prev_dropoff_count   = self.episode_dropoff_count

        W_COMP     = 1.0
        W_WAIT     = 1.5
        W_DEADLINE = 5.0
        WAIT_CAP   = max(1.0, float(self.max_wait_delay_s))

        reward = float(new_complete) * W_COMP

        # Wait penalty: assessed at moment of pickup
        for robot in self.robots.values():
            picked_id = robot.get("just_picked_up_task")
            if picked_id:
                t = self.tasks.get(picked_id)
                if t is not None:
                    wait_s  = max(0.0, self.current_time - t["release_time"])
                    reward += -(min(wait_s, WAIT_CAP) / WAIT_CAP) * W_WAIT
                robot["just_picked_up_task"] = None

        # Obsolescence penalty
        reward -= float(new_obsolete) * (W_DEADLINE * 0.5)

        return reward

    # =========================================================================
    # TERMINATION
    # =========================================================================

    def _check_episode_done(self) -> bool:
        """
        The episode is done when:
          1. Every task from every batch has been released (injected into
             self.tasks) — meaning current_time has passed the last
             batch's release_time, AND
          2. Every released task is either completed or obsolete, AND
          3. All robots are idle (no current task and no queued tasks).
        """
        # Condition 1: all tasks released
        if len(self.tasks) < self.total_task_count:
            return False

        # Condition 2: no task still pending
        any_pending = any(
            not t.get("is_completed") and not t.get("is_obsolete")
            for t in self.tasks.values()
        )
        if any_pending:
            return False

        # Condition 3: robots idle
        robots_idle = all(
            len(r["assigned_tasks"]) == 0 and r["current_task"] is None
            for r in self.robots.values()
        )
        return robots_idle

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def action_mask(self) -> np.ndarray:
        """Return valid action mask (R, K_max+1). NO-OP is always valid."""
        mask = np.zeros((self.num_robots, self.K_max + 1), dtype=np.uint8)
        for r in range(self.num_robots):
            for k in range(min(self.K_max, len(self._last_cand_task_ids[r]))):
                if self._last_cand_task_ids[r][k] is not None:
                    mask[r, k] = 1
            mask[r, self._noop_index] = 1   # NO-OP always allowed
        return mask

    def close(self):
        pass