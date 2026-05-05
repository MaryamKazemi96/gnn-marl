import numpy as np
import gym
from gym import spaces
from pathlib import Path
import sys
import yaml
import random
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.graph_utils import get_edge_idx_graph, update_shared_attribute_matrix
# from utils.utils import enlarge_obstacles
from utils import utils as ut
from PIL import Image

# Constants for task data structure
TASK_RELEASE_TIME_INDEX = 7  # Index of release time in task info array
DEFAULT_BATCH_TIME = 180  # Default time buffer for completing a batch


class Planner:
    def __init__(self):
        # Load the map and create an obstacle grid
        root_path = Path(__file__).resolve().parent.parent.parent / "env"
        config_path = root_path / "ATC_wed.yaml"
        # print(config_path)
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
        grid_height, grid_width =  int(img_h * scale), int(img_w * scale)
        # print(scale, grid_height, grid_width, img_w, img_h) 
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)  # (rows, cols)
        for row in range(grid_height):
            for col in range(grid_width):
                px = int((col + 0.5) * img_w / grid_width)
                py = int((row + 0.5) * img_h / grid_height)
                grid[row, col] = 1 if self.map_img.getpixel((px, py)) < (self.threshold *255) else 0
        return grid
    
    def is_point_valid(self, point):
        # print(f"Checking validity of point: {point}")
        grid = self.get_obstacle_grid()
        w = point[1]
        h = point[0] 
        # print( row, col, grid.shape)
        if 0 <= h < grid.shape[0] and 0 <= w < grid.shape[1]:
            # print(f"Checking validity of point: {point} , valid")
            return grid[h,w] == 0
        # print(f"Checking validity of point: {point} , out of bounds")
        return False
    
    def get_plan(self, start, end):
        # print(f"Planning from {start} to {end}, obstacle grid shape: {self.get_obstacle_grid().shape}")
        """Generate a trajectory from origin to goal using A*."""
        grid = self.get_obstacle_grid()
        # print(grid, 'obstacle grid in get plan')
        # Use A* to find a path
        found, path = ut.astar(grid, start, end)
        # print(f"A* found: {found} after astar before smoothing, path: {path}")
        # if found:
        #     trajectory = ut.smooth_astar_path(path, self.Planning_resolution, self.origin_x, self.origin_y, self.average_velocity)
        # else:
        #         print(f"No valid path found")
        #         return 0, []
        return found, path # this path is all cell with [h,w] order
    
class Robot:
    def __init__(self, id, init_coordinates, maxCapacity=2, feature_size=9):
        self.robot_id = id
        self.coordinate = np.array([init_coordinates[1], init_coordinates[0], init_coordinates[2]], dtype=np.float32).reshape(1, -1)
        self.coordinate = self.coordinate[0]
        self.capacity = 0
        self.current_task_stage= []  # "pickup" or "dropoff"
        self.maxCapacity = maxCapacity
        self.feature_size = feature_size
        self.needs_replan = False
        self.reset()

    def reset(self):
        self.current_tasks_coords = [] # list(np.zeros((self.capacity, 2)))
        self.current_dropoff_coords = [] #list(np.zeros((self.capacity, 2)))
        self.current_tasks_id = [] #list(np.zeros(self.maxCapacity))
        self.trajectory = []
        self.goal_list = []
        self.capacity = 0 
        self.current_task_stage = []
        self.needs_replan = False

    def assign_trajectory(self, trajectory):
        self.trajectory = list(trajectory)
    
    def _task_reached(self, reach_tol=0.5):
        """
        Check whether the robot has reached any pickup or dropoff location.

        Returns:
            reached_pickup_ids: list[int]
            reached_dropoff_ids: list[int]
        """
        reached_pickup_ids = []
        reached_dropoff_ids = []

        if not self.current_tasks_coords:
            return reached_pickup_ids, reached_dropoff_ids

        # Convert to array safely
        coords = np.array(self.current_tasks_coords, dtype=np.float32)

        # Euclidean distance
        dists = np.linalg.norm(self.coordinate[:2] - coords[:, :2], axis=1)

        # Iterate BACKWARDS so pop() is safe
        for i in range(len(dists) - 1, -1, -1):

            # use tolerance (<= reach_tol means it's considered reached)
            if dists[i] > reach_tol:
                continue

            task_id = self.current_tasks_id[i]

            # -------- PICKUP --------
            if self.current_task_stage[i] == "pickup":
                reached_pickup_ids.append(task_id)

                # Switch target to dropoff
                self.current_tasks_coords[i] = self.current_dropoff_coords[i]
                self.goal_list[i] = self.current_dropoff_coords[i]
                self.current_task_stage[i] = "dropoff"
                # print(f"Robot {self.robot_id} picked up task {task_id}")

                # Re-order goals and mark for replan so the robot will head to the nearest goal next
                self.reorder_goals_by_distance()
                self.needs_replan = True

            # -------- DROPOFF --------
            else:
                reached_dropoff_ids.append(task_id)

                # Remove task completely
                self.current_tasks_coords.pop(i)
                self.current_dropoff_coords.pop(i)
                self.current_tasks_id.pop(i)
                self.current_task_stage.pop(i)
                self.goal_list.pop(i)
                self.capacity -= 1
                self.needs_replan = True
                # print(f"Robot {self.robot_id} dropped off task {task_id}")

        return reached_pickup_ids, reached_dropoff_ids


    def move(self):
        if self.trajectory:
            self.coordinate[:2] = np.array(self.trajectory.pop(0))
        return self.coordinate, self._task_reached()

    def add_task(self, task_id, task_pickup_coord, task_dropoff_coord):
        if self.capacity >= self.maxCapacity:
            return False
        self.current_tasks_id.append(task_id)
        self.current_tasks_coords.append(task_pickup_coord)
        self.current_dropoff_coords.append(task_dropoff_coord)
        self.goal_list.append(task_pickup_coord)
        self.current_task_stage.append("pickup")
        self.capacity += 1
        self.needs_replan = True
        return True
    
    def reorder_goals_by_distance(self):
        if not self.goal_list:
            return
        # Simple nearest-goal ordering
        current_pos = self.coordinate[:2]
        # sort goal_list and keep other related lists aligned
        order = sorted(range(len(self.goal_list)), key=lambda i: np.linalg.norm(current_pos - self.goal_list[i][:2]))
        # reorder goal_list and all parallel arrays to maintain consistency
        self.goal_list = [self.goal_list[i] for i in order]
        self.current_tasks_coords = [self.current_tasks_coords[i] for i in order]
        self.current_dropoff_coords = [self.current_dropoff_coords[i] for i in order]
        self.current_tasks_id = [self.current_tasks_id[i] for i in order]
        self.current_task_stage = [self.current_task_stage[i] for i in order]

    def get_attribute_array(self):
        coord_size = 3
        other_f_sizes = 1 + self.maxCapacity
        current_size = coord_size + other_f_sizes

        att = np.zeros(self.feature_size)
        # print(self.coordinate, 'robot coord in get attribute array')
        att[:3] = self.coordinate
        att[3] = self.capacity

        # Pad task IDs to maxCapacity
        padded_task_ids = np.zeros(self.maxCapacity)
        if len(self.current_tasks_id) > 0:
            padded_task_ids[:len(self.current_tasks_id)] = self.current_tasks_id

        att[4:4 + self.maxCapacity] = padded_task_ids
        return att

    
class Tasks_variable:
    def __init__(self, task_info_array, idle_allowance_time=60, init_time=0, feature_size=9):
        [task_id, w_origin, h_origin, yaw_origin,
         w_destination, h_destination, yaw_destination,
         t_release, pickupddl, estimatedTravelTime, dropoff_deadline] = task_info_array

        self.pick_up_coord = np.array([h_origin, w_origin, yaw_origin])
        self.drop_off_coord = np.array([h_destination, w_destination, yaw_destination])
        self.t_release = t_release
        self.id = task_id
        self.estimatedTravelTime = estimatedTravelTime
        self.idle_allowance_time = idle_allowance_time
        self.init_time = init_time
        self.feature_size = feature_size
        self.is_assigned = False
        self.reset(init_time=0)

    @property
    def is_active(self):
        if self.is_pickedup:
            return 0
        else:
            return 1

    def is_released(self, current_time=0):
        """Check if the task has been released based on its release time."""
        return current_time >= self.release_time

    def is_obsolete(self, current_time=0):
        if (not self.is_pickedup) and (current_time > self.ddl_pick * 4.2):
            return 1
        # if picked up but dropoff deadline passed and still not dropped
        if self.is_pickedup and (not self.is_droppedoff) and (current_time > self.ddl_dropoff *1):
            return 1
        return 0
        # if self.ddl_pick > current_time and not self.is_pickedup:
        #     return 0
        # elif self.ddl_dropoff > current_time and not self.is_droppedoff:
        #     return 0

    def picked_up(self):
        self.is_pickedup = 1
        self.node_type = "pick_up"
        self.coordinate = self.drop_off_coord

    def drop_off(self):
        self.is_droppedoff = 1
        # self.is_active = 0

    def reset(self, init_time=0):
        if init_time != 0:
            self.init_time = init_time
        self.is_pickedup = 0
        self.is_droppedoff = 0
        self.release_time = int(self.t_release)
        self.ddl_pick = self.release_time + int(self.idle_allowance_time)
        self.ddl_dropoff = self.ddl_pick + int(self.estimatedTravelTime * 1.5)
        self.coordinate = np.array(self.pick_up_coord, dtype=np.float32).reshape(1, -1)
        # bookkeeping for one-time rewards / penalties
        self.picked_by = None                 # which robot picked it up
        self.pickup_reward_given = False      # one-time pickup reward flag
        self.delivered_by = None              # set at dropoff
        self.delivered_reward_given = False  # one-time delivery reward flag
        self.assigned_to = None               # robot id assigned (set when assigned)
        self.obsolete_penalty_given = False   # one-time obsolete penalty flag

    def picked_up(self):
        self.is_pickedup = 1
        self.node_type = "pick_up"
        self.coordinate = self.drop_off_coord
        # picked_by will be set by the environment (step) to the robot id that picked it

    def drop_off(self):
        self.is_droppedoff = 1
        # delivered_by will be set by the environment (step) to the robot id that dropped it

    def get_attribute_array(self):
        coord_size = 3
        other_f_sizes = 6
        current_size = coord_size + other_f_sizes
        if current_size > self.feature_size:
            return False
        att = np.zeros(self.feature_size)
        att[:3] = self.coordinate
        att[3:current_size] = [
            self.ddl_pick,
            self.ddl_dropoff,
            float(self.is_obsolete(self.init_time)),
            float(self.is_pickedup),
            float(self.is_droppedoff),
            float(self.is_assigned) 
        ]
        return att
class MultiTaskAllocationEnv(gym.Env):
    def __init__(self, agents_cont_coord_array, task_cont_coord_array, radius=20, feature_size=9, use_true_id=False, all_batches=False, reward_mode="new"):
        super(MultiTaskAllocationEnv, self).__init__()
        self.planner = Planner()
        self.robot_capacity = 2
        self.radius = radius
        self.feature_size = feature_size
        self.agents_cont_coord_array = agents_cont_coord_array
        self.n_robots = len(self.agents_cont_coord_array)
        self.use_true_id = use_true_id
        self.current_traj = {i: [] for i in range(self.n_robots)}
        self.reward_mode = reward_mode
        
        # Handle all_batches mode: flatten list of batches into single task array
        if all_batches and isinstance(task_cont_coord_array, list):
            # task_cont_coord_array is a list of batches, flatten them
            self.task_cont_coord_array = []
            for batch in task_cont_coord_array:
                self.task_cont_coord_array.extend(batch)
            self.task_cont_coord_array = np.array(self.task_cont_coord_array)
            
            # Calculate max steps based on number of batches and their release times
            # Assuming last batch has highest release time
            if len(self.task_cont_coord_array) > 0:
                max_release_time = max(task[TASK_RELEASE_TIME_INDEX] for task in self.task_cont_coord_array)
                # Set batch_time to max_release_time + buffer for completing last batch
                self.batch_time = int(max_release_time + DEFAULT_BATCH_TIME)
            else:
                self.batch_time = DEFAULT_BATCH_TIME
        else:
            # Single batch mode (original behavior)
            self.task_cont_coord_array = task_cont_coord_array
            self.batch_time = DEFAULT_BATCH_TIME
            
        self.tasks_batches = task_cont_coord_array
        self.n_tasks = len(self.task_cont_coord_array)
        self.time_count = 0
        self.observation_space = spaces.Box(0, self.n_robots, shape=(feature_size,), dtype=int)
        self.action_space = spaces.Discrete(self.n_robots)
        self.reset()

    # Insert this inside the MultiTaskAllocationEnv class (e.g., after reset_tasks)

    def set_batch(self, task_batch):
        """
        Replace environment's current tasks with `task_batch` and reset task/robot
        state so an episode starts cleanly on this batch.

        task_batch should be an iterable (list/ndarray) of task_info rows
        in the same format your Tasks_variable expects (the same as
        elements of task_cont_coord_array).
        """
        # Replace the container used by reset_tasks
        self.task_cont_coord_array = task_batch

        # Rebuild the tasks list using your existing reset_tasks() helper
        # which reads from self.task_cont_coord_array
        self.reset_tasks()

        # Reset robots (positions/capacities). We reset robots so episodes
        # start from the same robot initialization each episode.
        self.reset_robot()

        # Rebuild the shared attribute matrix (robots then tasks)
        self._init_attribute_matrix()

        # reset time counter so deadlines / time-dependent behavior start fresh
        self.time_count = 0

    def reset(self):
        self.time_count = 0
        self.assign_traj = []
        self.reset_tasks()
        self.reset_robot()
        self._init_attribute_matrix()
        return self._get_observations(update_node_att=False), {}

    def reset_robot(self):
        self.robots = []
        self.robots_id = []
        self.robots_info = []

        for oneagent in self.agents_cont_coord_array:
            rid = oneagent[0]
            coord = oneagent[1:]
            agent = Robot(rid, coord, self.robot_capacity)
            self.robots.append(agent)
            self.robots_id.append(rid)
            self.robots_info.append(agent.get_attribute_array())
        self.robots_info = np.array(self.robots_info)

    # def reset_tasksmain(self):
    #     self.tasks = []
    #     self.tasks_id = []
    #     self.tasks_info = []
    #     for tasks_info in self.task_cont_coord_array:
    #         one_task = Tasks_variable(tasks_info)
    #         self.tasks.append(one_task)
    #         self.tasks_id.append(one_task.id)
    #         self.tasks_info.append(one_task.get_attribute_array())
    #     self.tasks_info = np.array(self.tasks_info)
    #     self.taskid_to_task = {t.id: t for t in self.tasks}

    def reset_tasks(self):
        self.tasks = []
        self.tasks_id = []
        self.tasks_info = []
        for tasks_info in self.task_cont_coord_array:
            one_task = Tasks_variable(tasks_info)
            self.tasks.append(one_task)
            self.tasks_id.append(one_task.id)
            self.tasks_info.append(one_task.get_attribute_array())

        # Ensure tasks_info is 2D even when there are zero tasks
        if len(self.tasks_info) == 0:
            self.tasks_info = np.zeros((0, self.feature_size), dtype=np.float32)
        else:
            self.tasks_info = np.array(self.tasks_info, dtype=np.float32)

        # Update number of tasks and mapping
        self.n_tasks = len(self.tasks)
        self.taskid_to_task = {t.id: t for t in self.tasks}
        # print(self.tasks_info, 'tasks info in reset tasks')

    def _init_attribute_matrix(self):
        # print("[debug] Initializing attributes matrix with robots_info and tasks_info", self.tasks_info)
        self.attributes_matrix = np.row_stack((self.robots_info, self.tasks_info))

    def _get_observations(self, update_node_att=True):
        if update_node_att:
            # print(self.attributes_matrix.shape, 'attributes matrix shape before update in get observations')
            self.update_nodes_attr()
            # print("[debug] Attributes Matrix in _get_observations after update_nodes_attr:", self.attributes_matrix)
            # print(f"Task IDs after update_shared_attribute_matrix: {self.attributes_matrix[:, 0]}")
        self.update_graph()
        # print("[debug] attributes_matrix in _get_observationsafter update_graph", self.attributes_matrix)
        return self.list_ego_graphs, self.attributes_matrix

    def update_nodes_attr(self):
        # self.attributes_matrix, self.trueid_idx_mapping = update_shared_attribute_matrix(
        #     self.attributes_matrix, self.robots_info, self.tasks_info
        # )
        # print("[debug] Attributes Matrix Before Update:", self.attributes_matrix)
        # print("[debug] Tasks Info:", self.tasks_info)
        # print("[debug] Robots Info:", self.robots_info)
        self.attributes_matrix, self.trueid_idx_mapping = update_shared_attribute_matrix(
            self.attributes_matrix, self.robots_info, self.tasks_info
        )
        # print("[debug] Attributes Matrix After Update:", self.attributes_matrix)
    def update_nodes_attr2(self):
        """
        Update the shared attributes_matrix and trueid->index mapping.

        Try the incremental merge for efficiency. If anything looks incompatible
        (different id sets, shape mismatch, or an exception), fall back to a
        full rebuild that stacks robots_info and tasks_info from scratch.
        """
        # Recompute canonical robots_info and tasks_info arrays for the current objects
        robots_info = np.array([r.get_attribute_array() for r in self.robots], dtype=np.float32)

        if hasattr(self, "tasks") and len(self.tasks) > 0:
            tasks_info = np.array([t.get_attribute_array() for t in self.tasks], dtype=np.float32)
        else:
            tasks_info = np.zeros((0, self.feature_size), dtype=np.float32)

        # Fast path: try incremental update only if we already have an attributes_matrix
        try:
            if hasattr(self, "attributes_matrix") and self.attributes_matrix is not None and self.attributes_matrix.size != 0:
                updated_attr, updated_mapping = update_shared_attribute_matrix(
                    self.attributes_matrix, robots_info, tasks_info
                )
                # Sanity check: mapping size must match current number of tasks
                if isinstance(updated_mapping, dict) and len(updated_mapping) == tasks_info.shape[0]:
                    # Accept incremental result
                    self.attributes_matrix = updated_attr
                    self.trueid_idx_mapping = updated_mapping
                    self.robots_info = robots_info
                    self.tasks_info = tasks_info
                    return
                # else fall through to full rebuild
        except Exception:
            # any failure -> fall back to full rebuild
            pass

        # Full rebuild (safe)
        self.robots_info = robots_info
        self.tasks_info = tasks_info
        # Stack robots then tasks into attributes_matrix (handles zero tasks safely)
        self.attributes_matrix = np.row_stack((self.robots_info, self.tasks_info))
        # Rebuild mapping from true task id to row index (task rows start at self.n_robots)
        self.trueid_idx_mapping = {t.id: (i + self.n_robots) for i, t in enumerate(self.tasks)}

    def robot_id_to_index(self, robot_id):
        """
        Map a robot's unique id to its index in the robots list.
        Returns None if not found.
        """
        for idx, robot in enumerate(self.robots):
            if robot.robot_id == robot_id:
                return idx
        return None
    def update_graph(self):
        # Filter tasks based on release time
        # print(f"Task States: {[t.id for t in self.tasks]} -> {[t.is_assigned for t in self.tasks]} -> {[t.release_time for t in self.tasks]}")
        # available_tasks = [
        #     t for t in self.tasks if t.release_time <= self.time_count and not t.is_assigned
        # ]
        available_tasks = [
            t for t in self.tasks if t.release_time <= self.time_count
        ]

        # Update the task info to include only available tasks
        self.tasks_info = np.array(
            [t.get_attribute_array() for t in available_tasks], dtype=np.float32
        )
        self.n_tasks = len(available_tasks)

        # Rebuild the attribute matrix (robots + available tasks)
        self._init_attribute_matrix()

        # Separate robots into active and full-capacity
        full_capacity_robots = [robot for robot in self.robots if robot.capacity >= 2]
        active_robots = [robot for robot in self.robots if robot.capacity < 2]

        # task_assigned_flags = self.attributes_matrix[self.n_robots:, -1]  # assuming last column is is_assigned
        # available_task_idx = np.where(task_assigned_flags == 0)[0]
        # print(f"Available Task Indices: {available_task_idx}")
        # Generate ego graphs for the current state
        self.list_ego_graphs, _, self.trueid_idx_mapping = get_edge_idx_graph(
            self.attributes_matrix, self.n_tasks, len(self.robots), self.radius, self.use_true_id
        )
        # print("[ENV] use_true_id:", self.use_true_id)
        # keys = list(self.list_ego_graphs.keys())
        # print("[ENV] ego keys sample:", keys[:10])
        # if keys:
        #     e0 = self.list_ego_graphs[keys[0]][0][:5]
        #     print("[ENV] first edges sample:", e0)
        # print("[ENV] attr first-col ids sample:", self.attributes_matrix[:15, 0].astype(int).tolist())

        # print("[DEBUG base_env] list_ego_graphs[0] sample:", self.list_ego_graphs.get(0, [])[:1])
        # print("[DEBUG base_env] list_ego_graphs[1] sample:", self.list_ego_graphs.get(1, [])[:1])
        # print("[DEBUG base_env] tasks_info true ids:", self.tasks_info[:,0].astype(int).tolist() if len(self.tasks_info) > 0 else [])
        # print(f"Ego Graphs before removals: {self.list_ego_graphs}")
        
        # ---- DEBUG: ego graph structure (print only sometimes) ----
        # if not hasattr(self, "_debug_ego_prints"):
        #     self._debug_ego_prints = 0
        # if self._debug_ego_prints < 3:  # print only first 3 calls
        #     self._debug_ego_prints += 1
        #     print("\n[DEBUG base_env] use_true_id =", self.use_true_id)
        #     print("[DEBUG base_env] attributes_matrix shape:", self.attributes_matrix.shape)
        #     print("[DEBUG base_env] first 10 true_ids in attributes_matrix[:,0]:",
        #         self.attributes_matrix[:10, 0].astype(int).tolist())

        #     keys = list(self.list_ego_graphs.keys())
        #     print("[DEBUG base_env] ego_graph keys sample:", keys[:min(10, len(keys))])
        #     # Show one robot ego list
        #     if keys:
        #         k0 = keys[0]
        #         ego_list = self.list_ego_graphs.get(k0, [])
        #         print(f"[DEBUG base_env] ego_list for key={k0}: blocks={len(ego_list)}")
        #         if ego_list:
        #             blk0 = np.asarray(ego_list[0])
        #             print("[DEBUG base_env] first block shape:", blk0.shape)
        #             print("[DEBUG base_env] first 5 edges of first block:", blk0[:5].tolist())

        #     # mapping format check
        #     m = self.trueid_idx_mapping
        #     if isinstance(m, (list, tuple)) and len(m) == 2:
        #         print("[DEBUG base_env] trueid_idx_mapping list lens:",
        #             len(m[0]), len(m[1]),
        #             "sample ids:", np.asarray(m[1])[:10].astype(int).tolist())
        #     elif isinstance(m, dict):
        #         print("[DEBUG base_env] trueid_idx_mapping is dict, size:", len(m))
        #     else:
        #         print("[DEBUG base_env] trueid_idx_mapping type:", type(m))
        # Remove edges for full-capacity robots
        full_capacity_robot_ids = [robot.robot_id for robot in full_capacity_robots]
        
        full_capacity_robot_indexes = [self.robot_id_to_index(rid) for rid in full_capacity_robot_ids]
        

        # print(full_capacity_robot_ids, 'full capacity robot ids in update graph')
        from utils.graph_utils import remove_robot_edges
        remove_robot_edges(self.list_ego_graphs, full_capacity_robot_indexes)

        # Remove assigned tasks from the ego graphs
        assigned_task_ids = [t.id for t in self.tasks if t.is_assigned]
        print(assigned_task_ids, 'assigned task ids in update graph')
        id_to_index = {task_id: idx for idx, task_id in enumerate(self.taskid_to_task.keys())}
        mapped_indices = [id_to_index[t_id] for t_id in assigned_task_ids]
        print(mapped_indices, 'mapped indices of assigned tasks in update graph')
        id_to_remove = [i + len(self.robots_id) for i in mapped_indices]
        print(id_to_remove, 'mapped indices in update graph')

        from utils.graph_utils import delete_taskid_in_graph
        # delete_taskid_in_graph(self.list_ego_graphs, id_to_remove)
        # print(f"Ego graphs before removing assigned tasks: {self.list_ego_graphs}")
        delete_taskid_in_graph(self.list_ego_graphs, id_to_remove)
        # print(f"Ego graphs after removing assigned tasks and full capacity robots: {self.list_ego_graphs}")
        return self.list_ego_graphs

    def resolve_conflicts(self, assignments):
        """
        Resolve conflicts using top-2 task choices per robot.
        Returns a dict {robot_id: assigned_task_id}.
        """
        final_assignments = {}
        taken_tasks = set()

        # Process robots in order of their IDs
        for rid in sorted(assignments.keys()):
            top2_tasks = assignments[rid]
            if not top2_tasks or top2_tasks[0] is None:
                continue  # no task available

            first_choice = top2_tasks[0]
            second_choice = top2_tasks[1] if len(top2_tasks) > 1 else None

            if first_choice not in taken_tasks:
                final_assignments[rid] = first_choice
                taken_tasks.add(first_choice)
            elif second_choice is not None and second_choice not in taken_tasks:
                final_assignments[rid] = second_choice
                taken_tasks.add(second_choice)
            else:
                # no available task
                final_assignments[rid] = None

        # Remove robots that couldn't be assigned any task
        final_assignments = {rid: tid for rid, tid in final_assignments.items() if tid is not None}
        return final_assignments

    def _get_task_from_assignment_id(self, task_identifier):
        """
        Accept either:
         - unique task id (task.id), or
         - node index used in graph (n_robots + task_index)
        and return the corresponding Tasks_variable instance or None.
        """
        # If it's exactly a unique task id in our mapping, return it directly
        if task_identifier in self.taskid_to_task:
            return self.taskid_to_task[task_identifier]

        # If it looks like a node index (>= n_robots), map to task_index
        if isinstance(task_identifier, int) and task_identifier >= len(self.robots_id):
            task_idx = task_identifier - len(self.robots_id)
            if 0 <= task_idx < len(self.tasks):
                return self.tasks[task_idx]
        # no mapping found
        return None
    
    def get_available_task_ids(self):
        # print([self.taskid_to_task[t_id].is_assigned for t_id in self.taskid_to_task], 'task assigned status in get available task ids')
        return [
            tid for tid, t in self.taskid_to_task.items()
            # if t.is_active and not t.is_assigned and t.is_released(self.time_count)
            if t.is_active and not t.is_assigned and t.release_time <= self.time_count and t.is_obsolete(self.time_count) == 0
        ]

    def _plan_robot_trajectory(self, robot):
        """
        Build a full trajectory for `robot` by planning from the robot's current position
        through each goal in robot.goal_list in order. Returns a list (possibly empty).
        """
        full_trajectory = []
        start = (int(robot.coordinate[0]), int(robot.coordinate[1]))
        for goal in robot.goal_list:
            end = (int(goal[0]), int(goal[1]))
            try:
                found, traj = self.planner.get_plan(start, end)
            except Exception as e:
                print(f"Planner error when planning {start} -> {end}: {e}")
                continue
            if found and traj:
                full_trajectory.extend(traj)
                start = end
            else:
                # if planning to this goal segment failed, skip it and continue
                continue
        return full_trajectory
    def _cancel_task_on_robot(self, rid: int, task_id: int):
        """
        Remove task_id from robot rid's internal lists and free capacity if present.
        Safe even if task_id is not found.
        """
        robot = self.robots[rid]
        # remove all occurrences (should be at most one)
        removed_any = False
        for i in range(len(robot.current_tasks_id) - 1, -1, -1):
            if robot.current_tasks_id[i] == task_id:
                robot.current_tasks_id.pop(i)
                robot.current_tasks_coords.pop(i)
                robot.current_dropoff_coords.pop(i)
                robot.current_task_stage.pop(i)
                robot.goal_list.pop(i)
                robot.capacity = max(0, robot.capacity - 1)
                removed_any = True
        if removed_any:
            robot.needs_replan = True

    def step(self, list_t2r_assignments=None, assignment_interval=8):
    
        final_assignments_for_step = {}
        
        # -------------------------------------------------
        # 1. ROBOT MOTION + TASK INTERACTION (unchanged)
        # -------------------------------------------------
        for ridx, robot in enumerate(self.robots):
            if len(robot.trajectory) > 0:
                coord, reached = robot.move()
                reached_pick, reached_drop = reached
            else:
                reached_pick, reached_drop = robot._task_reached()

            for task_id in reached_pick:
                task = self._get_task_from_assignment_id(task_id)
                if task is not None:
                    task.picked_by = ridx
                    task.picked_up()
                    task.pickup_reward_given = False

            for task_id in reached_drop:
                task = self._get_task_from_assignment_id(task_id)
                if task is not None:
                    task.delivered_by = ridx
                    task.drop_off()
                    task.delivered_reward_given = False

            if robot.needs_replan or (len(robot.trajectory) == 0 and len(robot.goal_list) > 0):
                full_trajectory = self._plan_robot_trajectory(robot)
                if full_trajectory:
                    robot.assign_trajectory(full_trajectory)
                robot.needs_replan = False
        
        # -------------------------------------------------
        # 2. ASSIGNMENT LOGIC
        # -------------------------------------------------
        decision_step = (self.time_count % assignment_interval == 0)
        
        # NEW: Check if there are actually available tasks
        available_tasks = self.get_available_task_ids()
        has_available_tasks = len(available_tasks) > 0
        
        # NEW: Check if any robot can accept tasks
        has_available_robots = any(r.capacity < r.maxCapacity for r in self.robots)
        
        # MODIFIED: Only mark as meaningful decision step if actions are possible
        meaningful_decision_step = decision_step and has_available_tasks and has_available_robots
        
        if decision_step:
            if has_available_tasks and has_available_robots:
                # Generate assignments
                if list_t2r_assignments is None:
                    list_t2r_assignments = {}
                    
                    for rid in range(self.n_robots):
                        robot = self.robots[rid]
                        
                        if robot.capacity >= robot.maxCapacity:
                            list_t2r_assignments[rid] = [None, None]
                            continue
                        
                        if not available_tasks:
                            list_t2r_assignments[rid] = [None, None]
                            continue
                        
                        num_choices = min(2, len(available_tasks))
                        top2 = np.random.choice(
                            available_tasks,
                            size=num_choices,
                            replace=False
                        ).tolist()
                        
                        if len(top2) == 1:
                            top2.append(top2[0])
                        
                        list_t2r_assignments[rid] = top2
                
                resolved_assignments = self.resolve_conflicts(list_t2r_assignments)
                self._get_final_assigment(resolved_assignments)
                final_assignments_for_step = resolved_assignments.copy()
                # print(f"Decision Step: {self.time_count} | Assignments: {final_assignments_for_step}")
        
        #----handel obsolete tasks---=
        for task in self.tasks:
            if task.is_assigned and (not task.is_pickedup) and task.is_obsolete(self.time_count):
                rid = task.assigned_to
                if rid is not None and 0 <= rid < self.n_robots:
                    self._cancel_task_on_robot(rid, task.id)
                task.is_assigned = False
                task.assigned_to = None
        # -------------------------------------------------
        # 3. COMPUTE REWARD
        # -------------------------------------------------
        if self.reward_mode == "new":
            reward, info_reward = self.reward(debug=True)
        else:
            reward, info_reward = self.rewardold(debug=True)
        # print(info_reward, 'reward info in step')
        # print(reward, 'reward in step')
        
        # -------------------------------------------------
        # 4. CHECK TERMINATION
        # -------------------------------------------------
        terminated = all([
            t.is_droppedoff or t.is_obsolete(self.time_count)
            for t in self.tasks
        ])

        truncated = self.time_count >= self.batch_time
        
        # -------------------------------------------------
        # 5. UPDATE OBSERVATIONS
        # -------------------------------------------------
        obs = self._get_observations(update_node_att=True)
        # print("[debug] obs in base env step:", obs)
        # -------------------------------------------------
        # 6. INCREMENT TIME AND RETURN
        # -------------------------------------------------
        
        
        info = {
            "resolved_assignments": final_assignments_for_step,
            "decision_step": decision_step,
            "meaningful_decision_step": meaningful_decision_step,  
            "time_count": self.time_count,
            "available_tasks": len(available_tasks),  
            "available_robots": sum(1 for r in self.robots if r.capacity < r.maxCapacity),
            
        }
        info_reward["reward_mode"] = self.reward_mode
        # print(info, 'info in step')
        # print('final_assignments_for_step', final_assignments_for_step)
        # print(f"Step {self.time_count} | Reward: {reward} | Terminated: {terminated} | is_obsolete: {is_obsolete}|| is_droppedoff: {is_droppedoff}||Truncated: {truncated} | Info: {info}")
        self.time_count += 1
        return obs, reward, terminated, truncated, info_reward, info


    def _get_task_from_assignment_id(self, task_identifier):
        """
        Accept either:
         - unique task id (task.id), or
         - node index used in graph (n_robots + task_index)
        and return the corresponding Tasks_variable instance or None.
        """
        # If it's exactly a unique task id in our mapping, return it directly
        if task_identifier in self.taskid_to_task:
            return self.taskid_to_task[task_identifier]

        # If it looks like a node index (>= n_robots), map to task_index
        if isinstance(task_identifier, int) and task_identifier >= len(self.robots_id):
            task_idx = task_identifier - len(self.robots_id)
            if 0 <= task_idx < len(self.tasks):
                return self.tasks[task_idx]
        # no mapping found
        return None
    
    def _get_final_assigment(self, assignments):
        """
        assignments: dict {robot_id: task_identifier}
          - task_identifier may be either the unique task.id or a graph node index (n_robots + task_index)
        Behavior:
          - For each robot, attempt to add the task to the robot (respecting capacity).
          - If add_task succeeds, mark task.is_assigned = True.
          - Reorder robot.goal_list by distance and set robot.needs_replan = True
            (actual planning happens in step(), centrally).
        """
        # Process in deterministic order
        for rid in sorted(assignments.keys()):
            # validate robot id
            if rid < 0 or rid >= len(self.robots):
                # print(f"_get_final_assigment: invalid robot id {rid}, skipping")
                continue

            robot = self.robots[rid]
            task_identifier = assignments[rid]

            # skip if robot is full
            if robot.capacity >= robot.maxCapacity:
                # print(f"Robot {rid} at capacity {robot.capacity}/{robot.maxCapacity}, skipping assignment {task_identifier}")
                continue

            # Map identifier to task object (handles both unique task ids and graph node indices)
            task = self._get_task_from_assignment_id(task_identifier)
            if task is None:
                # print(f"_get_final_assigment: could not map task identifier {task_identifier} to a task object")
                continue

            # Skip if not active or already assigned
            if not task.is_active or task.is_assigned:
                # print(f"Task {task.id} not active or already assigned; skipping")
                continue

            # Attempt to add the task to the robot (this respects robot.maxCapacity)
            success = robot.add_task(task.id, task.pick_up_coord, task.drop_off_coord)
            if not success:
                # couldn't add task (capacity race), skip marking is_assigned
                # print(f"Robot {rid} failed to add task {task.id} (maybe capacity).")
                continue

            task.is_assigned = True
            task.assigned_to = rid 
            # mark the task assigned only after successful add_task
            task.is_assigned = True

            # Reorder goals so the robot will next go to the closest goal (could be another pickup)
            robot.reorder_goals_by_distance()

            # Mark that a replan is required; actual planning will be done in step()
            robot.needs_replan = True
        # print(f"Task {task.id}, is_assigned: {task.is_assigned} in get_final_assignment")
    
    def _reward_tasks_completion_per_robot(self, robot):
        """
        Count tasks delivered by this robot (uses task.delivered_by set at dropoff).
        Returns integer count.
        """
        delivered_by_id = getattr(robot, "robot_id", None)
        return sum(1 for t in self.tasks if t.is_droppedoff and getattr(t, "delivered_by", None) == delivered_by_id)

    # (4) Replace your reward() with the following function
    # In environment.py, modify the reward function:
    #new reward closer to Klavdiia's implementation

    def reward(self, debug: bool = True):
        """
        Colleague-like reward = weighted sum of interpretable components per robot.

        Terms (per robot):
        - completion: +W_COMPLETION for each dropoff event (one-time)
        - capacity:   +W_CAPACITY * current capacity (dense shaping)
        - step:       small negative per step (global-scaled)
        - wait_at_pickups: negative proportional to waiting time of assigned-not-picked tasks
        - missed_deadline/abandoned: penalty when task becomes obsolete (one-time)
        - nonserved/backlog: penalty for released & feasible tasks that remain unassigned

        Notes:
        - Uses task flags already in your env (picked_by, delivered_by, assigned_to, is_assigned, is_pickedup, etc.)
        - Uses simple waiting-time proxy: (now - release_time) for tasks not yet picked up.
        """
        n_robots = max(1, len(self.robots))
        rewards = {rid: 0.0 for rid in range(n_robots)}

        # ---------------- weights (start here, tune later) ----------------
        W_COMPLETION = 10.0          # like colleague's comp * 40
        W_CAPACITY = 2.0             # like colleague's cap * 2
        W_STEP = -0.1                # IMPORTANT: apply as GLOBAL then divide (see below)
        W_WAIT = -0.01               # penalty per second of waiting (tune)
        W_ABANDONED = -10.0          # like colleague's abandoned * -10
        W_MISSED_DROPOFF = -10.0     # picked but missed dropoff deadline
        W_BACKLOG = -0.02            # per released-unassigned task per step (tune)

        # ---------------- bookkeeping / debug counters ----------------
        completion_events = 0
        abandoned_events = 0
        missed_dropoff_events = 0

        # ---------------- 1) step penalty (global-scaled) ----------------
        # Your previous version did -0.1 per robot per step => total -0.1*n_robots.
        # Colleague's step penalty is effectively small; so we apply a GLOBAL penalty and split.
        step_each = (W_STEP / n_robots) if n_robots > 0 else 0.0
        for rid in range(n_robots):
            rewards[rid] += step_each

        # ---------------- 2) capacity shaping (dense) ----------------
        for rid, robot in enumerate(self.robots):
            rewards[rid] += W_CAPACITY * float(max(0, robot.capacity))

        # ---------------- 3) completion reward (one-time on dropoff event) ----------------
        for task in self.tasks:
            if task.is_droppedoff and (not getattr(task, "delivered_reward_given", False)):
                did = getattr(task, "delivered_by", None)
                if did is not None and 0 <= int(did) < n_robots:
                    rewards[int(did)] += W_COMPLETION
                    completion_events += 1
                task.delivered_reward_given = True

        # ---------------- 4) waiting penalty at pickups (assigned but not picked) ----------------
        # Penalize the robot that is holding responsibility for pickup.
        # This encourages choosing tasks you can actually reach soon.
        now = int(self.time_count)
        for task in self.tasks:
            if task.release_time <= now and task.is_assigned and (not task.is_pickedup) and (not task.is_obsolete(now)):
                rid = getattr(task, "assigned_to", None)
                if rid is None:
                    continue
                if 0 <= int(rid) < n_robots:
                    wait_t = max(0, now - int(task.release_time))
                    rewards[int(rid)] += W_WAIT * float(wait_t)

        # ---------------- 5) deadline miss / abandoned penalty (one-time) ----------------
        # Use your is_obsolete() but separate the two cases for debugging/weighting:
        # - not picked up => abandoned
        # - picked up but not dropped => missed dropoff
        for task in self.tasks:
            if task.is_obsolete(now) and (not getattr(task, "obsolete_penalty_given", False)):
                rid = getattr(task, "assigned_to", None)
                if rid is None:
                    rid = getattr(task, "picked_by", None)

                if rid is not None and 0 <= int(rid) < n_robots:
                    if (not task.is_pickedup):
                        rewards[int(rid)] += W_ABANDONED
                        abandoned_events += 1
                    elif (task.is_pickedup and not task.is_droppedoff):
                        rewards[int(rid)] += W_MISSED_DROPOFF
                        missed_dropoff_events += 1

                task.obsolete_penalty_given = True

        # ---------------- 6) backlog / nonserved (dense global) ----------------
        # Penalize unassigned released tasks that are not obsolete yet (system-level pressure).
        # Split across robots so scale doesn't explode with fleet size.
        backlog_tasks = 0
        for task in self.tasks:
            if task.release_time <= now and (not task.is_obsolete(now)) and (not task.is_assigned) and (not task.is_pickedup):
                backlog_tasks += 1

        if backlog_tasks > 0:
            backlog_each = (W_BACKLOG * float(backlog_tasks)) / n_robots
            for rid in range(n_robots):
                rewards[rid] += backlog_each

        if debug:
            info = {
                "sum_rewards": float(sum(rewards.values())),
                "terms": {
                    # system-level summaries (you can also log per-robot if you want)
                    "completion_events": int(completion_events),
                    "abandoned_events": int(abandoned_events),
                    "missed_dropoff_events": int(missed_dropoff_events),
                    "backlog_tasks": int(backlog_tasks),
                },
                "robot_capacities": [int(r.capacity) for r in self.robots],
            }
            return rewards, info

        return rewards

    def rewardmain(self, debug: bool = True):
        """
        Colleague-style reward (event-based + time pressure):
        - pickup reward: paid once when task becomes picked up
        - delivery reward: paid once when task becomes dropped off
        - obsolete penalty: paid once when a task becomes obsolete (picked or not)
        - step penalty: small negative each step to push faster completion
        """
        n_robots = max(1, len(self.robots))

        # Tunable constants (start here, then tune)
        R_PICKUP = 0.0
        R_DELIVERY = 10.0
        P_OBSOLETE = -5.0
        STEP_PENALTY = -0.1  # per robot per step (small)

        rewards = {rid: 0.0 for rid in range(n_robots)}
        pickup_count = 0
        delivery_count = 0
        obsolete_count = 0

        # per-step time pressure
        if STEP_PENALTY != 0.0:
            for rid in range(n_robots):
                rewards[rid] += STEP_PENALTY

        # pickup events (one-time)
        for task in self.tasks:
            if task.is_pickedup and (not getattr(task, "pickup_reward_given", False)):
                pid = getattr(task, "picked_by", None)
                if pid is None:
                    pid = getattr(task, "assigned_to", None)
                if pid is not None and 0 <= pid < n_robots:
                    rewards[int(pid)] += R_PICKUP
                    pickup_count += 1
                task.pickup_reward_given = True

        # delivery events (one-time)
        for task in self.tasks:
            if task.is_droppedoff and (not getattr(task, "delivered_reward_given", False)):
                did = getattr(task, "delivered_by", None)
                if did is not None and 0 <= did < n_robots:
                    rewards[int(did)] += R_DELIVERY
                    delivery_count += 1
                task.delivered_reward_given = True

        # obsolete events (one-time)
        for task in self.tasks:
            if task.is_obsolete(self.time_count) and (not getattr(task, "obsolete_penalty_given", False)):
                # attribute penalty to assigned robot if known, else nobody
                rid = getattr(task, "assigned_to", None)
                if rid is None:
                    rid = getattr(task, "picked_by", None)
                if rid is not None and 0 <= rid < n_robots:
                    rewards[int(rid)] += P_OBSOLETE
                obsolete_count += 1
                task.obsolete_penalty_given = True

        if debug:
            info = {
                "sum_rewards": float(sum(rewards.values())),
                "pickups_this_step": pickup_count,
                "deliveries_this_step": delivery_count,
                "obsolete_this_step": obsolete_count,
                # "step_penalty": STEP_PENALTY,
                # "R_PICKUP": R_PICKUP,
                # "R_DELIVERY": R_DELIVERY,
                # "P_OBSOLETE": P_OBSOLETE,
                "robot_capacities": [r.capacity for r in self.robots],
            }
            return rewards, info

        return rewards


    def _reward_tasks_completion_per_robot(self, robot):
        # Count tasks completed by this robot
        completed_task_ids = set(robot.current_tasks_id)
        reward = 0
        for task in self.tasks:
            if task.is_droppedoff and task.id in completed_task_ids:
                reward += 1
        return reward

    def _reward_steps(self):
        return -self.time_count * 1

    def _reward_agent_capacity(self):
        # print(f"{[r.capacity for r in self.robots]} robot capacities in reward agent capacity")
        return [max(0, r.capacity) for r in self.robots]
        # return max(0, self.robots[rid].capacity)

    def _reward_tasks_abandone_penalty(self):
        return -sum(1 for t in self.tasks if t.is_obsolete(self.time_count) and not t.is_pickedup)

    def render(self, mode="human"):
        pass