
# import gymnasium as gym
# import numpy as np
# from typing import Dict, Any, Tuple, Optional, List
# from pathlib import Path
# import sys
# import yaml
# from PIL import Image
# import torch as th
# from scipy.optimize import linear_sum_assignment
# sys.path.append(str(Path(__file__).resolve().parent.parent))
 
# from src.utils.ego_graph_builder import build_padded_ego_batch
# from src.utils.feature_fn import make_feature_fn, compute_feature_dim
# from utils import utils as ut
 
 
# # =============================================================================
# # Planner — A* path planning with cached obstacle grid
# # =============================================================================
 
# class Planner:
#     """
#     A* path planner over the ATC obstacle grid.
 
#     The obstacle grid is built once from the map image and cached on the
#     instance so that every call to get_plan() / is_point_valid() reuses the
#     same array instead of re-sampling the PNG pixel-by-pixel each time.
#     """
 
#     def __init__(self):
#         root_path = Path(__file__).resolve().parent.parent.parent / "env"
#         config_path = root_path / "ATC_wed.yaml"
#         with open(config_path, "r") as fh:
#             params = yaml.safe_load(fh)
 
#         map_path = root_path / params["map_filename"]
#         self.map_img = Image.open(map_path).convert("L")
#         self.map_resolution = params["map_resolution"]
#         self.Planning_resolution = params["Planning_resolution"]
#         self.threshold = params["obstacle_threshold"]
#         self.origin_x = params["origin_x"]
#         self.origin_y = params["origin_y"]
#         self.average_velocity = params["average_velocity"]
 
#         self._obstacle_grid: Optional[np.ndarray] = None
 
#     def get_obstacle_grid(self) -> np.ndarray:
#         """Return cached obstacle grid, building it on first call."""
#         if self._obstacle_grid is not None:
#             return self._obstacle_grid
 
#         img_w, img_h = self.map_img.size
#         scale = self.map_resolution / self.Planning_resolution
#         grid_height = int(img_h * scale)
#         grid_width = int(img_w * scale)
#         grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
 
#         for row in range(grid_height):
#             for col in range(grid_width):
#                 px = int((col + 0.5) * img_w / grid_width)
#                 py = int((row + 0.5) * img_h / grid_height)
#                 if self.map_img.getpixel((px, py)) < (self.threshold * 255):
#                     grid[row, col] = 1
 
#         self._obstacle_grid = grid
#         return grid
 
#     def is_point_valid(self, point: Tuple[int, int]) -> bool:
#         grid = self.get_obstacle_grid()
#         h, w = point
#         if 0 <= h < grid.shape[0] and 0 <= w < grid.shape[1]:
#             return grid[h, w] == 0
#         return False
 
#     def get_plan(self, start: Tuple[int, int], end: Tuple[int, int]):
#         """Return (found: bool, path: list[(row,col)]) via A*."""
#         grid = self.get_obstacle_grid()
#         return ut.astar(grid, start, end)
 
 
# # =============================================================================
# # MultiAgentTaskEnv
# # =============================================================================
 
# class MultiAgentTaskEnv(gym.Env):
#     """
#     Multi-agent task allocation environment.
#     """
 
#     def __init__(
#         self,
#         agents: np.ndarray = None,
#         tasks_batches: list = None,
#         agents_cont_coord_array: np.ndarray = None,
#         task_cont_coord_array: np.ndarray = None,
#         use_xy_pickup: bool = False,
#         normalize_features: bool = True,
#         use_node_type: bool = True,
#         use_ego_robot: bool = True,
#         use_edge_rt: bool = False,
#         edge_features=None,
#         N_max: int = 15,
#         E_max: int = 50,
#         K_max: int = 5,
#         max_robot_capacity: int = 2,
#         max_wait_delay_s: float = 600.0,
#         max_travel_delay_s: float = 3600.0,
#         max_steps: int = 2000,
#         two_hop: bool = False,
#         two_hop_directed: bool = False,
#         vicinity_m: float = 100.0,
#         movement_speed: float = 1.0,
#         decision_interval: int = 8,
#         radius: int = 100,
#         feature_size: int = 9,
#         use_true_id: bool = False,
#         reward_mode: str = "new",
#         capacity_method: str = "assigned",
#         conflict_resolution: str = "greedy",
#         W_COMP: float = 2.0,
#         W_WAIT: float = 1.0,
#         W_DEADLINE: float = 10.0,
#         W_OBS: float = 0.5,
#         candidates_sorting: str = "distance",
#         reward_type: str = "legacy",
#         W_TRAVEL: float = 1.25,
#     ):
#         super().__init__()
 
#         if agents is not None and tasks_batches is not None:
#             self.init_mode = "new"
#             self.agents_data = agents
#             self.tasks_batches = tasks_batches
#             self.num_robots = len(agents)
#             self.planner = Planner()
#         elif agents_cont_coord_array is not None and task_cont_coord_array is not None:
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
 
#         self.N_max = N_max
#         self.E_max = E_max
#         self.K_max = K_max
#         self.max_robot_capacity = max_robot_capacity
#         self.vicinity_m = vicinity_m
#         self.two_hop = two_hop
#         self.two_hop_directed = two_hop_directed
#         self.max_steps = max_steps
#         self.movement_speed = movement_speed
#         self.decision_interval = decision_interval
#         self.max_wait_delay_s = max_wait_delay_s
#         self.max_travel_delay_s = max_travel_delay_s
#         self.W_COMP = float(W_COMP)
#         self.W_WAIT = float(W_WAIT)
#         self.W_DEADLINE = float(W_DEADLINE)
#         self.W_OBS = float(W_OBS)
 
#         self.robots = {}
#         self.tasks = {}
#         self.current_time = 0.0
#         self.current_step = 0
#         self.total_task_count = 0
 
#         self.episode_completed_count = 0
#         self.episode_obsolete_count = 0
#         self.episode_pickup_count = 0
#         self.episode_dropoff_count = 0
#         self._prev_completed_count = 0
#         self._prev_obsolete_count = 0
#         self._prev_pickup_count = 0
#         self._prev_dropoff_count = 0
 
#        # diagnostics (episode-level)
#         self.debug_invalid_action_count = 0
#         self.debug_total_action_count = 0
#         self.debug_valid_action_count = 0
#         self.debug_conflict_dropped_count = 0
#         self.debug_capacity_rejected_count = 0
#         self.debug_noop_forced_count = 0     # noop because no candidates were offered
#         self.debug_noop_chosen_count = 0     # noop despite candidates being available
#         self.debug_had_candidates_count = 0  # decisions where >=1 real candidate existed
#         self.debug_decisions_total = 0
 
#         # diagnostics (last-step)
#         self.debug_last_invalid_action_count = 0
#         self.debug_last_total_action_count = 0
#         self.debug_last_valid_action_count = 0
#         self.debug_last_conflict_dropped_count = 0
#         self.debug_last_capacity_rejected_count = 0
#         self.debug_last_mask_zero_count = 0
#         self.debug_last_noop_forced_count = 0
#         self.debug_last_noop_chosen_count = 0
#         self.debug_last_had_candidates_count = 0
#         self.debug_last_decisions_total = 0
 
#         self.debug_last_r_comp = 0.0
#         self.debug_last_r_wait = 0.0
#         self.debug_last_r_deadline = 0.0
#         self.debug_last_r_obsolete = 0.0
 
#         self.debug_ep_r_comp = 0.0
#         self.debug_ep_r_wait = 0.0
#         self.debug_ep_r_deadline = 0.0
#         self.debug_ep_r_obsolete = 0.0
 
#         if self.init_mode == "new":
#             self.max_position = max(np.max(agents[:, 1]), np.max(agents[:, 2]))
#         else:
#             self.max_position = 100.0
 
#         self.F = compute_feature_dim(
#             use_xy_pickup=use_xy_pickup,
#             use_node_type=use_node_type,
#             use_edge_rt=use_edge_rt,
#             use_ego_robot=use_ego_robot,
#         )
 
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
 
#         if use_edge_rt:
#             self.edge_features = edge_features or ["dx", "dy", "eta"]
#             self.edge_feat_dim = len(self.edge_features)
#         else:
#             self.edge_features = []
#             self.edge_feat_dim = 0
 
#         self.observation_space = gym.spaces.Dict({
#             "x": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_robots, N_max, self.F), dtype=np.float32),
#             "node_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_robots, N_max), dtype=np.uint8),
#             "edge_index": gym.spaces.Box(low=0, high=N_max, shape=(self.num_robots, 2, E_max), dtype=np.int64),
#             "edge_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_robots, E_max), dtype=np.uint8),
#             "cand_idx": gym.spaces.Box(low=0, high=N_max, shape=(self.num_robots, K_max), dtype=np.int64),
#             "cand_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_robots, K_max), dtype=np.uint8),
#         })
 
#         if self.edge_feat_dim > 0:
#             self.observation_space.spaces["edge_attr"] = gym.spaces.Box(
#                 low=-np.inf, high=np.inf, shape=(self.num_robots, E_max, self.edge_feat_dim), dtype=np.float32
#             )
 
#         self.action_space = gym.spaces.MultiDiscrete([K_max + 1] * self.num_robots)
#         self._noop_index = K_max
#         self._last_cand_task_ids = [[] for _ in range(self.num_robots)]
 
#         self.capacity_method = capacity_method.lower()
#         if self.capacity_method not in ("assigned", "pickup"):
#             raise ValueError(
#                 "capacity_method must be 'assigned' or 'pickup'"
#             )
        
#         self.conflict_resolution = conflict_resolution.lower()
#         _valid_resolvers = ("greedy", "random", "hungarian", "hungarian_bids",
#                             "predicted_reward", "predicted_reward_joint",
#                             "capacity", "closest_than_capacity")
#         if self.conflict_resolution not in _valid_resolvers:
#             raise ValueError(f"conflict_resolution must be one of {_valid_resolvers}")

#         self.candidates_sorting = candidates_sorting.lower()
#         if self.candidates_sorting not in ("distance", "randomized"):
#             raise ValueError("candidates_sorting must be 'distance' or 'randomized'")

#         self.reward_type = reward_type.lower()
#         if self.reward_type not in ("legacy", "wait_travel"):
#             raise ValueError("reward_type must be 'legacy' or 'wait_travel'")
#         self.W_TRAVEL = float(W_TRAVEL)
 
#         # Populated externally, once per macro-step, by RTGNNPolicy.forward()
#         # via set_pending_logits() — only used by conflict_resolution=
#         # 'hungarian_bids'. None until the first policy forward() call.
#         self._pending_logits = None
#     # =========================================================================
#     # RESET
#     # =========================================================================
 
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
 
#         self._pending_logits = None
#         self.current_time = 0.0
#         self.current_step = 0
#         self.episode_completed_count = 0
#         self.episode_obsolete_count = 0
#         self.episode_pickup_count = 0
#         self.episode_dropoff_count = 0
#         self._prev_completed_count = 0
#         self._prev_obsolete_count = 0
#         self._prev_pickup_count = 0
#         self._prev_dropoff_count = 0
 
#         self.debug_invalid_action_count = 0
#         self.debug_total_action_count = 0
#         self.debug_valid_action_count = 0
#         self.debug_conflict_dropped_count = 0
#         self.debug_capacity_rejected_count = 0
#         self.debug_noop_forced_count = 0
#         self.debug_noop_chosen_count = 0
#         self.debug_had_candidates_count = 0
#         self.debug_decisions_total = 0
 
#         self.debug_last_invalid_action_count = 0
#         self.debug_last_total_action_count = 0
#         self.debug_last_valid_action_count = 0
#         self.debug_last_conflict_dropped_count = 0
#         self.debug_last_capacity_rejected_count = 0
#         self.debug_last_mask_zero_count = 0
#         self.debug_last_noop_forced_count = 0
#         self.debug_last_noop_chosen_count = 0
#         self.debug_last_had_candidates_count = 0
#         self.debug_last_decisions_total = 0
 
#         self.debug_last_r_comp = 0.0
#         self.debug_last_r_wait = 0.0
#         self.debug_last_r_deadline = 0.0
#         self.debug_last_r_obsolete = 0.0
 
#         self.debug_ep_r_comp = 0.0
#         self.debug_ep_r_wait = 0.0
#         self.debug_ep_r_deadline = 0.0
#         self.debug_ep_r_obsolete = 0.0
 
#         if self.init_mode == "new":
#             self._reset_new_mode()
#         else:
#             self._reset_old_mode()
 
#         obs = self._build_observation()
#         return obs, {"action_mask": self.action_mask()}
 
#     def _reset_new_mode(self):
#         self.robots = {}
#         for agent in self.agents_data:
#             robot_id = str(int(agent[0]))
#             self.robots[robot_id] = {
#                 "id": robot_id,
#                 "x": float(agent[1]),
#                 "y": float(agent[2]),
#                 "max_capacity": self.max_robot_capacity,
#                 "current_capacity": 0,          # == len(onboard_tasks), kept for back-compat
#                 "assigned_tasks": [],           # task_ids assigned, not yet picked up
#                 "onboard_tasks": [],            # task_ids picked up, not yet dropped off
#                 "current_stop": None,           # {"task_id":, "kind": "pickup"|"dropoff"} or None
#                 "target_location": None,
#                 "path": [],
#                 "just_picked_up_task": None,
#             }
 
#         self.total_task_count = sum(len(b) for b in self.tasks_batches)
#         self.tasks = {}
#         self._release_pending_tasks()
 
#     def _reset_old_mode(self):
#         pass
 
#     # =========================================================================
#     # TASK RELEASE
#     # =========================================================================
 
#     def _release_pending_tasks(self):
#         for batch in self.tasks_batches:
#             for task_data in batch:
#                 task_id = str(int(task_data[0]))
#                 if task_id in self.tasks:
#                     continue
#                 if float(task_data[5]) <= self.current_time:
#                     self.tasks[task_id] = {
#                         "id": task_id,
#                         "pickup_x": float(task_data[1]),
#                         "pickup_y": float(task_data[2]),
#                         "dropoff_x": float(task_data[3]),
#                         "dropoff_y": float(task_data[4]),
#                         "release_time": float(task_data[5]),
#                         "pickup_deadline": float(task_data[6]),
#                         "est_travel_time": float(task_data[7]),
#                         "dropoff_deadline": float(task_data[8]),
#                         "is_assigned": False,
#                         "is_obsolete": False,
#                         "is_picked_up": False,
#                         "is_completed": False,
#                         "assigned_robot": None,
#                     }
 
#     # =========================================================================
#     # STEP
#     # =========================================================================
#     def _debug_robot_state(self):
#         print("\n================ ROBOT STATE ================")
#         print(f"time={self.current_time:.1f} step={self.current_step}")
 
#         for robot_id in sorted(self.robots.keys()):
#             r = self.robots[robot_id]
 
#             print(
#                 f"Robot {robot_id} | "
#                 f"cap={r['current_capacity']}/{r['max_capacity']} | "
#                 f"onboard={r['onboard_tasks']} | "
#                 f"stop={r['current_stop']} | "
#                 f"queue={r['assigned_tasks']}"
#             )
 
#             # onboard tasks
#             for tid in r["onboard_tasks"]:
#                 t = self.tasks[tid]
#                 print(
#                     f"   ONBOARD {tid}: "
#                     f"picked={t['is_picked_up']} "
#                     f"completed={t['is_completed']} "
#                     f"obsolete={t['is_obsolete']}"
#                 )
 
#             # queued tasks
#             for tid in r["assigned_tasks"]:
#                 t = self.tasks[tid]
#                 print(
#                     f"   QUEUED {tid}: "
#                     f"assigned={t['is_assigned']} "
#                     f"picked={t['is_picked_up']} "
#                     f"completed={t['is_completed']} "
#                     f"obsolete={t['is_obsolete']}"
#                 )
 
#         print("=============================================\n")
#     def step(self, actions):
#         # print(f"Step {self.current_step}: actions={actions}")
#         action_info = self._process_actions(actions)
 
#         # macro-step component accumulators
#         macro_r_comp = 0.0
#         macro_r_wait = 0.0
#         macro_r_deadline = 0.0
#         macro_r_obsolete = 0.0
 
#         self._release_pending_tasks()
#         self._update_task_deadlines()
#         self._execute_robot_movements_and_tasks()
#         self.current_time += 1.0
#         self.current_step += 1
#         reward = self._compute_rewards(action_info)
 
#         macro_r_comp += self.debug_last_r_comp
#         macro_r_wait += self.debug_last_r_wait
#         macro_r_deadline += self.debug_last_r_deadline
#         macro_r_obsolete += self.debug_last_r_obsolete
 
#         for _ in range(self.decision_interval - 1):
#             if self.current_step >= self.max_steps:
#                 break
#             self._release_pending_tasks()
#             self._update_task_deadlines()
#             self._execute_robot_movements_and_tasks()
#             self.current_time += 1.0
#             self.current_step += 1
#             reward += self._compute_rewards({})
 
#             macro_r_comp += self.debug_last_r_comp
#             macro_r_wait += self.debug_last_r_wait
#             macro_r_deadline += self.debug_last_r_deadline
#             macro_r_obsolete += self.debug_last_r_obsolete
 
#             if self._check_episode_done():
#                 break
 
#         terminated = self._check_episode_done()
#         truncated = self.current_step >= self.max_steps
#         # self._debug_robot_state()
        
#         obs = self._build_observation()
#         mask = self.action_mask()
#         # if terminated or truncated:
#             # print({
                
#             #     "candidate_ratio":
#             #         self.debug_had_candidates_count /
#             #         max(1, self.debug_decisions_total),
 
#             #     "chosen_noop_when_candidate":
#             #         self.debug_noop_chosen_count /
#             #         max(1, self.debug_had_candidates_count),
 
#             #     "forced_noop_ratio":
#             #         self.debug_noop_forced_count /
#             #         max(1, self.debug_decisions_total),
#             # })
#         info = {
#             "action_mask": mask,
#             "step": self.current_step,
#             "time": self.current_time,
#             "completed_count": self.episode_completed_count,
#             "obsolete_count": self.episode_obsolete_count,
#             "pickup_count": self.episode_pickup_count,
#             "dropoff_count": self.episode_dropoff_count,
 
#             "invalid_action_count": self.debug_last_invalid_action_count,
#             "total_action_count": self.debug_last_total_action_count,
#             "valid_action_count": self.debug_last_valid_action_count,
#             "conflict_dropped_count": self.debug_last_conflict_dropped_count,
#             "capacity_rejected_count": self.debug_last_capacity_rejected_count,
#             "mask_zero_count": self.debug_last_mask_zero_count,
 
#             # noop diagnostics
#             "noop_forced_count": self.debug_last_noop_forced_count,
#             "noop_chosen_count": self.debug_last_noop_chosen_count,
#             "had_candidates_count": self.debug_last_had_candidates_count,
#             "decisions_total": self.debug_last_decisions_total,
 
#             # IMPORTANT: macro-step sums (not last micro-step)
#             "r_comp": float(macro_r_comp),
#             "r_wait": float(macro_r_wait),
#             "r_deadline": float(macro_r_deadline),
#             "r_obsolete": float(macro_r_obsolete),
 
#             "ep_r_comp": self.debug_ep_r_comp,
#             "ep_r_wait": self.debug_ep_r_wait,
#             "ep_r_deadline": self.debug_ep_r_deadline,
#             "ep_r_obsolete": self.debug_ep_r_obsolete,
#         }
#         # print(f'completed_count:=',self.episode_completed_count, 'obsolete_count:=',self.episode_obsolete_count, 'pickup_count:=',
#         #       self.episode_pickup_count, 'dropoff_count:=',self.episode_dropoff_count)
#         return obs, reward, terminated, truncated, info
   
#     # =========================================================================
#     # ACTION PROCESSING
#     # =========================================================================
#     # =========================================================================
#     # CONFLICT RESOLUTION
#     # =========================================================================
#     # All three resolvers take the same input — a list of (dist, robot_id,
#     # task_id) requests, already filtered so every task_id refers to a task
#     # that's real/unassigned/not obsolete/not completed at proposal time —
#     # and return a list of (robot_id, task_id) winners. Capacity checking
#     # happens in the caller (_process_actions) identically for all three, so
#     # differences in outcomes are purely about *who wins a contested task*,
#     # not about capacity semantics.
 
#     def _resolve_conflicts(self, requests):
#         if not requests:
#             return []
#         if self.conflict_resolution == "greedy":
#             return self._resolve_conflicts_greedy(requests)
#         elif self.conflict_resolution == "random":
#             return self._resolve_conflicts_random(requests)
#         elif self.conflict_resolution == "hungarian":
#             return self._resolve_conflicts_hungarian(requests)
#         elif self.conflict_resolution == "hungarian_bids":
#             return self._resolve_conflicts_hungarian_bids(requests)
#         elif self.conflict_resolution == "predicted_reward":
#             return self._resolve_conflicts_predicted_reward(requests)
#         elif self.conflict_resolution == "predicted_reward_joint":
#             return self._resolve_conflicts_predicted_reward_joint(requests)
#         elif self.conflict_resolution in ("capacity", "closest_than_capacity"):
#             # closest_than_capacity is an alias for 'greedy' (distance-sort,
#             # then first-come-wins under capacity) — matches the reference
#             # repo's naming for that same behavior, not a separate
#             # implementation. 'capacity' is the "dumbest" variant: no
#             # priority ordering at all, just raw arrival order + capacity.
#             if self.conflict_resolution == "closest_than_capacity":
#                 return self._resolve_conflicts_greedy(requests)
#             return self._resolve_conflicts_capacity(requests)
#         else:
#             raise ValueError(f"Unknown conflict_resolution: {self.conflict_resolution}")
 
#     def _resolve_conflicts_greedy(self, requests):
#         """Original behavior: process requests in ascending-distance order;
#         first robot to claim a task wins it, later claims for the same task
#         are dropped as conflicts."""
#         ordered = sorted(requests, key=lambda r: r[0])
#         assigned_this_step = set()
#         winners = []
#         for dist, robot_id, task_id in ordered:
#             if task_id in assigned_this_step:
#                 self.debug_last_conflict_dropped_count += 1
#                 continue
#             assigned_this_step.add(task_id)
#             winners.append((robot_id, task_id))
#         return winners
 
#     def _resolve_conflicts_capacity(self, requests):
#         """The simplest possible resolver: NO priority ordering at all —
#         not distance, not randomized — requests are processed in whatever
#         order they were constructed in (robot iteration order from
#         _process_actions), first-come-wins per contested task, with
#         capacity as the only real constraint. Deliberately the 'dumbest'
#         baseline resolver other resolvers should be expected to beat."""
#         assigned_this_step = set()
#         winners = []
#         for dist, robot_id, task_id in requests:  # kept in original (unsorted) order
#             if task_id in assigned_this_step:
#                 self.debug_last_conflict_dropped_count += 1
#                 continue
#             assigned_this_step.add(task_id)
#             winners.append((robot_id, task_id))
#         return winners
 
#     def _resolve_conflicts_random(self, requests):
#         """Same first-come-first-served structure as greedy, but processing
#         order is a uniformly random permutation instead of ascending
#         distance — so among robots contesting the same task, the winner is
#         random rather than always the nearest one. Uses self.np_random
#         (seeded via reset(seed=...)) for reproducibility across episodes
#         run with the same seed."""
#         order = self.np_random.permutation(len(requests))
#         assigned_this_step = set()
#         winners = []
#         for idx in order:
#             dist, robot_id, task_id = requests[int(idx)]
#             if task_id in assigned_this_step:
#                 self.debug_last_conflict_dropped_count += 1
#                 continue
#             assigned_this_step.add(task_id)
#             winners.append((robot_id, task_id))
#         return winners
 
#     def _resolve_conflicts_hungarian(self, requests):
#         """Centralized optimal assignment via the Hungarian algorithm.
 
#         Under this action model each robot proposes exactly one task per
#         decision, so restricting Hungarian to literally-proposed (robot,
#         task) pairs would be mathematically identical to greedy (each robot
#         has only one edge in the bipartite graph, so there's no cross-task
#         tradeoff to exploit). To make this a genuinely different strategy,
#         a robot is eligible for ANY task proposed by ANY robot this round,
#         but ONLY if that task was also in the robot's own candidate list
#         this step (self._last_cand_task_ids) — so it can still only be
#         assigned something it could actually have seen/chosen under the
#         mask, just not necessarily the specific one it happened to pick.
#         The solver then finds the minimum-total-distance one-to-one
#         matching across that whole eligible set at once, instead of
#         resolving conflicts one task at a time.
#         """
#         robot_ids = sorted({r for _, r, _ in requests})
#         task_ids  = sorted({t for _, _, t in requests})
#         R, T = len(robot_ids), len(task_ids)
#         if R == 0 or T == 0:
#             return []
 
#         r_idx = {rid: i for i, rid in enumerate(robot_ids)}
#         t_idx = {tid: i for i, tid in enumerate(task_ids)}
 
#         # Map robot_id -> its own candidate task_ids this step, for the
#         # eligibility filter described above.
#         robot_ids_sorted_all = sorted(self.robots.keys())
#         own_candidates = {}
#         for rid in robot_ids:
#             try:
#                 r_pos = robot_ids_sorted_all.index(rid)
#                 offered = self._last_cand_task_ids[r_pos]
#             except (ValueError, IndexError):
#                 offered = []
#             own_candidates[rid] = set(t for t in offered if t is not None)
 
#         INFEASIBLE = 1e9
#         cost = np.full((R, T), INFEASIBLE, dtype=np.float64)
#         for rid in robot_ids:
#             robot = self.robots[rid]
#             eligible = own_candidates[rid]
#             for tid in task_ids:
#                 if tid not in eligible:
#                     continue
#                 task = self.tasks.get(tid)
#                 if task is None:
#                     continue
#                 d = np.sqrt(
#                     (robot["x"] - task["pickup_x"]) ** 2 +
#                     (robot["y"] - task["pickup_y"]) ** 2
#                 )
#                 cost[r_idx[rid], t_idx[tid]] = d
 
#         row_ind, col_ind = linear_sum_assignment(cost)
 
#         winners = []
#         matched_robots = set()
#         for ri, ti in zip(row_ind, col_ind):
#             if cost[ri, ti] >= INFEASIBLE:
#                 continue  # not a real eligible pairing, skip
#             winners.append((robot_ids[ri], task_ids[ti]))
#             matched_robots.add(robot_ids[ri])
 
#         # Robots that made a request but weren't matched by the solver
#         # (either genuinely infeasible or lost out in the optimal solution)
#         # count the same way an unmatched greedy/random loser would.
#         self.debug_last_conflict_dropped_count += max(0, R - len(winners))
#         return winners
 
#     def set_pending_logits(self, logits: np.ndarray) -> None:
#         """Called externally (by RTGNNPolicy.forward(), see
#         src/models/sb3_gnn_policy.py) once per macro-step, BEFORE step() is
#         called for that same decision, with this robot-step's raw candidate
#         logits — shape [R, K_max], same per-robot ordering as
#         sorted(self.robots.keys()) and same per-slot ordering as
#         self._last_cand_task_ids. Used only by
#         conflict_resolution='hungarian_bids' as bid values in place of
#         distance. Not required for 'greedy'/'random'/'hungarian'."""
#         self._pending_logits = np.asarray(logits, dtype=np.float64)
 
#     def _resolve_conflicts_hungarian_bids(self, requests):
#         """Centralized optimal assignment via the Hungarian algorithm, using
#         the POLICY'S OWN LOGITS as bid values instead of distance — i.e. a
#         genuine auction: each robot's bid for a task is how strongly its
#         policy already wants that task (higher logit = stronger bid), and
#         the solver finds the assignment that maximizes total bid value
#         (equivalently: minimizes total NEGATIVE bid) across every robot
#         that made a request and every task any of them proposed, subject to
#         the same eligibility rule as _resolve_conflicts_hungarian (a robot
#         can only be assigned a task that was actually in its own candidate
#         list this step).
 
#         Requires set_pending_logits() to have been called this step (see
#         that method's docstring) — raises clearly if not, rather than
#         silently falling back to something else, since a silent fallback
#         would make it easy to not notice bids were never actually wired up.
#         """
#         if self._pending_logits is None:
#             raise RuntimeError(
#                 "conflict_resolution='hungarian_bids' requires set_pending_logits() "
#                 "to be called each step before step(actions) — see "
#                 "RTGNNPolicy.forward() in src/models/sb3_gnn_policy.py, and make "
#                 "sure model.policy._bid_env is wired to this env's VecEnv after "
#                 "construction (see train_ppo.py)."
#             )
 
#         robot_ids = sorted({r for _, r, _ in requests})
#         task_ids  = sorted({t for _, _, t in requests})
#         R, T = len(robot_ids), len(task_ids)
#         if R == 0 or T == 0:
#             return []
 
#         r_idx = {rid: i for i, rid in enumerate(robot_ids)}
#         t_idx = {tid: i for i, tid in enumerate(task_ids)}
 
#         robot_ids_sorted_all = sorted(self.robots.keys())
 
#         # Map robot_id -> {task_id: bid_logit}, using that robot's OWN
#         # candidate list crossed with that SAME robot's OWN logits for
#         # those same slots (both indexed identically by slot position).
#         own_bids = {}
#         for rid in robot_ids:
#             try:
#                 r_pos = robot_ids_sorted_all.index(rid)
#                 offered = self._last_cand_task_ids[r_pos]
#                 logits_row = self._pending_logits[r_pos]  # [K_max]
#             except (ValueError, IndexError):
#                 offered, logits_row = [], None
#             bid_map = {}
#             if logits_row is not None:
#                 for slot, tid_at_slot in enumerate(offered):
#                     if tid_at_slot is not None and slot < len(logits_row):
#                         bid_map[tid_at_slot] = float(logits_row[slot])
#             own_bids[rid] = bid_map
 
#         INFEASIBLE = 1e9
#         cost = np.full((R, T), INFEASIBLE, dtype=np.float64)
#         for rid in robot_ids:
#             bid_map = own_bids[rid]
#             for tid in task_ids:
#                 if tid in bid_map:
#                     cost[r_idx[rid], t_idx[tid]] = -bid_map[tid]  # maximize bid == minimize -bid
 
#         row_ind, col_ind = linear_sum_assignment(cost)
 
#         winners = []
#         for ri, ti in zip(row_ind, col_ind):
#             if cost[ri, ti] >= INFEASIBLE:
#                 continue
#             winners.append((robot_ids[ri], task_ids[ti]))
 
#         self.debug_last_conflict_dropped_count += max(0, R - len(winners))
#         return winners
 
#     def _resolve_conflicts_hungarian_with_bid_fn(self, requests, bid_fn):
#         """Shared centralized-assignment machinery for any resolver that
#         scores (robot, task) pairs with a pluggable bid_fn(robot_id,
#         task_id) -> float, instead of distance or policy logits. Used by
#         both _resolve_conflicts_predicted_reward and
#         _resolve_conflicts_predicted_reward_joint. Same eligibility rule
#         as the other hungarian variants: a robot can only be matched to a
#         task that was actually in its own candidate list this step."""
#         robot_ids = sorted({r for _, r, _ in requests})
#         task_ids  = sorted({t for _, _, t in requests})
#         R, T = len(robot_ids), len(task_ids)
#         if R == 0 or T == 0:
#             return []
 
#         r_idx = {rid: i for i, rid in enumerate(robot_ids)}
#         t_idx = {tid: i for i, tid in enumerate(task_ids)}
#         robot_ids_sorted_all = sorted(self.robots.keys())
 
#         eligible_tasks = {}
#         for rid in robot_ids:
#             try:
#                 r_pos = robot_ids_sorted_all.index(rid)
#                 offered = self._last_cand_task_ids[r_pos]
#             except (ValueError, IndexError):
#                 offered = []
#             eligible_tasks[rid] = {t for t in offered if t is not None}
 
#         INFEASIBLE = 1e9
#         cost = np.full((R, T), INFEASIBLE, dtype=np.float64)
#         for rid in robot_ids:
#             for tid in task_ids:
#                 if tid not in eligible_tasks[rid]:
#                     continue
#                 bid = bid_fn(rid, tid)
#                 if bid is None or not np.isfinite(bid):
#                     continue
#                 cost[r_idx[rid], t_idx[tid]] = -bid  # maximize bid == minimize -bid
 
#         row_ind, col_ind = linear_sum_assignment(cost)
 
#         winners = []
#         for ri, ti in zip(row_ind, col_ind):
#             if cost[ri, ti] >= INFEASIBLE:
#                 continue
#             winners.append((robot_ids[ri], task_ids[ti]))
 
#         self.debug_last_conflict_dropped_count += max(0, R - len(winners))
#         return winners
 
#     def _resolve_conflicts_predicted_reward(self, requests):
#         """Centralized assignment using predict_candidate_score() (single-
#         candidate simulated pickup/dropoff scoring) as bid values —
#         matches the reference repo's 'predicted_reward' resolver: the
#         resolver reuses the exact same scoring function as the
#         predicted_reward baseline/proposer, just as the auction's bids
#         instead of a per-robot ranking."""
#         return self._resolve_conflicts_hungarian_with_bid_fn(
#             requests, lambda rid, tid: self.predict_candidate_score(rid, tid)
#         )
 
#     def _resolve_conflicts_predicted_reward_joint(self, requests):
#         """Same as _resolve_conflicts_predicted_reward, but bids are the
#         MARGINAL score (predict_candidate_score_joint: R_after - R_before
#         over the robot's whole route) — matches the reference repo's
#         'predicted_reward_joint' resolver."""
#         return self._resolve_conflicts_hungarian_with_bid_fn(
#             requests, lambda rid, tid: self.predict_candidate_score_joint(rid, tid)
#         )
 
#     def _process_actions(self, actions) -> Dict:
#         """
#         Uses _last_cand_task_ids from observation-time snapshot to prevent
#         index mismatch between policy output and candidate list.
#         """
#         robot_ids = sorted(self.robots.keys())
#         requests = []
 
#         invalid_action_count = 0
#         valid_action_count = 0
#         conflict_dropped_count = 0
#         capacity_rejected_count = 0
 
#         step_noop_forced = 0
#         step_noop_chosen = 0
#         step_had_candidates = 0
#         step_decisions = 0
 
#         for r_idx, action in enumerate(actions):
 
#             if r_idx >= len(robot_ids):
#                 break
 
#             offered = self._last_cand_task_ids[r_idx]
#             # print(f"Step {self.current_step}: Robot {robot_ids[r_idx]} action={action}, cands={offered}, noop_index={self._noop_index}")
#             had_candidates = any(t is not None for t in offered)
#             is_noop = (int(action) == self._noop_index)
 
#             # ---------- episode totals ----------
#             self.debug_decisions_total += 1
#             step_decisions += 1
 
#             if had_candidates:
#                 self.debug_had_candidates_count += 1
#                 step_had_candidates += 1
 
#             if is_noop:
#                 if had_candidates:
#                     self.debug_noop_chosen_count += 1
#                     step_noop_chosen += 1
#                 else:
#                     self.debug_noop_forced_count += 1
#                     step_noop_forced += 1
#                 continue
 
#             robot_id = robot_ids[r_idx]
#             cands = self._last_cand_task_ids[r_idx]
 
#             if int(action) >= len(cands) or cands[int(action)] is None:
#                 invalid_action_count += 1
#                 continue
 
#             task_id = cands[int(action)]
#             task = self.tasks.get(task_id)
 
#             if (
#                 task is None
#                 or task.get("is_assigned")
#                 or task.get("is_obsolete")
#                 or task.get("is_completed")
#             ):
#                 invalid_action_count += 1
#                 continue
 
#             robot = self.robots[robot_id]
 
#             dist = np.sqrt(
#                 (robot["x"] - task["pickup_x"]) ** 2
#                 + (robot["y"] - task["pickup_y"]) ** 2
#             )
 
#             requests.append((dist, robot_id, task_id))
 
#         # ---------------------------------------------------
#         # Resolve conflicts
#         # ---------------------------------------------------
#         # requests.sort()
#         # print(f"Step {self.current_step}: {len(requests)} , {requests},requests before conflict resolution")
#         winners = self._resolve_conflicts(requests)
 
#         assigned_this_step = set()
#         action_info = {}
#         # print(assigned_this_step, winners, "winners after conflict resolution")
#         # for _, robot_id, task_id in requests:
 
#         #     if task_id in assigned_this_step:
#         #         conflict_dropped_count += 1
#         #         continue
 
#         #     robot = self.robots[robot_id]
#         #     task = self.tasks.get(task_id)
 
#         #     if (
#         #         task is None
#         #         or task.get("is_assigned")
#         #         or task.get("is_obsolete")
#         #         or task.get("is_completed")
#         #     ):
#         #         invalid_action_count += 1
#         #         continue
 
#         #     if self.capacity_method == "assigned":
#         #         total_committed = (
#         #             len(robot["onboard_tasks"])
#         #             + len(robot["assigned_tasks"])
#         #         )
#         #     else:
#         #         total_committed = len(robot["onboard_tasks"])
 
#         #     if total_committed >= self.max_robot_capacity:
#         #         capacity_rejected_count += 1
#         #         continue
 
#         #     robot["assigned_tasks"].append(task_id)
#         #     task["is_assigned"] = True
#         #     task["assigned_robot"] = robot_id
 
#         #     assigned_this_step.add(task_id)
#         #     print(assigned_this_step)
#         #     action_info[robot_id] = {"assigned_task": task_id}
#         #     print(f"Step {self.current_step}: Robot {robot_id} assigned to task {task_id}")
#         #     valid_action_count += 1
#         assigned_this_step = set()
#         action_info = {}
 
#         winner_set = set(winners)
 
#         for robot_id, task_id in winners:
 
#             robot = self.robots[robot_id]
#             task = self.tasks.get(task_id)
 
#             if (
#                 task is None
#                 or task.get("is_assigned")
#                 or task.get("is_obsolete")
#                 or task.get("is_completed")
#             ):
#                 invalid_action_count += 1
#                 continue
 
#             if self.capacity_method == "assigned":
#                 total_committed = (
#                     len(robot["onboard_tasks"])
#                     + len(robot["assigned_tasks"])
#                 )
#             else:
#                 total_committed = len(robot["onboard_tasks"])
 
#             if total_committed >= self.max_robot_capacity:
#                 capacity_rejected_count += 1
#                 continue
 
#             robot["assigned_tasks"].append(task_id)
#             task["is_assigned"] = True
#             task["assigned_robot"] = robot_id
 
#             assigned_this_step.add(task_id)
 
#             action_info[robot_id] = {
#                 "assigned_task": task_id
#             }
 
#             valid_action_count += 1
#         conflict_dropped_count = len(requests) - len(winners)
#         # ---------------------------------------------------
#         # Per-step diagnostics
#         # ---------------------------------------------------
#         self.debug_last_invalid_action_count = invalid_action_count
#         self.debug_last_valid_action_count = valid_action_count
#         self.debug_last_conflict_dropped_count = conflict_dropped_count
#         self.debug_last_capacity_rejected_count = capacity_rejected_count
 
#         self.debug_last_noop_forced_count = step_noop_forced
#         self.debug_last_noop_chosen_count = step_noop_chosen
#         self.debug_last_had_candidates_count = step_had_candidates
#         self.debug_last_decisions_total = step_decisions
 
#         # ---------------------------------------------------
#         # Episode diagnostics
#         # ---------------------------------------------------
#         self.debug_invalid_action_count += invalid_action_count
#         self.debug_valid_action_count += valid_action_count
#         self.debug_conflict_dropped_count += conflict_dropped_count
#         self.debug_capacity_rejected_count += capacity_rejected_count
 
#         action_info["_diag"] = {
#             "invalid_action_count": invalid_action_count,
#             "valid_action_count": valid_action_count,
#             "conflict_dropped_count": conflict_dropped_count,
#             "capacity_rejected_count": capacity_rejected_count,
#             "noop_forced_count": step_noop_forced,
#             "noop_chosen_count": step_noop_chosen,
#             "had_candidates_count": step_had_candidates,
#             "decisions_total": step_decisions,
#         }
#         # print(step_noop_chosen, step_noop_forced, step_had_candidates, step_decisions)
#         return action_info
#     def _process_actionsold(self, actions) -> Dict:
#         """
#         Uses _last_cand_task_ids from observation-time snapshot to prevent
#         index mismatch between policy output and candidate list.
#         """
#         robot_ids = sorted(self.robots.keys())
#         requests = []
 
#         invalid_action_count = 0
#         valid_action_count = 0
#         conflict_dropped_count = 0
#         total_action_count = 0
#         capacity_rejected_count = 0
 
#         step_noop_forced = 0
#         step_noop_chosen = 0
#         step_had_candidates = 0
#         step_decisions = 0
 
#         act_arr = np.asarray(actions).flatten()
 
#         # for r_idx, action in enumerate(act_arr):
#         #     if r_idx >= len(robot_ids):
#         #         break
 
#         #     total_action_count += 1
#         #     a = int(action)
 
#         #     if a == self._noop_index:
#         #         continue
 
#         #     robot_id = robot_ids[r_idx]
#         #     cands = self._last_cand_task_ids[r_idx] if r_idx < len(self._last_cand_task_ids) else []
 
#         #     if a < 0 or a >= len(cands):
#         #         invalid_action_count += 1
#         #         continue
 
#         #     task_id = cands[a]
#         for r_idx, action in enumerate(actions):
#             # total_action_count += 1
#             # print(f"Step {self.current_step}: Robot {robot_ids[r_idx]} action={action}, cands={self._last_cand_task_ids[r_idx]}, noop_index={self._noop_index}", {int(action)})
#             offered = self._last_cand_task_ids[r_idx]
#             # True if at least one task candidate exists
#             had_candidates = any(t is not None for t in offered)
#             is_noop = (int(action) == self._noop_index)
 
#             self.debug_decisions_total += 1
 
#             if had_candidates:
#                 self.debug_had_candidates_count += 1
 
#             if is_noop:
#                 if had_candidates:
#                     self.debug_noop_chosen_count += 1
#                 else:
#                     self.debug_noop_forced_count += 1
 
#                 continue
 
#             # if int(action) == self._noop_index:
#             #     continue
#             if r_idx >= len(robot_ids):
#                 break
#             robot_id = robot_ids[r_idx]
#             cands = self._last_cand_task_ids[r_idx]     # <-- use cached list, not recomputed
#             if int(action) >= len(cands) or cands[int(action)] is None:
#                 continue
#             task_id = cands[int(action)]
#             task = self.tasks.get(task_id)
 
#             if task is None or task.get("is_assigned") or task.get("is_obsolete") or task.get("is_completed"):
#                 invalid_action_count += 1
#                 continue
 
#             robot = self.robots[robot_id]
#             dist = np.sqrt(
#                 (robot["x"] - task["pickup_x"]) ** 2 +
#                 (robot["y"] - task["pickup_y"]) ** 2
#             )
#             requests.append((dist, robot_id, task_id))
 
#         requests.sort()
#         assigned_this_step = set()
#         action_info = {}
 
#         for _dist, robot_id, task_id in requests:
#             if task_id in assigned_this_step:
#                 conflict_dropped_count += 1
#                 continue
 
#             robot = self.robots[robot_id]
#             task = self.tasks.get(task_id)
#             if task is None or task.get("is_assigned") or task.get("is_obsolete") or task.get("is_completed"):
#                 invalid_action_count += 1
#                 continue
 
#             # capacity_method="assigned": count onboard + queued-not-yet-picked
#             #   (conservative — reserves a seat for every task promised, even
#             #   ones not yet physically onboard)
#             # capacity_method="pickup": count onboard only (len(onboard_tasks))
#             #   (permissive — allows queuing more pickups than max_capacity as
#             #   long as physical onboard load never exceeds it; relies on
#             #   _assign_next_stop's room_to_pickup check to enforce the real
#             #   physical limit at pickup time)
#             if self.capacity_method == "assigned":
#                 total_committed = (
#                     len(robot["onboard_tasks"])
#                     + len(robot["assigned_tasks"])
#                 )
#             else:
#                 total_committed = len(robot["onboard_tasks"])
 
#             if total_committed >= self.max_robot_capacity:
#                 capacity_rejected_count += 1
#                 continue
 
#             robot["assigned_tasks"].append(task_id)
#             task["is_assigned"] = True
#             task["assigned_robot"] = robot_id
#             assigned_this_step.add(task_id)
#             action_info[robot_id] = {"assigned_task": task_id}
#             print(f"Step {self.current_step}: Robot {robot_id} assigned to Task {task_id}")
#         valid_action_count = len(action_info)
 
#         self.debug_last_invalid_action_count = int(invalid_action_count)
#         self.debug_last_total_action_count = int(total_action_count)
#         self.debug_last_valid_action_count = int(valid_action_count)
#         self.debug_last_conflict_dropped_count = int(conflict_dropped_count)
#         self.debug_last_capacity_rejected_count = int(capacity_rejected_count)
 
#         self.debug_invalid_action_count += int(invalid_action_count)
#         self.debug_total_action_count += int(total_action_count)
#         self.debug_valid_action_count += int(valid_action_count)
#         self.debug_conflict_dropped_count += int(conflict_dropped_count)
#         self.debug_capacity_rejected_count += int(capacity_rejected_count)
 
#         action_info["_diag"] = {
#             "invalid_action_count": int(invalid_action_count),
#             "total_action_count": int(total_action_count),
#             "valid_action_count": int(valid_action_count),
#             "conflict_dropped_count": int(conflict_dropped_count),
#             "capacity_rejected_count": int(capacity_rejected_count),
#         }
#         # print(f"Step {self.current_step}: invalid={invalid_action_count}, total={total_action_count}, valid={valid_action_count}, conflict_dropped={conflict_dropped_count}, capacity_rejected={capacity_rejected_count}")
#         return action_info
 
#     # =========================================================================
#     # CANDIDATE TASKS
#     # =========================================================================
# # src/environment/environment.py
 
#     def _remaining_capacity(self, robot_id) -> int:
#         """Free 'seats' on this robot right now.
#         capacity_method='assigned': onboard + queued-not-yet-picked both count
#             (conservative — matches candidate gating with _process_actions).
#         capacity_method='pickup': onboard only counts (permissive)."""
#         robot = self.robots.get(str(robot_id))
#         if robot is None:
#             return 0
 
#         if self.capacity_method == "assigned":
#             committed = len(robot["onboard_tasks"]) + len(robot["assigned_tasks"])
#         else:   # pickup
#             committed = len(robot["onboard_tasks"])
 
#         return max(0, robot["max_capacity"] - committed)
 
 
#     def _get_candidate_tasks(self, robot_id) -> List[str]:
#         """Return up to K_max available tasks within vicinity_m of the robot,
#         sorted by ascending Euclidean distance to pickup location.
#         Gated by the robot's own remaining capacity — a full robot gets an
#         empty candidate list (forced no-op), matching the reference adapter.
#         """
#         assigned = 0
#         completed = 0
#         obsolete = 0
#         future = 0
#         deadline = 0
#         far = 0
#         accepted = 0
#         robot = self.robots.get(str(robot_id))
#         if robot is None:
#             return []
 
#         if self._remaining_capacity(robot_id) <= 0:
#             return []
 
#         candidates = []
#         for task_id, task in self.tasks.items():
#             if task["is_assigned"]:
#                 assigned += 1
                
 
#             if task["is_completed"]:
#                 completed += 1
                
 
#             if task["is_obsolete"]:
#                 obsolete += 1
                
 
#             if task["release_time"] > self.current_time:
#                 future += 1
#                 continue
 
#             if task["pickup_deadline"] <= self.current_time:
#                 deadline += 1
#                 continue
#             dist = np.sqrt(
#                 (robot["x"] - task["pickup_x"]) ** 2 +
#                 (robot["y"] - task["pickup_y"]) ** 2
#             )
#             if task["is_assigned"] or task["is_completed"] or task["is_obsolete"]:
#                 continue
#             if dist > self.vicinity_m:
#                 far += 1
#                 continue
#             if dist <= self.vicinity_m:
#                 accepted += 1
#                 candidates.append((dist, task_id))
 
#         candidates.sort()
#         top_k = candidates[: self.K_max]

#         # candidates_sorting='randomized' (matches reference config) keeps
#         # the SAME selection (still the K_max closest — this only changes
#         # which SLOT each one lands in, not which tasks are visible at
#         # all), but shuffles slot order so the GNN can't learn a lazy
#         # "prefer low slot index" shortcut instead of actually reasoning
#         # about each candidate's own features. Uses self.np_random (seeded
#         # via reset(seed=...)) for reproducibility. NOTE: any baseline that
#         # assumes slot 0 == nearest (e.g. distance-based tie-breaks) needs
#         # to look up actual distance explicitly instead when this is on —
#         # see eval_baseline.py's pickup_deadline_distance_action, which
#         # already does this correctly via task lookup rather than slot
#         # position.
#         if self.candidates_sorting == "randomized" and len(top_k) > 1:
#             order = self.np_random.permutation(len(top_k))
#             top_k = [top_k[i] for i in order]

#         # print(
#         #     robot_id,
#         #     "accepted", accepted,
#         #     "assigned", assigned,s
#         #     "completed", completed,
#         #     "obsolete", obsolete,
#         #     "future", future,
#         #     "deadline", deadline,
#         #     "far", far,
#         #     "candidates", len(candidates)
#         # )
#         return [tid for _, tid in top_k]
#     def _get_candidate_tasks_no_capacity_check(self, robot_id) -> List[str]:
#         robot = self.robots.get(str(robot_id))
#         if robot is None:
#             return []
 
#         candidates = []
#         for task_id, task in self.tasks.items():
#             if task.get("is_assigned") or task.get("is_completed") or task.get("is_obsolete"):
#                 continue
#             if task.get("release_time", 0) > self.current_time:
#                 continue
#             if task.get("pickup_deadline", float("inf")) <= self.current_time:
#                 continue
#             dist = np.sqrt(
#                 (robot["x"] - task["pickup_x"]) ** 2 +
#                 (robot["y"] - task["pickup_y"]) ** 2
#             )
#             if dist <= self.vicinity_m:
#                 candidates.append((dist, task_id))
 
#         candidates.sort()
#         return [tid for _, tid in candidates[: self.K_max]]
 
#     # =========================================================================
#     # TASK LIFECYCLE — deadlines
#     # =========================================================================
 
#     def _update_task_deadlines(self):
#         """
#         Deadline handling policy:
#         - Not picked up yet: pickup deadline expiry => obsolete.
#         - Already picked up: NEVER obsolete; keep delivery, penalize lateness in reward.
#         """
#         for task_id, task in list(self.tasks.items()):
#             if task.get("is_completed") or task.get("is_obsolete"):
#                 continue
 
#             if not task.get("is_picked_up"):
#                 expired_pickup = task.get("pickup_deadline", float("inf")) <= self.current_time
#                 if not expired_pickup:
#                     continue
 
#                 task["is_obsolete"] = True
#                 self.episode_obsolete_count += 1
 
#                 assigned_id = task.get("assigned_robot")
#                 if assigned_id and assigned_id in self.robots:
#                     robot = self.robots[assigned_id]
 
#                     if task_id in robot["assigned_tasks"]:
#                         robot["assigned_tasks"].remove(task_id)
 
#                     stop = robot["current_stop"]
#                     if stop is not None and stop["task_id"] == task_id and stop["kind"] == "pickup":
#                         robot["current_stop"]    = None
#                         robot["target_location"] = None
#                         robot["path"]            = []
 
#             else:
#                 # Picked up tasks are kept alive; lateness handled at dropoff reward.
#                 pass
 
#     # =========================================================================
#     # ROBOT MOVEMENT — A* path following
#     # =========================================================================
 
#     def _execute_robot_movements_and_tasks(self):
#         for robot_id, robot in self.robots.items():
#             if robot["current_stop"] is None:
#                 self._assign_next_stop(robot)
 
#             if robot["current_stop"] is not None:
#                 self._move_robot_toward_target(robot_id)
 
#     def _assign_next_stop(self, robot):
#         """
#         Nearest-stop routing policy for multi-capacity robots.
 
#         Candidate next stops:
#           - a pickup for any task in assigned_tasks, but only if the robot
#             currently has room to carry another (len(onboard_tasks) < max_capacity)
#           - a dropoff for any task in onboard_tasks
 
#         The nearest candidate (Euclidean distance from current position) is
#         chosen, letting a robot interleave pickups and dropoffs instead of
#         finishing one task before starting the next.
#         """
#         candidates = []  # (dist, kind, task_id, location)
 
#         room_to_pickup = len(robot["onboard_tasks"]) < robot["max_capacity"]
#         if room_to_pickup:
#             for task_id in robot["assigned_tasks"]:
#                 task = self.tasks.get(task_id)
#                 if task is None or task.get("is_obsolete"):
#                     continue
#                 loc  = (task["pickup_x"], task["pickup_y"])
#                 dist = np.sqrt((robot["x"] - loc[0]) ** 2 + (robot["y"] - loc[1]) ** 2)
#                 candidates.append((dist, "pickup", task_id, loc))
 
#         for task_id in robot["onboard_tasks"]:
#             task = self.tasks.get(task_id)
#             if task is None:
#                 continue
#             loc  = (task["dropoff_x"], task["dropoff_y"])
#             dist = np.sqrt((robot["x"] - loc[0]) ** 2 + (robot["y"] - loc[1]) ** 2)
#             candidates.append((dist, "dropoff", task_id, loc))
 
#         if not candidates:
#             robot["current_stop"]    = None
#             robot["target_location"] = None
#             robot["path"]            = []
#             return
 
#         candidates.sort(key=lambda c: c[0])
#         _, kind, task_id, loc = candidates[0]
#         robot["current_stop"]    = {"task_id": task_id, "kind": kind}
#         robot["target_location"] = loc
#         robot["path"]            = []
 
#     def _move_robot_toward_target(self, robot_id: str):
#         robot = self.robots[robot_id]
#         target_x, target_y = robot["target_location"]
 
#         if not robot["path"]:
#             start = (int(round(robot["y"])), int(round(robot["x"])))
#             goal = (int(round(target_y)), int(round(target_x)))
#             if start != goal:
#                 found, path = self.planner.get_plan(start, goal)
#                 if found and path and len(path) > 1:
#                     robot["path"] = list(path[1:])
#                 else:
#                     robot["path"] = []
 
#         if robot["path"]:
#             next_row, next_col = robot["path"][0]
#             dx = float(next_col) - robot["x"]
#             dy = float(next_row) - robot["y"]
#             dist = np.sqrt(dx * dx + dy * dy)
 
#             if dist <= self.movement_speed:
#                 robot["x"] = float(next_col)
#                 robot["y"] = float(next_row)
#                 robot["path"].pop(0)
#             else:
#                 robot["x"] += (dx / dist) * self.movement_speed
#                 robot["y"] += (dy / dist) * self.movement_speed
#             return
 
#         dx = target_x - robot["x"]
#         dy = target_y - robot["y"]
#         dist = np.sqrt(dx * dx + dy * dy)
 
#         if dist > 0.5:
#             move = min(self.movement_speed, dist)
#             robot["x"] += (dx / dist) * move
#             robot["y"] += (dy / dist) * move
#             return
 
#         # ── Arrival ───────────────────────────────────────────────────────
#         stop = robot["current_stop"]
#         if stop is None:
#             return
#         task_id = stop["task_id"]
#         task    = self.tasks.get(task_id)
#         if task is None:
#             robot["current_stop"]    = None
#             robot["target_location"] = None
#             robot["path"]            = []
#             return
 
#         if stop["kind"] == "pickup":
#             if task_id in robot["assigned_tasks"]:
#                 robot["assigned_tasks"].remove(task_id)
#             robot["onboard_tasks"].append(task_id)
#             robot["current_capacity"]  = len(robot["onboard_tasks"])
#             task["is_picked_up"]       = True
#             self.episode_pickup_count += 1
#             task["pickup_time"]        = self.current_time
#             robot["just_picked_up_task"] = task_id
#             # Stop is cleared; _assign_next_stop() picks the next pickup or
#             # dropoff (whichever is nearest) next tick — this is what lets
#             # onboard_tasks hold more than one task at a time.
#             robot["current_stop"]      = None
#             robot["target_location"]   = None
#             robot["path"]              = []
 
#         elif stop["kind"] == "dropoff":
#             if task_id in robot["onboard_tasks"]:
#                 robot["onboard_tasks"].remove(task_id)
#             robot["current_capacity"]     = len(robot["onboard_tasks"])
#             task["dropoff_time"]          = self.current_time
#             task["is_completed"]          = True
#             self.episode_dropoff_count   += 1
#             self.episode_completed_count += 1
#             robot["current_stop"]         = None
#             robot["target_location"]      = None
#             robot["path"]                 = []
 
#     # =========================================================================
#     # OBSERVATION
#     # =========================================================================
 
#     def _build_observation(self) -> Dict:
#         robot_ids = sorted(self.robots.keys())
#         if len(robot_ids) < self.num_robots:
#             robot_ids += [None] * (self.num_robots - len(robot_ids))
#         robot_ids = robot_ids[: self.num_robots]
 
#         candidate_lists = [
#             self._get_candidate_tasks(rid) if rid is not None else []
#             for rid in robot_ids
#         ]
 
#         obs_dict, cand_task_ids = build_padded_ego_batch(
#             robots=robot_ids,
#             robots_dict=self.robots,
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
 
#         self._last_cand_task_ids = cand_task_ids
#         return obs_dict
 
#     # =========================================================================
#     # ROUTE-INSERTION PREDICTION (predicted_reward / predicted_reward_joint)
#     # =========================================================================
#     #
#     # These simulate "what would happen if we inserted this candidate task
#     # into this robot's route" WITHOUT mutating any real robot/task state —
#     # used by the predicted_reward baseline/resolver (single-candidate
#     # score) and predicted_reward_joint (marginal score: how much does
#     # adding this candidate change the score of the WHOLE route, not just
#     # the candidate itself).
#     #
#     # The simulated walk deliberately mirrors _assign_next_stop()'s
#     # nearest-next-stop rule exactly (greedy route construction), so the
#     # prediction is consistent with what the robot would actually do if
#     # this candidate were assigned. One documented approximation: the walk
#     # uses straight-line distance / movement_speed for travel time, while
#     # real execution follows an A* path (_move_robot_toward_target) that
#     # can be longer if obstacles are in the way — so predictions are an
#     # estimate, not an exact replay, the same caveat the reference
#     # implementation's own travel-time estimator carries.
 
#     def _simulate_route_with_candidate(self, robot_id, candidate_task_id):
#         """Greedy nearest-next-stop walk over (robot's committed stops +
#         candidate's pickup/dropoff), starting from the robot's current
#         position/time. Returns (predicted_pickup_time, predicted_dropoff_time)
#         for the candidate specifically, or (None, None) if the candidate
#         never gets reached (e.g. capacity never frees up in the simulated
#         walk before stops run out)."""
#         robot = self.robots[robot_id]
#         candidate = self.tasks.get(candidate_task_id)
#         if candidate is None:
#             return None, None
 
#         task_locs, pending, onboard = self._build_walk_state(robot, extra_pending={candidate_task_id: candidate})
 
#         return self._walk_stops(
#             start_x=robot["x"], start_y=robot["y"], start_time=self.current_time,
#             max_capacity=robot["max_capacity"],
#             pending_pickup_ids=pending, onboard_ids=onboard,
#             task_locs=task_locs, track_task_id=candidate_task_id,
#         )
 
#     def _build_walk_state(self, robot, extra_pending=None):
#         """Build the (task_locs, pending_pickup_ids, onboard_ids) inputs
#         _walk_stops needs, from a robot's real committed state plus
#         optionally one extra not-yet-assigned candidate task. task_locs
#         maps every relevant task_id -> {"pickup": (x,y), "dropoff": (x,y)}
#         so _walk_stops can look up a task's dropoff location the moment
#         its pickup is visited, even though it wasn't in an initial fixed
#         stop list."""
#         task_locs = {}
#         pending = set()
#         onboard = set()
 
#         for tid in robot["assigned_tasks"]:
#             t = self.tasks.get(tid)
#             if t is None or t.get("is_obsolete"):
#                 continue
#             task_locs[tid] = {"pickup": (t["pickup_x"], t["pickup_y"]), "dropoff": (t["dropoff_x"], t["dropoff_y"])}
#             pending.add(tid)
 
#         for tid in robot["onboard_tasks"]:
#             t = self.tasks.get(tid)
#             if t is None:
#                 continue
#             task_locs[tid] = {"pickup": (t["pickup_x"], t["pickup_y"]), "dropoff": (t["dropoff_x"], t["dropoff_y"])}
#             onboard.add(tid)
 
#         if extra_pending:
#             for tid, t in extra_pending.items():
#                 task_locs[tid] = {"pickup": (t["pickup_x"], t["pickup_y"]), "dropoff": (t["dropoff_x"], t["dropoff_y"])}
#                 pending.add(tid)
 
#         return task_locs, pending, onboard
 
#     def _walk_stops(self, start_x, start_y, start_time, max_capacity, pending_pickup_ids, onboard_ids, task_locs, track_task_id):
#         """Shared greedy nearest-next-stop walk, mirroring
#         _assign_next_stop()'s real dynamic behavior: eligible next stops
#         are re-derived every iteration from CURRENT pending/onboard sets
#         (pickups for pending tasks, if there's room; dropoffs for onboard
#         tasks) — NOT a fixed precomputed list. Once a task's pickup is
#         visited it moves from pending to onboard, making its dropoff
#         eligible on the NEXT iteration, exactly like real execution. This
#         is what makes multi-task routes (a not-yet-picked-up task whose
#         dropoff hasn't happened yet) score correctly instead of getting
#         stuck with no dropoff stop ever appearing.
 
#         Returns (pickup_time, dropoff_time) for track_task_id, or (None,
#         None) if it's never reached in the simulated walk."""
#         cur_x, cur_y, cur_time = start_x, start_y, start_time
#         pending = set(pending_pickup_ids)
#         onboard = set(onboard_ids)
#         pickup_time = dropoff_time = None
 
#         while pending or onboard:
#             room = len(onboard) < max_capacity
#             candidates = []
#             if room:
#                 for tid in pending:
#                     x, y = task_locs[tid]["pickup"]
#                     candidates.append((tid, "pickup", x, y))
#             for tid in onboard:
#                 x, y = task_locs[tid]["dropoff"]
#                 candidates.append((tid, "dropoff", x, y))
 
#             if not candidates:
#                 break
 
#             nxt = min(candidates, key=lambda s: (s[2] - cur_x) ** 2 + (s[3] - cur_y) ** 2)
#             dist = float(np.sqrt((nxt[2] - cur_x) ** 2 + (nxt[3] - cur_y) ** 2))
#             cur_time += dist / max(1e-9, self.movement_speed)
#             cur_x, cur_y = nxt[2], nxt[3]
 
#             tid, kind = nxt[0], nxt[1]
#             if kind == "pickup":
#                 pending.discard(tid)
#                 onboard.add(tid)
#                 if tid == track_task_id:
#                     pickup_time = cur_time
#             else:
#                 onboard.discard(tid)
#                 if tid == track_task_id:
#                     dropoff_time = cur_time
#                     break
 
#         return pickup_time, dropoff_time
 
#     def _score_predicted_times(self, task, pickup_time, dropoff_time):
#         """Same-shaped scoring formula as _compute_rewards, but evaluated
#         against PREDICTED (simulated) times instead of actual ones — see
#         module docstring above. valid_completion is binary (matches the
#         reference implementation's predicted_reward scoring exactly),
#         which is simpler than the continuous lateness penalty
#         _compute_rewards uses for tasks that actually complete late."""
#         if pickup_time is None or dropoff_time is None:
#             return float("-inf")
 
#         WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
#         DEADLINE_CAP = max(1.0, float(self.max_travel_delay_s))
 
#         wait = max(0.0, pickup_time - task["release_time"])
#         norm_wait = min(wait, WAIT_CAP) / WAIT_CAP
 
#         ride_time = max(0.0, dropoff_time - pickup_time)
#         excess_ride = max(0.0, ride_time - task.get("est_travel_time", 0.0))
#         norm_excess = min(excess_ride, DEADLINE_CAP) / DEADLINE_CAP
 
#         valid_completion = (
#             pickup_time <= task.get("pickup_deadline", float("inf"))
#             and dropoff_time <= task.get("dropoff_deadline", float("inf"))
#         )
 
#         return (
#             self.W_COMP * float(valid_completion)
#             - self.W_WAIT * norm_wait
#             - self.W_DEADLINE * norm_excess
#         )
 
#     def predict_candidate_score(self, robot_id, candidate_task_id):
#         """predicted_reward: score of inserting candidate_task_id into
#         robot_id's route, based on the candidate's OWN predicted
#         pickup/dropoff times only (not how it affects other already-
#         committed tasks — see predict_candidate_score_joint for that)."""
#         candidate = self.tasks.get(candidate_task_id)
#         if candidate is None:
#             return float("-inf")
#         pickup_time, dropoff_time = self._simulate_route_with_candidate(robot_id, candidate_task_id)
#         return self._score_predicted_times(candidate, pickup_time, dropoff_time)
 
#     def predict_candidate_score_joint(self, robot_id, candidate_task_id):
#         """predicted_reward_joint: marginal score of inserting
#         candidate_task_id — R_after (route WITH candidate, scored over
#         every task in the route) minus R_before (route WITHOUT it) —
#         since inserting a task can delay every stop that comes after it,
#         not just affect the candidate itself."""
#         robot = self.robots[robot_id]
#         candidate = self.tasks.get(candidate_task_id)
#         if candidate is None:
#             return float("-inf")
 
#         already_onboard = set(robot["onboard_tasks"])
#         committed_ids = list(robot["assigned_tasks"]) + list(robot["onboard_tasks"])
 
#         def _pickup_time_for(tid, walked_pickup_time):
#             # Already-onboard tasks were picked up in the PAST — the walk
#             # only tracks pickup_time for stops it actually visits, and an
#             # already-onboard task has no pickup stop left to visit, so it
#             # would otherwise always come back as None (-> -inf score).
#             # Use the real recorded pickup time instead; only tasks whose
#             # pickup hasn't happened yet (queued or the candidate) get
#             # their pickup time from the simulated walk.
#             if tid in already_onboard:
#                 return self.tasks[tid].get("pickup_time", self.current_time)
#             return walked_pickup_time
 
#         # R_before: route WITHOUT the candidate.
#         task_locs_before, pending_before, onboard_before = self._build_walk_state(robot)
#         r_before = 0.0
#         for tid in committed_ids:
#             t = self.tasks.get(tid)
#             if t is None:
#                 continue
#             pu, do = self._walk_stops(
#                 start_x=robot["x"], start_y=robot["y"], start_time=self.current_time,
#                 max_capacity=robot["max_capacity"],
#                 pending_pickup_ids=pending_before, onboard_ids=onboard_before,
#                 task_locs=task_locs_before, track_task_id=tid,
#             )
#             r_before += self._score_predicted_times(t, _pickup_time_for(tid, pu), do)
 
#         # R_after: same route WITH the candidate inserted — re-walked (and
#         # re-scored) per tracked task, since the candidate's presence can
#         # change which stop is nearest at each step.
#         task_locs_after, pending_after, onboard_after = self._build_walk_state(
#             robot, extra_pending={candidate_task_id: candidate}
#         )
#         r_after = 0.0
#         for tid in committed_ids + [candidate_task_id]:
#             t = self.tasks.get(tid)
#             if t is None:
#                 continue
#             pu, do = self._walk_stops(
#                 start_x=robot["x"], start_y=robot["y"], start_time=self.current_time,
#                 max_capacity=robot["max_capacity"],
#                 pending_pickup_ids=pending_after, onboard_ids=onboard_after,
#                 task_locs=task_locs_after, track_task_id=tid,
#             )
#             r_after += self._score_predicted_times(t, _pickup_time_for(tid, pu), do)
 
#         return r_after - r_before
 
#     # =========================================================================
#     # REWARD
#     # =========================================================================
 
#     def _compute_rewards(self, action_info) -> float:
#         reward_type = getattr(self, "reward_type", "legacy")
#         if reward_type == "wait_travel":
#             return self._compute_rewards_wait_travel(action_info)
#         return self._compute_rewards_legacy(action_info)

#     def _compute_rewards_legacy(self, action_info) -> float:
#         W_COMP = self.W_COMP
#         W_WAIT = self.W_WAIT
#         W_DEADLINE = self.W_DEADLINE
#         W_OBS = self.W_OBS
 
#         WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
#         DEADLINE_CAP = max(1.0, float(self.max_travel_delay_s))
 
#         reward = 0.0
#         r_comp = 0.0
#         r_wait = 0.0
#         r_deadline = 0.0
#         r_obsolete = 0.0
 
#         # 1) pickup wait penalty
#         for r in self.robots.values():
#             task_id = r.get("just_picked_up_task")
#             if not task_id:
#                 continue
#             task = self.tasks.get(task_id)
#             if task is None:
#                 r["just_picked_up_task"] = None
#                 continue
 
#             wait = max(0.0, self.current_time - task["release_time"])
#             delta = W_WAIT * (-min(wait, WAIT_CAP) / WAIT_CAP)
#             reward += delta
#             r_wait += delta
#             r["just_picked_up_task"] = None
 
#         # 2) completion + lateness penalties
#         for task in self.tasks.values():
#             if not task.get("is_completed"):
#                 continue
#             if task.get("_rewarded"):
#                 continue
#             task["_rewarded"] = True
 
#             reward += W_COMP
#             r_comp += W_COMP
 
#             pickup_time = task.get("pickup_time", self.current_time)
#             dropoff_time = task.get("dropoff_time", self.current_time)
 
#             if task.get("pickup_deadline") is not None:
#                 late_p = max(0.0, pickup_time - task["pickup_deadline"])
#                 delta = W_DEADLINE * (-min(late_p, DEADLINE_CAP) / DEADLINE_CAP)
#                 reward += delta
#                 r_deadline += delta
 
#             if task.get("dropoff_deadline") is not None:
#                 late_d = max(0.0, dropoff_time - task["dropoff_deadline"])
#                 delta = W_DEADLINE * (-min(late_d, DEADLINE_CAP) / DEADLINE_CAP)
#                 reward += delta
#                 r_deadline += delta
 
#         # 3) obsolete penalties (only not-picked tasks become obsolete by design)
#         for task in self.tasks.values():
#             if not task.get("is_obsolete"):
#                 continue
#             if task.get("_obsolete_rewarded"):
#                 continue
#             task["_obsolete_rewarded"] = True
 
#             delta_obs = -W_OBS
#             reward += delta_obs
#             r_obsolete += delta_obs
#             # print(r_obsolete,'r_obsolete')
#             late = max(0.0, self.current_time - task.get("pickup_deadline", self.current_time))
#             delta_dead = W_DEADLINE * (-min(late, DEADLINE_CAP) / DEADLINE_CAP)
#             reward += delta_dead
#             r_deadline += delta_dead
 
#         self.debug_last_r_comp = float(r_comp)
#         self.debug_last_r_wait = float(r_wait)
#         self.debug_last_r_deadline = float(r_deadline)
#         self.debug_last_r_obsolete = float(r_obsolete)
 
#         self.debug_ep_r_comp += float(r_comp)
#         self.debug_ep_r_wait += float(r_wait)
#         self.debug_ep_r_deadline += float(r_deadline)
#         self.debug_ep_r_obsolete += float(r_obsolete)
#         # print(self.debug_ep_r_obsolete,'debug_ep_r_obsolete')
#         return float(reward)

#     def _compute_rewards_wait_travel(self, action_info) -> float:
#         """Matches the reference repo's reward_type: wait_travel /
#         completion_mode: valid_dropoff. Two structural differences from
#         _compute_rewards_legacy, both deliberate:

#         1. Completion reward (W_COMP) is GATED on genuinely on-time
#            completion (pickup AND dropoff both within their deadlines) —
#            a late completion earns ZERO completion credit, not "completed
#            plus a separate lateness penalty" like legacy mode. This
#            mirrors predicted_reward's own internal valid_completion gate
#            in _score_predicted_times, so the real reward and the
#            heuristic baselines' own scoring now agree on what "on time"
#            means.
#         2. No separate continuous deadline-lateness penalty at all
#            (matches her w_deadline: 0) — deadline compliance lives
#            entirely in the binary gate above. In its place, a dedicated
#            EXCESS RIDE TIME penalty (W_TRAVEL): how much longer the actual
#            ride took than its own estimated travel time, independent of
#            whether the deadline was hit.

#         Wait penalty and obsolete-task handling are unchanged from legacy
#         mode.

#         NOTE ON DEBUG FIELDS: this mode has no "r_deadline" component at
#         all, so — to avoid touching every downstream consumer of
#         debug_last_r_deadline/debug_ep_r_deadline — this repurposes that
#         SAME field to carry r_travel's value when reward_type ==
#         "wait_travel". Its printed/logged label still says "r_dead" /
#         "r_deadline" either way; mentally read that as "r_travel" whenever
#         this mode is active.
#         """
#         W_COMP = self.W_COMP
#         W_WAIT = self.W_WAIT
#         W_TRAVEL = getattr(self, "W_TRAVEL", 1.25)
#         W_OBS = self.W_OBS

#         WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
#         TRAVEL_CAP = max(1.0, float(self.max_travel_delay_s))

#         reward = 0.0
#         r_comp = 0.0
#         r_wait = 0.0
#         r_travel = 0.0
#         r_obsolete = 0.0

#         # 1) pickup wait penalty — identical to legacy mode
#         for r in self.robots.values():
#             task_id = r.get("just_picked_up_task")
#             if not task_id:
#                 continue
#             task = self.tasks.get(task_id)
#             if task is None:
#                 r["just_picked_up_task"] = None
#                 continue

#             wait = max(0.0, self.current_time - task["release_time"])
#             delta = W_WAIT * (-min(wait, WAIT_CAP) / WAIT_CAP)
#             reward += delta
#             r_wait += delta
#             r["just_picked_up_task"] = None

#         # 2) gated completion reward + dedicated excess-ride-time penalty
#         for task in self.tasks.values():
#             if not task.get("is_completed"):
#                 continue
#             if task.get("_rewarded"):
#                 continue
#             task["_rewarded"] = True

#             pickup_time = task.get("pickup_time", self.current_time)
#             dropoff_time = task.get("dropoff_time", self.current_time)

#             valid_completion = True
#             if task.get("pickup_deadline") is not None:
#                 valid_completion = valid_completion and (pickup_time <= task["pickup_deadline"])
#             if task.get("dropoff_deadline") is not None:
#                 valid_completion = valid_completion and (dropoff_time <= task["dropoff_deadline"])

#             if valid_completion:
#                 reward += W_COMP
#                 r_comp += W_COMP

#             ride_time = max(0.0, dropoff_time - pickup_time)
#             est_travel = task.get("est_travel_time", 0.0)
#             excess_ride = max(0.0, ride_time - est_travel)
#             delta = W_TRAVEL * (-min(excess_ride, TRAVEL_CAP) / TRAVEL_CAP)
#             reward += delta
#             r_travel += delta

#         # 3) obsolete penalty — unchanged structurally from legacy, minus
#         # the extra deadline-style term.
#         for task in self.tasks.values():
#             if not task.get("is_obsolete"):
#                 continue
#             if task.get("_obsolete_rewarded"):
#                 continue
#             task["_obsolete_rewarded"] = True

#             delta_obs = -W_OBS
#             reward += delta_obs
#             r_obsolete += delta_obs

#         self.debug_last_r_comp = float(r_comp)
#         self.debug_last_r_wait = float(r_wait)
#         self.debug_last_r_deadline = float(r_travel)  # see NOTE in docstring above
#         self.debug_last_r_obsolete = float(r_obsolete)

#         self.debug_ep_r_comp += float(r_comp)
#         self.debug_ep_r_wait += float(r_wait)
#         self.debug_ep_r_deadline += float(r_travel)  # see NOTE in docstring above
#         self.debug_ep_r_obsolete += float(r_obsolete)

#         return float(reward)

 
#     # =========================================================================
#     # TERMINATION
#     # =========================================================================
 
#     def _check_episode_done(self) -> bool:
#         if len(self.tasks) < self.total_task_count:
#             return False
 
#         any_pending = any(
#             not t.get("is_completed") and not t.get("is_obsolete")
#             for t in self.tasks.values()
#         )
#         if any_pending:
#             return False
 
#         robots_idle = all(
#             len(r["assigned_tasks"]) == 0 and len(r["onboard_tasks"]) == 0
#             for r in self.robots.values()
#         )
#         return robots_idle
 
#     # =========================================================================
#     # UTILITIES
#     # =========================================================================
 
#     def action_mask(self) -> np.ndarray:
#         mask = np.zeros((self.num_robots, self.K_max + 1), dtype=np.uint8)
#         for r in range(self.num_robots):
#             cand_list = self._last_cand_task_ids[r] if r < len(self._last_cand_task_ids) else []
#             for k in range(min(self.K_max, len(cand_list))):
#                 if cand_list[k] is not None:
#                     mask[r, k] = 1
#             mask[r, self._noop_index] = 1
 
#         self.debug_last_mask_zero_count = int(np.sum(mask[:, :self._noop_index] == 0))
#         return mask
 
#     def close(self):
#         pass





import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import sys
import yaml
from PIL import Image
import torch as th
from scipy.optimize import linear_sum_assignment
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
        capacity_method: str = "assigned",
        conflict_resolution: str = "greedy",
        W_COMP: float = 2.0,
        W_WAIT: float = 1.0,
        W_DEADLINE: float = 10.0,
        W_OBS: float = 0.5,
        candidates_sorting: str = "distance",
        reward_type: str = "legacy",
        W_TRAVEL: float = 1.25,
        completion_mode: str = "dropoff",
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
        self.W_COMP = float(W_COMP)
        self.W_WAIT = float(W_WAIT)
        self.W_DEADLINE = float(W_DEADLINE)
        self.W_OBS = float(W_OBS)
 
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
        self.debug_noop_forced_count = 0     # noop because no candidates were offered
        self.debug_noop_chosen_count = 0     # noop despite candidates being available
        self.debug_had_candidates_count = 0  # decisions where >=1 real candidate existed
        self.debug_decisions_total = 0
 
        # diagnostics (last-step)
        self.debug_last_invalid_action_count = 0
        self.debug_last_total_action_count = 0
        self.debug_last_valid_action_count = 0
        self.debug_last_conflict_dropped_count = 0
        self.debug_last_capacity_rejected_count = 0
        self.debug_last_mask_zero_count = 0
        self.debug_last_noop_forced_count = 0
        self.debug_last_noop_chosen_count = 0
        self.debug_last_had_candidates_count = 0
        self.debug_last_decisions_total = 0
 
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
        
        self.conflict_resolution = conflict_resolution.lower()
        _valid_resolvers = ("greedy", "random", "hungarian", "hungarian_bids",
                            "predicted_reward", "predicted_reward_joint",
                            "capacity", "closest_than_capacity")
        if self.conflict_resolution not in _valid_resolvers:
            raise ValueError(f"conflict_resolution must be one of {_valid_resolvers}")
 
        self.candidates_sorting = candidates_sorting.lower()
        if self.candidates_sorting not in ("distance", "randomized"):
            raise ValueError("candidates_sorting must be 'distance' or 'randomized'")
 
        self.reward_type = reward_type.lower()
        if self.reward_type not in ("legacy", "wait_travel"):
            raise ValueError("reward_type must be 'legacy' or 'wait_travel'")
        self.W_TRAVEL = float(W_TRAVEL)
 
        self.completion_mode = completion_mode.lower()
        if self.completion_mode not in ("pickup", "dropoff", "valid_dropoff"):
            raise ValueError("completion_mode must be 'pickup', 'dropoff', or 'valid_dropoff'")
 
        # Populated externally, once per macro-step, by RTGNNPolicy.forward()
        # via set_pending_logits() — only used by conflict_resolution=
        # 'hungarian_bids'. None until the first policy forward() call.
        self._pending_logits = None
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
 
        self._pending_logits = None
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
        self.debug_noop_forced_count = 0
        self.debug_noop_chosen_count = 0
        self.debug_had_candidates_count = 0
        self.debug_decisions_total = 0
 
        self.debug_last_invalid_action_count = 0
        self.debug_last_total_action_count = 0
        self.debug_last_valid_action_count = 0
        self.debug_last_conflict_dropped_count = 0
        self.debug_last_capacity_rejected_count = 0
        self.debug_last_mask_zero_count = 0
        self.debug_last_noop_forced_count = 0
        self.debug_last_noop_chosen_count = 0
        self.debug_last_had_candidates_count = 0
        self.debug_last_decisions_total = 0
 
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
        # if terminated or truncated:
            # print({
                
            #     "candidate_ratio":
            #         self.debug_had_candidates_count /
            #         max(1, self.debug_decisions_total),
 
            #     "chosen_noop_when_candidate":
            #         self.debug_noop_chosen_count /
            #         max(1, self.debug_had_candidates_count),
 
            #     "forced_noop_ratio":
            #         self.debug_noop_forced_count /
            #         max(1, self.debug_decisions_total),
            # })
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
 
            # noop diagnostics
            "noop_forced_count": self.debug_last_noop_forced_count,
            "noop_chosen_count": self.debug_last_noop_chosen_count,
            "had_candidates_count": self.debug_last_had_candidates_count,
            "decisions_total": self.debug_last_decisions_total,
 
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
        return obs, reward, terminated, truncated, info
   

 
    def _resolve_conflicts(self, requests):
        if not requests:
            return []
        if self.conflict_resolution == "greedy":
            return self._resolve_conflicts_greedy(requests)
        elif self.conflict_resolution == "random":
            return self._resolve_conflicts_random(requests)
        elif self.conflict_resolution == "hungarian":
            return self._resolve_conflicts_hungarian(requests)
        elif self.conflict_resolution == "hungarian_bids":
            return self._resolve_conflicts_hungarian_bids(requests)
        elif self.conflict_resolution == "predicted_reward":
            return self._resolve_conflicts_predicted_reward(requests)
        elif self.conflict_resolution == "predicted_reward_joint":
            return self._resolve_conflicts_predicted_reward_joint(requests)
        elif self.conflict_resolution in ("capacity", "closest_than_capacity"):
            # closest_than_capacity is an alias for 'greedy' (distance-sort,
            # then first-come-wins under capacity) — matches the reference
            # repo's naming for that same behavior, not a separate
            # implementation. 'capacity' is the "dumbest" variant: no
            # priority ordering at all, just raw arrival order + capacity.
            if self.conflict_resolution == "closest_than_capacity":
                return self._resolve_conflicts_greedy(requests)
            return self._resolve_conflicts_capacity(requests)
        else:
            raise ValueError(f"Unknown conflict_resolution: {self.conflict_resolution}")
 
    def _resolve_conflicts_greedy(self, requests):
        """Original behavior: process requests in ascending-distance order;
        first robot to claim a task wins it, later claims for the same task
        are dropped as conflicts."""
        ordered = sorted(requests, key=lambda r: r[0])
        assigned_this_step = set()
        winners = []
        for dist, robot_id, task_id in ordered:
            if task_id in assigned_this_step:
                self.debug_last_conflict_dropped_count += 1
                continue
            assigned_this_step.add(task_id)
            winners.append((robot_id, task_id))
        return winners
 
    def _resolve_conflicts_capacity(self, requests):
        """The simplest possible resolver: NO priority ordering at all —
        not distance, not randomized — requests are processed in whatever
        order they were constructed in (robot iteration order from
        _process_actions), first-come-wins per contested task, with
        capacity as the only real constraint. Deliberately the 'dumbest'
        baseline resolver other resolvers should be expected to beat."""
        assigned_this_step = set()
        winners = []
        for dist, robot_id, task_id in requests:  # kept in original (unsorted) order
            if task_id in assigned_this_step:
                self.debug_last_conflict_dropped_count += 1
                continue
            assigned_this_step.add(task_id)
            winners.append((robot_id, task_id))
        return winners
 
    def _resolve_conflicts_random(self, requests):
        """Same first-come-first-served structure as greedy, but processing
        order is a uniformly random permutation instead of ascending
        distance — so among robots contesting the same task, the winner is
        random rather than always the nearest one. Uses self.np_random
        (seeded via reset(seed=...)) for reproducibility across episodes
        run with the same seed."""
        order = self.np_random.permutation(len(requests))
        assigned_this_step = set()
        winners = []
        for idx in order:
            dist, robot_id, task_id = requests[int(idx)]
            if task_id in assigned_this_step:
                self.debug_last_conflict_dropped_count += 1
                continue
            assigned_this_step.add(task_id)
            winners.append((robot_id, task_id))
        return winners
 
    def _resolve_conflicts_hungarian(self, requests):
        """Centralized optimal assignment via the Hungarian algorithm.
 
        Under this action model each robot proposes exactly one task per
        decision, so restricting Hungarian to literally-proposed (robot,
        task) pairs would be mathematically identical to greedy (each robot
        has only one edge in the bipartite graph, so there's no cross-task
        tradeoff to exploit). To make this a genuinely different strategy,
        a robot is eligible for ANY task proposed by ANY robot this round,
        but ONLY if that task was also in the robot's own candidate list
        this step (self._last_cand_task_ids) — so it can still only be
        assigned something it could actually have seen/chosen under the
        mask, just not necessarily the specific one it happened to pick.
        The solver then finds the minimum-total-distance one-to-one
        matching across that whole eligible set at once, instead of
        resolving conflicts one task at a time.
        """
        robot_ids = sorted({r for _, r, _ in requests})
        task_ids  = sorted({t for _, _, t in requests})
        R, T = len(robot_ids), len(task_ids)
        if R == 0 or T == 0:
            return []
 
        r_idx = {rid: i for i, rid in enumerate(robot_ids)}
        t_idx = {tid: i for i, tid in enumerate(task_ids)}
 
        # Map robot_id -> its own candidate task_ids this step, for the
        # eligibility filter described above.
        robot_ids_sorted_all = sorted(self.robots.keys())
        own_candidates = {}
        for rid in robot_ids:
            try:
                r_pos = robot_ids_sorted_all.index(rid)
                offered = self._last_cand_task_ids[r_pos]
            except (ValueError, IndexError):
                offered = []
            own_candidates[rid] = set(t for t in offered if t is not None)
 
        INFEASIBLE = 1e9
        cost = np.full((R, T), INFEASIBLE, dtype=np.float64)
        for rid in robot_ids:
            robot = self.robots[rid]
            eligible = own_candidates[rid]
            for tid in task_ids:
                if tid not in eligible:
                    continue
                task = self.tasks.get(tid)
                if task is None:
                    continue
                d = np.sqrt(
                    (robot["x"] - task["pickup_x"]) ** 2 +
                    (robot["y"] - task["pickup_y"]) ** 2
                )
                cost[r_idx[rid], t_idx[tid]] = d
 
        row_ind, col_ind = linear_sum_assignment(cost)
 
        winners = []
        matched_robots = set()
        for ri, ti in zip(row_ind, col_ind):
            if cost[ri, ti] >= INFEASIBLE:
                continue  # not a real eligible pairing, skip
            winners.append((robot_ids[ri], task_ids[ti]))
            matched_robots.add(robot_ids[ri])
 
        # Robots that made a request but weren't matched by the solver
        # (either genuinely infeasible or lost out in the optimal solution)
        # count the same way an unmatched greedy/random loser would.
        self.debug_last_conflict_dropped_count += max(0, R - len(winners))
        return winners
 
    def set_pending_logits(self, logits: np.ndarray) -> None:
        """Called externally (by RTGNNPolicy.forward(), see
        src/models/sb3_gnn_policy.py) once per macro-step, BEFORE step() is
        called for that same decision, with this robot-step's raw candidate
        logits — shape [R, K_max], same per-robot ordering as
        sorted(self.robots.keys()) and same per-slot ordering as
        self._last_cand_task_ids. Used only by
        conflict_resolution='hungarian_bids' as bid values in place of
        distance. Not required for 'greedy'/'random'/'hungarian'."""
        self._pending_logits = np.asarray(logits, dtype=np.float64)
 
    def _resolve_conflicts_hungarian_bids(self, requests):
        """Centralized optimal assignment via the Hungarian algorithm, using
        the POLICY'S OWN LOGITS as bid values instead of distance — i.e. a
        genuine auction: each robot's bid for a task is how strongly its
        policy already wants that task (higher logit = stronger bid), and
        the solver finds the assignment that maximizes total bid value
        (equivalently: minimizes total NEGATIVE bid) across every robot
        that made a request and every task any of them proposed, subject to
        the same eligibility rule as _resolve_conflicts_hungarian (a robot
        can only be assigned a task that was actually in its own candidate
        list this step).
 
        Requires set_pending_logits() to have been called this step (see
        that method's docstring) — raises clearly if not, rather than
        silently falling back to something else, since a silent fallback
        would make it easy to not notice bids were never actually wired up.
        """
        if self._pending_logits is None:
            raise RuntimeError(
                "conflict_resolution='hungarian_bids' requires set_pending_logits() "
                "to be called each step before step(actions) — see "
                "RTGNNPolicy.forward() in src/models/sb3_gnn_policy.py, and make "
                "sure model.policy._bid_env is wired to this env's VecEnv after "
                "construction (see train_ppo.py)."
            )
 
        robot_ids = sorted({r for _, r, _ in requests})
        task_ids  = sorted({t for _, _, t in requests})
        R, T = len(robot_ids), len(task_ids)
        if R == 0 or T == 0:
            return []
 
        r_idx = {rid: i for i, rid in enumerate(robot_ids)}
        t_idx = {tid: i for i, tid in enumerate(task_ids)}
 
        robot_ids_sorted_all = sorted(self.robots.keys())
 
        # Map robot_id -> {task_id: bid_logit}, using that robot's OWN
        # candidate list crossed with that SAME robot's OWN logits for
        # those same slots (both indexed identically by slot position).
        own_bids = {}
        for rid in robot_ids:
            try:
                r_pos = robot_ids_sorted_all.index(rid)
                offered = self._last_cand_task_ids[r_pos]
                logits_row = self._pending_logits[r_pos]  # [K_max]
            except (ValueError, IndexError):
                offered, logits_row = [], None
            bid_map = {}
            if logits_row is not None:
                for slot, tid_at_slot in enumerate(offered):
                    if tid_at_slot is not None and slot < len(logits_row):
                        bid_map[tid_at_slot] = float(logits_row[slot])
            own_bids[rid] = bid_map
 
        INFEASIBLE = 1e9
        cost = np.full((R, T), INFEASIBLE, dtype=np.float64)
        for rid in robot_ids:
            bid_map = own_bids[rid]
            for tid in task_ids:
                if tid in bid_map:
                    cost[r_idx[rid], t_idx[tid]] = -bid_map[tid]  # maximize bid == minimize -bid
 
        row_ind, col_ind = linear_sum_assignment(cost)
 
        winners = []
        for ri, ti in zip(row_ind, col_ind):
            if cost[ri, ti] >= INFEASIBLE:
                continue
            winners.append((robot_ids[ri], task_ids[ti]))
 
        self.debug_last_conflict_dropped_count += max(0, R - len(winners))
        return winners
 
    def _resolve_conflicts_hungarian_with_bid_fn(self, requests, bid_fn):
        robot_ids = sorted({r for _, r, _ in requests})
        task_ids  = sorted({t for _, _, t in requests})
        R, T = len(robot_ids), len(task_ids)
        if R == 0 or T == 0:
            return []
 
        r_idx = {rid: i for i, rid in enumerate(robot_ids)}
        t_idx = {tid: i for i, tid in enumerate(task_ids)}
        robot_ids_sorted_all = sorted(self.robots.keys())
 
        eligible_tasks = {}
        for rid in robot_ids:
            try:
                r_pos = robot_ids_sorted_all.index(rid)
                offered = self._last_cand_task_ids[r_pos]
            except (ValueError, IndexError):
                offered = []
            eligible_tasks[rid] = {t for t in offered if t is not None}
 
        INFEASIBLE = 1e9
        cost = np.full((R, T), INFEASIBLE, dtype=np.float64)
        for rid in robot_ids:
            for tid in task_ids:
                if tid not in eligible_tasks[rid]:
                    continue
                bid = bid_fn(rid, tid)
                if bid is None or not np.isfinite(bid):
                    continue
                cost[r_idx[rid], t_idx[tid]] = -bid  # maximize bid == minimize -bid
 
        row_ind, col_ind = linear_sum_assignment(cost)
 
        winners = []
        for ri, ti in zip(row_ind, col_ind):
            if cost[ri, ti] >= INFEASIBLE:
                continue
            winners.append((robot_ids[ri], task_ids[ti]))
 
        self.debug_last_conflict_dropped_count += max(0, R - len(winners))
        return winners
 
    def _resolve_conflicts_predicted_reward(self, requests):
   
        return self._resolve_conflicts_hungarian_with_bid_fn(
            requests, lambda rid, tid: self.predict_candidate_score(rid, tid)
        )
 
    def _resolve_conflicts_predicted_reward_joint(self, requests):
        """Same as _resolve_conflicts_predicted_reward, but bids are the
        MARGINAL score (predict_candidate_score_joint: R_after - R_before
        over the robot's whole route) — matches the reference repo's
        'predicted_reward_joint' resolver."""
        return self._resolve_conflicts_hungarian_with_bid_fn(
            requests, lambda rid, tid: self.predict_candidate_score_joint(rid, tid)
        )
 
    def _process_actions(self, actions) -> Dict:
    
        robot_ids = sorted(self.robots.keys())
        requests = []
 
        invalid_action_count = 0
        valid_action_count = 0
        conflict_dropped_count = 0
        capacity_rejected_count = 0
 
        step_noop_forced = 0
        step_noop_chosen = 0
        step_had_candidates = 0
        step_decisions = 0
 
        for r_idx, action in enumerate(actions):
 
            if r_idx >= len(robot_ids):
                break
 
            offered = self._last_cand_task_ids[r_idx]
            # print(f"Step {self.current_step}: Robot {robot_ids[r_idx]} action={action}, cands={offered}, noop_index={self._noop_index}")
            had_candidates = any(t is not None for t in offered)
            is_noop = (int(action) == self._noop_index)
 
            # ---------- episode totals ----------
            self.debug_decisions_total += 1
            step_decisions += 1
 
            if had_candidates:
                self.debug_had_candidates_count += 1
                step_had_candidates += 1
 
            if is_noop:
                if had_candidates:
                    self.debug_noop_chosen_count += 1
                    step_noop_chosen += 1
                else:
                    self.debug_noop_forced_count += 1
                    step_noop_forced += 1
                continue
 
            robot_id = robot_ids[r_idx]
            cands = self._last_cand_task_ids[r_idx]
 
            if int(action) >= len(cands) or cands[int(action)] is None:
                invalid_action_count += 1
                continue
 
            task_id = cands[int(action)]
            task = self.tasks.get(task_id)
 
            if (
                task is None
                or task.get("is_assigned")
                or task.get("is_obsolete")
                or task.get("is_completed")
            ):
                invalid_action_count += 1
                continue
 
            robot = self.robots[robot_id]
 
            dist = np.sqrt(
                (robot["x"] - task["pickup_x"]) ** 2
                + (robot["y"] - task["pickup_y"]) ** 2
            )
 
            requests.append((dist, robot_id, task_id))
 
        # ---------------------------------------------------
        # Resolve conflicts
        # ---------------------------------------------------
        # requests.sort()
        # print(f"Step {self.current_step}: {len(requests)} , {requests},requests before conflict resolution")
        winners = self._resolve_conflicts(requests)
 
        assigned_this_step = set()
        action_info = {}
        # print(assigned_this_step, winners, "winners after conflict resolution")
        # for _, robot_id, task_id in requests:
 
        #     if task_id in assigned_this_step:
        #         conflict_dropped_count += 1
        #         continue
 
        #     robot = self.robots[robot_id]
        #     task = self.tasks.get(task_id)
 
        #     if (
        #         task is None
        #         or task.get("is_assigned")
        #         or task.get("is_obsolete")
        #         or task.get("is_completed")
        #     ):
        #         invalid_action_count += 1
        #         continue
 
        #     if self.capacity_method == "assigned":
        #         total_committed = (
        #             len(robot["onboard_tasks"])
        #             + len(robot["assigned_tasks"])
        #         )
        #     else:
        #         total_committed = len(robot["onboard_tasks"])
 
        #     if total_committed >= self.max_robot_capacity:
        #         capacity_rejected_count += 1
        #         continue
 
        #     robot["assigned_tasks"].append(task_id)
        #     task["is_assigned"] = True
        #     task["assigned_robot"] = robot_id
 
        #     assigned_this_step.add(task_id)
        #     print(assigned_this_step)
        #     action_info[robot_id] = {"assigned_task": task_id}
        #     print(f"Step {self.current_step}: Robot {robot_id} assigned to task {task_id}")
        #     valid_action_count += 1
        assigned_this_step = set()
        action_info = {}
 
        winner_set = set(winners)
 
        for robot_id, task_id in winners:
 
            robot = self.robots[robot_id]
            task = self.tasks.get(task_id)
 
            if (
                task is None
                or task.get("is_assigned")
                or task.get("is_obsolete")
                or task.get("is_completed")
            ):
                invalid_action_count += 1
                continue
 
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
 
            action_info[robot_id] = {
                "assigned_task": task_id
            }
 
            valid_action_count += 1
        conflict_dropped_count = len(requests) - len(winners)
        # ---------------------------------------------------
        # Per-step diagnostics
        # ---------------------------------------------------
        self.debug_last_invalid_action_count = invalid_action_count
        self.debug_last_valid_action_count = valid_action_count
        self.debug_last_conflict_dropped_count = conflict_dropped_count
        self.debug_last_capacity_rejected_count = capacity_rejected_count
 
        self.debug_last_noop_forced_count = step_noop_forced
        self.debug_last_noop_chosen_count = step_noop_chosen
        self.debug_last_had_candidates_count = step_had_candidates
        self.debug_last_decisions_total = step_decisions
 
        # ---------------------------------------------------
        # Episode diagnostics
        # ---------------------------------------------------
        self.debug_invalid_action_count += invalid_action_count
        self.debug_valid_action_count += valid_action_count
        self.debug_conflict_dropped_count += conflict_dropped_count
        self.debug_capacity_rejected_count += capacity_rejected_count
 
        action_info["_diag"] = {
            "invalid_action_count": invalid_action_count,
            "valid_action_count": valid_action_count,
            "conflict_dropped_count": conflict_dropped_count,
            "capacity_rejected_count": capacity_rejected_count,
            "noop_forced_count": step_noop_forced,
            "noop_chosen_count": step_noop_chosen,
            "had_candidates_count": step_had_candidates,
            "decisions_total": step_decisions,
        }
        # print(step_noop_chosen, step_noop_forced, step_had_candidates, step_decisions)
        return action_info
    def _process_actionsold(self, actions) -> Dict:
        """
        Uses _last_cand_task_ids from observation-time snapshot to prevent
        index mismatch between policy output and candidate list.
        """
        robot_ids = sorted(self.robots.keys())
        requests = []
 
        invalid_action_count = 0
        valid_action_count = 0
        conflict_dropped_count = 0
        total_action_count = 0
        capacity_rejected_count = 0
 
        step_noop_forced = 0
        step_noop_chosen = 0
        step_had_candidates = 0
        step_decisions = 0
 
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
            # print(f"Step {self.current_step}: Robot {robot_ids[r_idx]} action={action}, cands={self._last_cand_task_ids[r_idx]}, noop_index={self._noop_index}", {int(action)})
            offered = self._last_cand_task_ids[r_idx]
            # True if at least one task candidate exists
            had_candidates = any(t is not None for t in offered)
            is_noop = (int(action) == self._noop_index)
 
            self.debug_decisions_total += 1
 
            if had_candidates:
                self.debug_had_candidates_count += 1
 
            if is_noop:
                if had_candidates:
                    self.debug_noop_chosen_count += 1
                else:
                    self.debug_noop_forced_count += 1
 
                continue
 
            # if int(action) == self._noop_index:
            #     continue
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
            print(f"Step {self.current_step}: Robot {robot_id} assigned to Task {task_id}")
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
        top_k = candidates[: self.K_max]
 
      
        if self.candidates_sorting == "randomized" and len(top_k) > 1:
            order = self.np_random.permutation(len(top_k))
            top_k = [top_k[i] for i in order]
 
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
        return [tid for _, tid in top_k]
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
 
    
    def _simulate_route_with_candidate(self, robot_id, candidate_task_id):
        """Greedy nearest-next-stop walk over (robot's committed stops +
        candidate's pickup/dropoff), starting from the robot's current
        position/time. Returns (predicted_pickup_time, predicted_dropoff_time)
        for the candidate specifically, or (None, None) if the candidate
        never gets reached (e.g. capacity never frees up in the simulated
        walk before stops run out)."""
        robot = self.robots[robot_id]
        candidate = self.tasks.get(candidate_task_id)
        if candidate is None:
            return None, None
 
        task_locs, pending, onboard = self._build_walk_state(robot, extra_pending={candidate_task_id: candidate})
 
        return self._walk_stops(
            start_x=robot["x"], start_y=robot["y"], start_time=self.current_time,
            max_capacity=robot["max_capacity"],
            pending_pickup_ids=pending, onboard_ids=onboard,
            task_locs=task_locs, track_task_id=candidate_task_id,
        )
 
    def _build_walk_state(self, robot, extra_pending=None):
        """Build the (task_locs, pending_pickup_ids, onboard_ids) inputs
        _walk_stops needs, from a robot's real committed state plus
        optionally one extra not-yet-assigned candidate task. task_locs
        maps every relevant task_id -> {"pickup": (x,y), "dropoff": (x,y)}
        so _walk_stops can look up a task's dropoff location the moment
        its pickup is visited, even though it wasn't in an initial fixed
        stop list."""
        task_locs = {}
        pending = set()
        onboard = set()
 
        for tid in robot["assigned_tasks"]:
            t = self.tasks.get(tid)
            if t is None or t.get("is_obsolete"):
                continue
            task_locs[tid] = {"pickup": (t["pickup_x"], t["pickup_y"]), "dropoff": (t["dropoff_x"], t["dropoff_y"])}
            pending.add(tid)
 
        for tid in robot["onboard_tasks"]:
            t = self.tasks.get(tid)
            if t is None:
                continue
            task_locs[tid] = {"pickup": (t["pickup_x"], t["pickup_y"]), "dropoff": (t["dropoff_x"], t["dropoff_y"])}
            onboard.add(tid)
 
        if extra_pending:
            for tid, t in extra_pending.items():
                task_locs[tid] = {"pickup": (t["pickup_x"], t["pickup_y"]), "dropoff": (t["dropoff_x"], t["dropoff_y"])}
                pending.add(tid)
 
        return task_locs, pending, onboard
 
    def _walk_stops(self, start_x, start_y, start_time, max_capacity, pending_pickup_ids, onboard_ids, task_locs, track_task_id):
        """Shared greedy nearest-next-stop walk, mirroring
        _assign_next_stop()'s real dynamic behavior: eligible next stops
        are re-derived every iteration from CURRENT pending/onboard sets
        (pickups for pending tasks, if there's room; dropoffs for onboard
        tasks) — NOT a fixed precomputed list. Once a task's pickup is
        visited it moves from pending to onboard, making its dropoff
        eligible on the NEXT iteration, exactly like real execution. This
        is what makes multi-task routes (a not-yet-picked-up task whose
        dropoff hasn't happened yet) score correctly instead of getting
        stuck with no dropoff stop ever appearing.
 
        Returns (pickup_time, dropoff_time) for track_task_id, or (None,
        None) if it's never reached in the simulated walk."""
        cur_x, cur_y, cur_time = start_x, start_y, start_time
        pending = set(pending_pickup_ids)
        onboard = set(onboard_ids)
        pickup_time = dropoff_time = None
 
        while pending or onboard:
            room = len(onboard) < max_capacity
            candidates = []
            if room:
                for tid in pending:
                    x, y = task_locs[tid]["pickup"]
                    candidates.append((tid, "pickup", x, y))
            for tid in onboard:
                x, y = task_locs[tid]["dropoff"]
                candidates.append((tid, "dropoff", x, y))
 
            if not candidates:
                break
 
            nxt = min(candidates, key=lambda s: (s[2] - cur_x) ** 2 + (s[3] - cur_y) ** 2)
            dist = float(np.sqrt((nxt[2] - cur_x) ** 2 + (nxt[3] - cur_y) ** 2))
            cur_time += dist / max(1e-9, self.movement_speed)
            cur_x, cur_y = nxt[2], nxt[3]
 
            tid, kind = nxt[0], nxt[1]
            if kind == "pickup":
                pending.discard(tid)
                onboard.add(tid)
                if tid == track_task_id:
                    pickup_time = cur_time
            else:
                onboard.discard(tid)
                if tid == track_task_id:
                    dropoff_time = cur_time
                    break
 
        return pickup_time, dropoff_time
 
    def _score_predicted_times(self, task, pickup_time, dropoff_time):
        """Same-shaped scoring formula as _compute_rewards, but evaluated
        against PREDICTED (simulated) times instead of actual ones — see
        module docstring above. valid_completion is binary (matches the
        reference implementation's predicted_reward scoring exactly),
        which is simpler than the continuous lateness penalty
        _compute_rewards uses for tasks that actually complete late."""
        if pickup_time is None or dropoff_time is None:
            return float("-inf")
 
        WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
        DEADLINE_CAP = max(1.0, float(self.max_travel_delay_s))
 
        wait = max(0.0, pickup_time - task["release_time"])
        norm_wait = min(wait, WAIT_CAP) / WAIT_CAP
 
        ride_time = max(0.0, dropoff_time - pickup_time)
        excess_ride = max(0.0, ride_time - task.get("est_travel_time", 0.0))
        norm_excess = min(excess_ride, DEADLINE_CAP) / DEADLINE_CAP
 
        valid_completion = (
            pickup_time <= task.get("pickup_deadline", float("inf"))
            and dropoff_time <= task.get("dropoff_deadline", float("inf"))
        )
 
        return (
            self.W_COMP * float(valid_completion)
            - self.W_WAIT * norm_wait
            - self.W_DEADLINE * norm_excess
        )
 
    def predict_candidate_score(self, robot_id, candidate_task_id):
        """predicted_reward: score of inserting candidate_task_id into
        robot_id's route, based on the candidate's OWN predicted
        pickup/dropoff times only (not how it affects other already-
        committed tasks — see predict_candidate_score_joint for that)."""
        candidate = self.tasks.get(candidate_task_id)
        if candidate is None:
            return float("-inf")
        pickup_time, dropoff_time = self._simulate_route_with_candidate(robot_id, candidate_task_id)
        return self._score_predicted_times(candidate, pickup_time, dropoff_time)
 
    def predict_candidate_score_joint(self, robot_id, candidate_task_id):
        """predicted_reward_joint: marginal score of inserting
        candidate_task_id — R_after (route WITH candidate, scored over
        every task in the route) minus R_before (route WITHOUT it) —
        since inserting a task can delay every stop that comes after it,
        not just affect the candidate itself."""
        robot = self.robots[robot_id]
        candidate = self.tasks.get(candidate_task_id)
        if candidate is None:
            return float("-inf")
 
        already_onboard = set(robot["onboard_tasks"])
        committed_ids = list(robot["assigned_tasks"]) + list(robot["onboard_tasks"])
 
        def _pickup_time_for(tid, walked_pickup_time):
           
            if tid in already_onboard:
                return self.tasks[tid].get("pickup_time", self.current_time)
            return walked_pickup_time
 
        # R_before: route WITHOUT the candidate.
        task_locs_before, pending_before, onboard_before = self._build_walk_state(robot)
        r_before = 0.0
        for tid in committed_ids:
            t = self.tasks.get(tid)
            if t is None:
                continue
            pu, do = self._walk_stops(
                start_x=robot["x"], start_y=robot["y"], start_time=self.current_time,
                max_capacity=robot["max_capacity"],
                pending_pickup_ids=pending_before, onboard_ids=onboard_before,
                task_locs=task_locs_before, track_task_id=tid,
            )
            r_before += self._score_predicted_times(t, _pickup_time_for(tid, pu), do)
 
        # R_after: same route WITH the candidate inserted — re-walked (and
        # re-scored) per tracked task, since the candidate's presence can
        # change which stop is nearest at each step.
        task_locs_after, pending_after, onboard_after = self._build_walk_state(
            robot, extra_pending={candidate_task_id: candidate}
        )
        r_after = 0.0
        for tid in committed_ids + [candidate_task_id]:
            t = self.tasks.get(tid)
            if t is None:
                continue
            pu, do = self._walk_stops(
                start_x=robot["x"], start_y=robot["y"], start_time=self.current_time,
                max_capacity=robot["max_capacity"],
                pending_pickup_ids=pending_after, onboard_ids=onboard_after,
                task_locs=task_locs_after, track_task_id=tid,
            )
            r_after += self._score_predicted_times(t, _pickup_time_for(tid, pu), do)
 
        return r_after - r_before
 
    # REWARD
   
    def _compute_rewards(self, action_info) -> float:
        reward_type = getattr(self, "reward_type", "legacy")
        if reward_type == "wait_travel":
            return self._compute_rewards_wait_travel(action_info)
        return self._compute_rewards_legacy(action_info)
 
    def _compute_rewards_legacy(self, action_info) -> float:
        W_COMP = self.W_COMP
        W_WAIT = self.W_WAIT
        W_DEADLINE = self.W_DEADLINE
        W_OBS = self.W_OBS
 
        WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
        DEADLINE_CAP = max(1.0, float(self.max_travel_delay_s))
 
        reward = 0.0
        r_comp = 0.0
        r_wait = 0.0
        r_deadline = 0.0
        r_obsolete = 0.0
 
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
 
    def _compute_rewards_wait_travel(self, action_info) -> float:
        
        W_COMP = self.W_COMP
        W_WAIT = self.W_WAIT
        W_TRAVEL = getattr(self, "W_TRAVEL", 1.25)
        completion_mode = getattr(self, "completion_mode", "dropoff")
 
        WAIT_CAP = max(1.0, float(self.max_wait_delay_s))
        TRAVEL_CAP = max(1.0, float(self.max_travel_delay_s))
 
        reward = 0.0
        r_comp = 0.0
        r_wait = 0.0
        r_travel = 0.0
        r_obsolete = 0.0
 
        # 1) pickup wait penalty + "pickup" mode completion reward — both
        # fire on the same pickup event.
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
 
            if completion_mode == "pickup":
                reward += W_COMP
                r_comp += W_COMP
 
            r["just_picked_up_task"] = None
 
        # 2) "dropoff"/"valid_dropoff" mode completion reward + excess-ride-
        # time penalty — both fire on genuine task completion (dropoff).
        for task in self.tasks.values():
            if not task.get("is_completed"):
                continue
            if task.get("_rewarded"):
                continue
            task["_rewarded"] = True
 
            pickup_time = task.get("pickup_time", self.current_time)
            dropoff_time = task.get("dropoff_time", self.current_time)
 
            if completion_mode == "valid_dropoff":
                valid_completion = True
                if task.get("pickup_deadline") is not None:
                    valid_completion = valid_completion and (pickup_time <= task["pickup_deadline"])
                if task.get("dropoff_deadline") is not None:
                    valid_completion = valid_completion and (dropoff_time <= task["dropoff_deadline"])
                if valid_completion:
                    reward += W_COMP
                    r_comp += W_COMP
            elif completion_mode == "dropoff":
                reward += W_COMP
                r_comp += W_COMP
            # completion_mode == "pickup": already awarded above, nothing here
 
            ride_time = max(0.0, dropoff_time - pickup_time)
            est_travel = task.get("est_travel_time", 0.0)
            excess_ride = max(0.0, ride_time - est_travel)
            delta = W_TRAVEL * (-min(excess_ride, TRAVEL_CAP) / TRAVEL_CAP)
            reward += delta
            r_travel += delta
 
        # 3) obsolete penalty — folded into wait (W_WAIT/WAIT_CAP), with a
        # FLOOR of 0.05 on the normalized fraction, matching her formula
        # exactly. No separate W_OBS term in this mode at all. Skipped
        # entirely (matching her code) if the task has no recorded
        # pickup_deadline.
        for task in self.tasks.values():
            if not task.get("is_obsolete"):
                continue
            if task.get("_obsolete_rewarded"):
                continue
            task["_obsolete_rewarded"] = True
 
            pickup_deadline = task.get("pickup_deadline")
            if pickup_deadline is None:
                continue
            late = max(1.0, self.current_time - float(pickup_deadline))
            delta = W_WAIT * (-max(0.05, min(late, WAIT_CAP) / WAIT_CAP))
            reward += delta
            r_obsolete += delta  # separate accumulator, not folded into r_wait's — see docstring
 
        self.debug_last_r_comp = float(r_comp)
        self.debug_last_r_wait = float(r_wait)
        self.debug_last_r_deadline = float(r_travel)  # repurposed field — holds r_travel in this mode
        self.debug_last_r_obsolete = float(r_obsolete)
 
        self.debug_ep_r_comp += float(r_comp)
        self.debug_ep_r_wait += float(r_wait)
        self.debug_ep_r_deadline += float(r_travel)
        self.debug_ep_r_obsolete += float(r_obsolete)
 
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