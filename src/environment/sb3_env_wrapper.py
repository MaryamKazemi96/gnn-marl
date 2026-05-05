# """
# Gymnasium-compatible wrapper for GNN-based PPO training.

# Converts the new graph-based MultiAgentTaskEnv to work with SB3's expectations:
# - Converts old observation format to new padded ego-graph format
# - Handles action decoding
# - Provides action masking
# - Manages PPO reward aggregation
# """
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from typing import Dict, Any, Tuple, Optional, List

# from src.utils.ego_graph_builder import build_padded_ego_batch
# from src.utils.feature_fn import make_feature_fn, compute_feature_dim, expand_edge_features


# class GNNPPOEnvWrapper(gym.Env):
#     """
#     Wrapper that adapts your MultiAgentTaskEnv (which uses old graph format)
#     to the new ego-graph padded format required by RTGNNPolicy + PPO.
    
#     Key changes from base env:
#     - Observation: Dict with padded ego-graphs per robot (new format)
#     - Action: MultiDiscrete([K_max+1] * n_robots) with NO-OP support
#     - Reward: Aggregated per-step from base env to single scalar
#     """

#     def __init__(
#         self,
#         base_env,
#         K_max: int = 5,
#         N_max: int = 15,
#         E_max: int = 50,
#         use_xy_pickup: bool = False,
#         normalize_features: bool = True,
#         use_node_type: bool = True,
#         use_ego_robot: bool = True,
#         use_edge_rt: bool = False,
#         edge_features: Optional[List[str]] = None,
#         two_hop: bool = False,
#         two_hop_directed: bool = False,
#         decision_interval: int = 8,
#         verbose: bool = False,
#     ):
#         """
#         Initialize wrapper.
        
#         Args:
#             base_env: Your MultiAgentTaskEnv instance
#             K_max: Max candidate tasks per robot
#             N_max: Max nodes in padded ego-graph
#             E_max: Max edges in padded ego-graph
#             use_xy_pickup: Include relative pickup location
#             normalize_features: Normalize features
#             use_node_type: Include node type indicators
#             use_ego_robot: Include ego robot indicator
#             use_edge_rt: Use edge route/travel features
#             edge_features: List of edge feature names
#             two_hop: Use 2-hop graph with competitors
#             two_hop_directed: Use directed edges in 2-hop
#             decision_interval: Steps between policy decisions
#             verbose: Print debug info
#         """
#         super().__init__()
        
#         self.base_env = base_env
#         self.K_max = int(K_max)
#         self.N_max = int(N_max)
#         self.E_max = int(E_max)
#         self.n_robots = base_env.n_robots
#         self.verbose = bool(verbose)
#         self.decision_interval = int(decision_interval)
        
#         # Feature configuration
#         self.use_xy_pickup = bool(use_xy_pickup)
#         self.normalize_features = bool(normalize_features)
#         self.use_node_type = bool(use_node_type)
#         self.use_ego_robot = bool(use_ego_robot)
#         self.use_edge_rt = bool(use_edge_rt)
#         self.edge_features = list(edge_features or [])
#         self.two_hop = bool(two_hop)
#         self.two_hop_directed = bool(two_hop_directed)
        
#         # Compute feature dimension
#         self.F = compute_feature_dim(
#             use_xy_pickup=use_xy_pickup,
#             use_node_type=use_node_type,
#             use_edge_rt=use_edge_rt,
#             use_ego_robot=use_ego_robot,
#         )
#         self.edge_feat_dim = len(self.edge_features) if use_edge_rt else 0
        
#         if self.verbose:
#             print(f"[GNNPPOEnvWrapper] Feature dim: {self.F}, Edge feat dim: {self.edge_feat_dim}")
        
#         # NO-OP action index
#         self._noop_index = self.K_max
        
#         # ===== Action & Observation Spaces =====
#         self.action_space = spaces.MultiDiscrete([self.K_max + 1] * self.n_robots)
        
#         obs_space = {
#             "x": spaces.Box(-np.inf, np.inf, (self.n_robots, self.N_max, self.F), dtype=np.float32),
#             "node_mask": spaces.MultiBinary((self.n_robots, self.N_max)),
#             "edge_index": spaces.Box(0, self.N_max - 1, (self.n_robots, 2, self.E_max), dtype=np.int64),
#             "edge_mask": spaces.MultiBinary((self.n_robots, self.E_max)),
#             "cand_idx": spaces.Box(0, self.N_max - 1, (self.n_robots, self.K_max), dtype=np.int64),
#             "cand_mask": spaces.MultiBinary((self.n_robots, self.K_max)),
#         }
        
#         if self.edge_feat_dim > 0:
#             obs_space["edge_attr"] = spaces.Box(
#                 -np.inf,
#                 np.inf,
#                 (self.n_robots, self.E_max, self.edge_feat_dim),
#                 dtype=np.float32,
#             )
        
#         self.observation_space = spaces.Dict(obs_space)
        
#         # ===== Initialize feature function =====
#         # Create a wrapper object that provides the interface expected by feature_fn
#         self._feature_fn_wrapper = _EnvStateWrapper(self.base_env)
#         self.feature_fn = make_feature_fn(
#             env_state=self._feature_fn_wrapper,
#             use_xy_pickup=use_xy_pickup,
#             normalize_features=normalize_features,
#             use_node_type=use_node_type,
#             use_edge_rt=use_edge_rt,
#             edge_features=self.edge_features,
#             use_ego_robot=use_ego_robot,
#             max_position=base_env.planner.map_img.size[0] * base_env.planner.map_resolution,
#             max_robot_capacity=base_env.robot_capacity,
#             max_wait_delay_s=600.0,
#             max_travel_delay_s=3600.0,
#             max_steps=base_env.batch_time,
#         )
        
#         # ===== State tracking =====
#         self._last_cand_task_ids: List[List[Optional[str]]] = [[] for _ in range(self.n_robots)]
#         self._last_robot_ids: List[Optional[str]] = [None] * self.n_robots
#         self._step_count = 0
#         self._episode_reward = 0.0
        
#     # =========================================================================
#     # Core Gym API
#     # =========================================================================
    
#     def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
#         """Reset environment and return initial observation."""
#         super().reset(seed=seed)
        
#         obs_tuple, info = self.base_env.reset()
#         self._step_count = 0
#         self._episode_reward = 0.0
        
#         # Build graph observation
#         obs = self._build_graph_obs()
        
#         return obs, {"action_mask": self.action_mask()}
    
#     def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
#         """
#         Execute one environment step.
        
#         Args:
#             action: [n_robots] action indices (0..K_max-1 for task, K_max for NO-OP)
        
#         Returns:
#             obs: Padded ego-graph observation dict
#             reward: Aggregated scalar reward
#             terminated: Episode done flag
#             truncated: Truncation flag
#             info: Metadata dict
#         """
#         action = np.asarray(action, dtype=np.int64)
        
#         # ===== 1. Decode actions =====
#         assignments = self._decode_actions(action)
        
#         # ===== 2. Step base environment =====
#         obs_tuple, base_reward, terminated, truncated, info_reward, info_base = self.base_env.step(
#             list_t2r_assignments=assignments,
#             assignment_interval=self.decision_interval
#         )
        
#         self._step_count += 1
        
#         # ===== 3. Aggregate reward =====
#         if isinstance(base_reward, dict):
#             reward = float(sum(base_reward.values()))
#         else:
#             reward = float(base_reward)
        
#         self._episode_reward += reward
        
#         # ===== 4. Build graph observation =====
#         obs = self._build_graph_obs()
        
#         # ===== 5. Construct info =====
#         info = {
#             "action_mask": self.action_mask(),
#             "decoded_assignments": assignments,
#             "base_info": info_base,
#             "base_reward_info": info_reward,
#         }
        
#         if terminated or truncated:
#             info["episode_reward"] = self._episode_reward
#             self._episode_reward = 0.0
        
#         return obs, reward, terminated, truncated, info
    
#     # =========================================================================
#     # Graph Building
#     # =========================================================================
    
#     def _build_graph_obs(self) -> Dict[str, np.ndarray]:
#         """
#         Build padded ego-graph observation from base environment state.
#         """
#         # Get current state from base environment
#         robot_list = [self.base_env.robots[i] if i < len(self.base_env.robots) else None 
#                       for i in range(self.n_robots)]
        
#         task_dict = {
#             task.id: task for task in self.base_env.tasks
#         }
        
#         # Get candidate lists for each robot
#         candidate_lists = []
#         for robot in robot_list:
#             if robot is None:
#                 candidate_lists.append([])
#                 continue
            
#             # Get available task IDs
#             available_task_ids = self.base_env.get_available_task_ids()
            
#             # Filter by capacity
#             if robot.capacity >= robot.maxCapacity:
#                 candidate_lists.append([])
#                 continue
            
#             # Find indices of candidate tasks in the task dict
#             task_id_list = list(task_dict.keys())
#             candidate_indices = []
#             for task_id in available_task_ids:
#                 if task_id in task_id_list:
#                     candidate_indices.append(task_id_list.index(task_id))
            
#             # Limit to K_max
#             candidate_lists.append(candidate_indices[:self.K_max])
        
#         # Call graph builder
#         obs_dict, cand_task_ids = build_padded_ego_batch(
#             robots=robot_list,
#             tasks=task_dict,
#             candidate_lists=candidate_lists,
#             N_max=self.N_max,
#             E_max=self.E_max,
#             K_max=self.K_max,
#             F=self.F,
#             G=0,
#             feature_fn=self.feature_fn,
#             two_hop=self.two_hop,
#             two_hop_directed=self.two_hop_directed,
#             normalize_features=self.normalize_features,
#             vicinity_m=self.base_env.radius,
#             use_edge_rt=self.use_edge_rt,
#             edge_feat_dim=self.edge_feat_dim,
#             edge_features=self.edge_features,
#         )
        
#         # Store for action decoding
#         self._last_cand_task_ids = cand_task_ids
#         self._last_robot_ids = [r.robot_id if r is not None else None for r in robot_list]
        
#         return obs_dict
    
#     # =========================================================================
#     # Action Handling
#     # =========================================================================
    
#     def _decode_actions(self, action_vec: np.ndarray) -> Dict[int, Optional[List]]:
#         """
#         Decode action vector to task assignments for base environment.
        
#         Args:
#             action_vec: [n_robots] action indices
        
#         Returns:
#             Dict[robot_idx -> [task_id1, task_id2, ...]] or None
#         """
#         assignments = {}
        
#         for r_idx in range(self.n_robots):
#             a = int(action_vec[r_idx])
            
#             # NO-OP
#             if a == self._noop_index:
#                 assignments[r_idx] = [None, None]
#                 continue
            
#             # Valid candidate index
#             if 0 <= a < len(self._last_cand_task_ids[r_idx]):
#                 task_id = self._last_cand_task_ids[r_idx][a]
#                 assignments[r_idx] = [task_id, task_id]  # Format: [first_choice, second_choice]
#             else:
#                 assignments[r_idx] = [None, None]
        
#         return assignments
    
#     def action_mask(self) -> np.ndarray:
#         """
#         Get action mask [n_robots, K_max+1].
#         1 = allowed, 0 = blocked.
#         """
#         mask = np.zeros((self.n_robots, self.K_max + 1), dtype=np.uint8)
        
#         for r in range(self.n_robots):
#             # Candidates
#             for k in range(min(self.K_max, len(self._last_cand_task_ids[r]))):
#                 if self._last_cand_task_ids[r][k] is not None:
#                     mask[r, k] = 1
            
#             # NO-OP always allowed
#             mask[r, self._noop_index] = 1
        
#         return mask
    
#     def close(self):
#         """Cleanup."""
#         if hasattr(self.base_env, 'close'):
#             self.base_env.close()


# # ============================================================================
# # Helper: Environment State Wrapper
# # ============================================================================

# class _EnvStateWrapper:
#     """
#     Wrapper that provides the interface expected by make_feature_fn.
#     Bridges base_env to feature function.
#     """
    
#     def __init__(self, base_env):
#         self.base_env = base_env
    
#     @property
#     def robots(self):
#         """Return dict-like interface for robots."""
#         return {
#             r.robot_id: {
#                 'x': float(r.coordinate[0]),
#                 'y': float(r.coordinate[1]),
#                 'capacity': r.maxCapacity,
#                 'assigned_tasks': [t_id for t_id in r.current_tasks_id],
#             }
#             for r in self.base_env.robots
#         }
    
#     @property
#     def tasks(self):
#         """Return dict-like interface for tasks."""
#         return {
#             t.id: {
#                 'id': t.id,
#                 'pickup_x': float(t.pick_up_coord[0]),
#                 'pickup_y': float(t.pick_up_coord[1]),
#                 'dropoff_x': float(t.drop_off_coord[0]),
#                 'dropoff_y': float(t.drop_off_coord[1]),
#                 'release_time': float(t.release_time),
#                 'est_travel_time': float(t.estimatedTravelTime),
#                 'is_assigned': bool(t.is_assigned),
#                 'is_obsolete': bool(t.is_obsolete(self.base_env.time_count)),
#             }
#             for t in self.base_env.tasks
#         }
    
#     @property
#     def current_time(self):
#         """Return current simulation time."""
#         return float(self.base_env.time_count)


# # ============================================================================
# # Testing
# # ============================================================================

# if __name__ == "__main__":
#     """
#     Quick test of the wrapper.
#     """
#     # This requires your base environment to be set up
#     # For now, just test the import
#     print("GNNPPOEnvWrapper imported successfully!")

"""
Minimal wrapper for GNN-PPO training with the new MultiAgentTaskEnv.

The new MultiAgentTaskEnv already:
- Generates graph observations natively
- Handles action decoding
- Provides action masking
- Manages rewards

This wrapper just provides SB3 compatibility.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.environment.environment import MultiAgentTaskEnv


class GNNPPOEnvWrapper(gym.Env):
    """
    Minimal wrapper for SB3 PPO compatibility.
    
    The underlying MultiAgentTaskEnv already handles:
    - Graph observation generation
    - Action masking
    - Reward computation
    - Task lifecycle (release, pickup, dropoff, obsolete)
    """

    def __init__(
        self,
        agents: np.ndarray,
        tasks_batches: list,
        K_max: int = 5,
        N_max: int = 15,
        E_max: int = 50,
        use_xy_pickup: bool = False,
        normalize_features: bool = True,
        use_node_type: bool = True,
        use_ego_robot: bool = True,
        use_edge_rt: bool = False,
        edge_features: Optional[list] = None,
        two_hop: bool = False,
        two_hop_directed: bool = False,
        vicinity_m: float = 50.0,
        max_steps: int = 1000,
        max_robot_capacity: int = 2,
        max_wait_delay_s: float = 600.0,
        max_travel_delay_s: float = 3600.0,
    ):
        """
        Initialize wrapper.
        
        Args:
            agents: Robot data array [n_robots, 3] with [id, w, h]
            tasks_batches: List of task batch arrays
            K_max: Max candidate tasks per robot
            N_max: Max nodes in ego-graph
            E_max: Max edges in ego-graph
            ... (other params passed to MultiAgentTaskEnv)
        """
        super().__init__()
        
        # Create the base environment
        self.base_env = MultiAgentTaskEnv(
            agents=agents,
            tasks_batches=tasks_batches,
            K_max=K_max,
            N_max=N_max,
            E_max=E_max,
            use_xy_pickup=use_xy_pickup,
            normalize_features=normalize_features,
            use_node_type=use_node_type,
            use_ego_robot=use_ego_robot,
            use_edge_rt=use_edge_rt,
            edge_features=edge_features or [],
            two_hop=two_hop,
            two_hop_directed=two_hop_directed,
            vicinity_m=vicinity_m,
            max_steps=max_steps,
            max_robot_capacity=max_robot_capacity,
            max_wait_delay_s=max_wait_delay_s,
            max_travel_delay_s=max_travel_delay_s,
        )
        
        # Use the base env's spaces directly (they're already correct!)
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        
        # NO-OP index
        self._noop_index = K_max
        
        # Tracking
        self._episode_return = 0.0
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed)
        
        obs, info = self.base_env.reset(seed=seed)
        self._episode_return = 0.0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: [n_robots] action indices
        
        Returns:
            obs: Graph observation
            reward: Scalar episode reward
            terminated: Episode done flag
            truncated: Truncation flag
            info: Metadata dict with action_mask
        """
        # Step the base environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Accumulate return
        self._episode_return += reward
        
        # Add episode return to info if done
        if terminated or truncated:
            info["episode_return"] = self._episode_return
            self._episode_return = 0.0
        
        return obs, reward, terminated, truncated, info
    
    def action_mask(self) -> np.ndarray:
        """Get action mask from base environment."""
        return self.base_env.action_mask()
    
    def close(self):
        """Cleanup."""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """Quick test of the wrapper."""
    import yaml
    from pathlib import Path
    
    # Load config
    config_path = Path("configs/training_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    from train_ppo import load_generated_data
    agents, tasks_batches = load_generated_data("../data")
    
    # Create wrapper
    env = GNNPPOEnvWrapper(
        agents=agents,
        tasks_batches=tasks_batches,
        K_max=config.get('K_max', 5),
        N_max=config.get('N_max', 15),
        E_max=config.get('E_max', 50),
        use_xy_pickup=config.get('use_xy_pickup', False),
        normalize_features=config.get('normalize_features', True),
        use_node_type=config.get('use_node_type', True),
        use_ego_robot=config.get('use_ego_robot', True),
        use_edge_rt=config.get('use_edge_rt', False),
        two_hop=config.get('two_hop', False),
        vicinity_m=config.get('vicinity_m', 20.0),
        max_steps=config.get('max_steps', 1000),
    )
    
    # Test reset
    obs, info = env.reset()
    print("✓ Reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action mask shape: {info['action_mask'].shape}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Step successful")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Info keys: {list(info.keys())}")
    
    env.close()
    print("✓ Wrapper test passed!")