
from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, List, cast

BASE_ROBOT_FEATURE_NAMES: List[str] = [
    "robot_pos_x", "robot_pos_y", "robot_free_capacity",
    "pad_3", "pad_4", "pad_5", "pad_6", "pad_7", "pad_8",
]

BASE_TASK_FEATURE_NAMES: List[str] = [
    "task_release_time_s", "task_waiting_time_s", "task_est_travel_time_s",
    "task_pickup_loc_x", "task_pickup_loc_y", "task_drop_loc_x", "task_drop_loc_y",
    "task_is_obsolete", "task_is_assigned",
]


def compute_feature_dim(
    use_xy_pickup: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> int:
    """
    Compute total feature dimension based on enabled features.
    
    Base: 9 (robot position x2, capacity, padding x6)
    + 2 if use_xy_pickup and not use_edge_rt
    + 2 if use_node_type
    + 1 if use_ego_robot
    """
    dim = 9
    if use_xy_pickup and not use_edge_rt:
        dim += 2
    if use_node_type:
        dim += 2
    if use_ego_robot:
        dim += 1
    return dim


def get_feature_names(
    use_xy_pickup: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> tuple[List[str], List[str]]:
    """Return human-readable feature names for robots and tasks."""
    robot_names = list(BASE_ROBOT_FEATURE_NAMES)
    task_names = list(BASE_TASK_FEATURE_NAMES)

    if use_xy_pickup and not use_edge_rt:
        robot_names += ["pad_9", "pad_10"]
        task_names = [
            "task_release_time_s", "task_waiting_time_s", "task_est_travel_time_s",
            "task_pickup_loc_x", "task_pickup_loc_y", "task_pickup_dx", "task_pickup_dy",
            "task_drop_loc_x", "task_drop_loc_y", "task_is_obsolete", "task_is_assigned",
        ]

    if use_node_type:
        robot_names += ["is_robot", "is_task"]
        task_names += ["is_robot", "is_task"]

    if use_ego_robot:
        robot_names += ["is_ego_robot"]
        task_names += ["is_ego_robot"]

    return robot_names, task_names


def expand_edge_features(
    edge_features: Optional[List[str]],
    robot_commitment: str = "none",
    route_slots_k: int = 2,
) -> List[str]:
    """
    Expand edge features list based on robot commitment strategy.
    If using route_slots commitment, add slot-based features for planned route.
    """
    feats = list(edge_features or [])
    if robot_commitment != "route_slots":
        return feats
    
    for idx in range(int(route_slots_k)):
        slot_names = [
            f"slot{idx}_pu_dx",
            f"slot{idx}_pu_dy",
            f"slot{idx}_do_dx",
            f"slot{idx}_do_dy",
            f"slot{idx}_valid",
        ]
        for name in slot_names:
            if name not in feats:
                feats.append(name)
    return feats


def make_feature_fn(
    env_state,
    use_xy_pickup: bool = False,
    normalize_features: bool = False,
    use_node_type: bool = False,
    use_edge_rt: bool = False,
    edge_features: Optional[List[str]] = None,
    use_ego_robot: bool = False,
    robot_commitment: str = "none",
    route_slots_k: int = 2,
    # Environment-specific parameters
    max_position: float = 100.0,
    max_robot_capacity: int = 2,
    max_wait_delay_s: float = 60.0,
    max_travel_delay_s: float = 360.0,
    max_steps: int = 1000,
):
    """
    Factory function that creates a feature extraction function for the GNN.
    
    Args:
        env_state: Your environment state object (with robots, tasks, current_time)
        use_xy_pickup: Include relative pickup location displacement
        normalize_features: Normalize features to [0, 1] or [-1, 1] ranges
        use_node_type: Add one-hot node type indicators (robot vs task)
        use_edge_rt: Use edge-level route/travel features
        edge_features: List of edge feature names to compute
        use_ego_robot: Mark which robot is the "ego" agent
        robot_commitment: Strategy for encoding robot commitments ("none" or "route_slots")
        route_slots_k: Number of planned route slots to encode
        max_position: Normalization scale for positions
        max_robot_capacity: Max robots capacity (for normalization)
        max_wait_delay_s: Max waiting time (for normalization)
        max_travel_delay_s: Max travel time (for normalization)
        max_steps: Max episode steps (for normalization)
    """
    feature_dim = compute_feature_dim(
        use_xy_pickup=use_xy_pickup,
        use_node_type=use_node_type,
        use_edge_rt=use_edge_rt,
        use_ego_robot=use_ego_robot,
        robot_commitment=robot_commitment,
        route_slots_k=route_slots_k,
    )
    edge_features = expand_edge_features(edge_features, robot_commitment, route_slots_k)
    
    # Normalization scales
    pos_scale = pos_scale = max(1.0, float(getattr(env_state, "vicinity_m", max_position))) #max(1.0, float(max_position))
    cap_scale = max(1.0, float(max_robot_capacity))
    wait_scale = max(1.0, float(max_wait_delay_s))
    travel_scale = max(1.0, float(max_travel_delay_s))
    time_scale = max(1.0, float(max_steps)) if max_steps else wait_scale

    # ============================================================================
    # Helper functions
    # ============================================================================

    def _normalize_rid(x: Any) -> Optional[str]:
        """Normalize robot ID: accept only valid non-empty strings."""
        if isinstance(x, str) and x and x.lower() != "none":
            return x
        return None

    def _valid_robot_id(rid: Optional[str]) -> bool:
        """Check if robot ID is valid and exists in environment."""
        if rid is None:
            return False
        # Assuming env_state has robots dict with robot IDs as keys
        if hasattr(env_state, "robots"):
            return rid in env_state.robots
        return False

    def _robot_xy(rid):
        robot = env_state.robots.get(str(rid) if rid is not None else "")
        if robot is None:
            return 0.0, 0.0
        return float(robot.get("x", 0.0)), float(robot.get("y", 0.0))

    def _task_xy(task_id, is_pickup=True):
        task = env_state.tasks.get(task_id) if task_id else None
        if task is None:
            return 0.0, 0.0
        if is_pickup:
            return float(task.get("pickup_x", 0.0)), float(task.get("pickup_y", 0.0))
        return float(task.get("dropoff_x", 0.0)), float(task.get("dropoff_y", 0.0))

    def _append_node_type(out: np.ndarray, node_type: str) -> None:
        """Append one-hot encoded node type to feature vector."""
        if not use_node_type:
            return
        
        if use_ego_robot:
            # Last 3 positions reserved: [is_robot, is_task, is_ego_robot]
            if out.shape[0] >= 3:
                out[-3] = 1.0 if node_type == "robot" else 0.0
                out[-2] = 1.0 if node_type == "task" else 0.0
        else:
            # Last 2 positions reserved: [is_robot, is_task]
            if out.shape[0] >= 2:
                out[-2] = 1.0 if node_type == "robot" else 0.0
                out[-1] = 1.0 if node_type == "task" else 0.0

    def _append_ego_robot(out: np.ndarray, is_ego: bool) -> None:
        """Append ego robot indicator to feature vector."""
        if not use_ego_robot:
            return
        if out.shape[0] >= 1:
            out[-1] = 1.0 if is_ego else 0.0

    def _robot_route_slots(rid_s: str) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """
        Extract the next K planned route slots for this robot.
        Each slot contains: (pickup_xy, dropoff_xy, validity_flag)
        """
        if robot_commitment != "route_slots":
            return []
        
        try:
            robot = env_state.robots.get(rid_s)
            if robot is None:
                return []
            
            # Assuming robot has a planned_route or assigned_tasks attribute
            planned_tasks = getattr(robot, "planned_route", []) or getattr(robot, "assigned_tasks", [])
            
            out_slots: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []
            seen: set[str] = set()
            
            for task_id in planned_tasks:
                if task_id in seen:
                    continue
                seen.add(task_id)
                
                pu_xy = _task_xy(task_id, is_pickup=True)
                do_xy = _task_xy(task_id, is_pickup=False)
                out_slots.append((pu_xy, do_xy, 1.0))
                
                if len(out_slots) >= int(route_slots_k):
                    break
            
            return out_slots
        except Exception:
            return []

    def _edge_rt_features(rid_s: str, task_id: str) -> np.ndarray:
        """
        Compute edge-level route/travel features for a robot-task pair.
        Includes: displacement (dx, dy), estimated arrival time (eta), route slots.
        """
        out = np.zeros((len(edge_features),), dtype=np.float32)
        if not edge_features:
            return out
        
        try:
            # Get positions
            rx, ry = _robot_xy(rid_s)
            px, py = _task_xy(task_id, is_pickup=True)
            tx_do, ty_do = _task_xy(task_id, is_pickup=False)
            
            # Displacement from robot to task pickup
            dx = float(px - rx)
            dy = float(py - ry)
            if normalize_features:
                dx /= pos_scale
                dy /= pos_scale
            
            # Estimated arrival time (ETA) - simple Euclidean distance / speed
            dist = np.sqrt(dx**2 + dy**2)
            speed = 5.0  # m/s (adjust to your simulation)
            eta = dist / speed if speed > 0 else 0.0
            if normalize_features:
                eta = float(np.clip(eta / travel_scale, 0.0, 1.0))
            
            # Route slot features (if enabled)
            slot_values: dict[str, float] = {}
            if robot_commitment == "route_slots":
                slots = _robot_route_slots(rid_s)
                for s_idx in range(int(route_slots_k)):
                    if s_idx < len(slots):
                        (pu_xy, do_xy, valid) = slots[s_idx]
                        pu_dx = float(pu_xy[0] - px)
                        pu_dy = float(pu_xy[1] - py)
                        do_dx = float(do_xy[0] - tx_do)
                        do_dy = float(do_xy[1] - ty_do)
                        
                        if normalize_features:
                            pu_dx /= pos_scale
                            pu_dy /= pos_scale
                            do_dx /= pos_scale
                            do_dy /= pos_scale
                        
                        slot_values[f"slot{s_idx}_pu_dx"] = pu_dx
                        slot_values[f"slot{s_idx}_pu_dy"] = pu_dy
                        slot_values[f"slot{s_idx}_do_dx"] = do_dx
                        slot_values[f"slot{s_idx}_do_dy"] = do_dy
                        slot_values[f"slot{s_idx}_valid"] = float(valid)
                    else:
                        slot_values[f"slot{s_idx}_pu_dx"] = 0.0
                        slot_values[f"slot{s_idx}_pu_dy"] = 0.0
                        slot_values[f"slot{s_idx}_do_dx"] = 0.0
                        slot_values[f"slot{s_idx}_do_dy"] = 0.0
                        slot_values[f"slot{s_idx}_valid"] = 0.0
            
            # Populate output array
            for i, name in enumerate(edge_features):
                if name == "dx":
                    out[i] = dx
                elif name == "dy":
                    out[i] = dy
                elif name == "eta":
                    out[i] = eta
                elif name == "is_ego_edge":
                    out[i] = 0.0
                elif name in slot_values:
                    out[i] = slot_values[name]
            
            return out
        except Exception:
            return out

    def _resolve_task(x: Any) -> Optional[dict]:
        """
        Resolve a task object from ID or direct object reference.
        Returns task dictionary/object if found, None otherwise.
        """
        if isinstance(x, dict):
            return x
        
        # task_id = str(x) if x is not None else ""
        # try:
        #     if hasattr(env_state, "tasks"):
        #         return env_state.tasks.get(task_id)
        # except Exception:
        #     pass
        
        # return None
        for key in (x, str(x) if x is not None else None, 
                int(x) if str(x).isdigit() else None):
            if key is not None and key in env_state.tasks:
                return env_state.tasks[key]
        return None

    # ============================================================================
    # Main feature extraction function (returned by factory)
    # ============================================================================

    def feature_fn(obj_a, obj_b, node_type: str) -> np.ndarray:
        """
        Extract features for a node in the bipartite graph.
        
        Args:
            obj_a: Robot ID (string) or object
            obj_b: Task ID (string), Task object, or None
            node_type: One of {
                "robot", "robot_ego", "robot_other",
                "task",
                "edge_rt"
            }
        
        Returns:
            np.ndarray of shape (feature_dim,) with extracted features
        """
        out = np.zeros((feature_dim,), dtype=np.float32)
        current_time = getattr(env_state, "current_time", 0.0)

        # ====================================================================
        # ROBOT FEATURES
        # ====================================================================
        # Structure: [pos_x, pos_y, free_capacity, pad*6, ...]
        
        if node_type in {"robot", "robot_ego", "robot_other"}:
            is_ego = node_type != "robot_other"
            rid = _normalize_rid(obj_a)
            
            if not _valid_robot_id(rid):
                return out  # Return zeros for padded/missing robots
            
            rid_s = cast(str, rid)

            # Position
            rx, ry = _robot_xy(rid_s)
            if normalize_features:
                out[0], out[1] = rx / pos_scale, ry / pos_scale
            else:
                out[0], out[1] = rx, ry

            # Free capacity (remaining slots)
            # try:
            #     robot = env_state.robots[rid_s]
            #     capacity = max(1, getattr(robot, "capacity", max_robot_capacity))
            #     onboard = len(getattr(robot, "assigned_tasks", []))
            # except Exception:
            #     capacity = max_robot_capacity
            #     onboard = 0
            
            # free_capacity = max(0, capacity - onboard)
            # if normalize_features:
            #     out[2] = float(free_capacity) / cap_scale
            # else:
            #     out[2] = float(free_capacity)
            try:
                robot      = env_state.robots[rid_s]
                max_cap    = int(robot.get("max_capacity", max_robot_capacity))
                cur_onboard = int(robot.get("current_capacity", 0))   # physically onboard
                free_cap   = max(0, max_cap - cur_onboard)
            except Exception:
                free_cap = max_robot_capacity
            if normalize_features:
                out[2] = float(free_cap) / cap_scale
            else:
                out[2] = float(free_cap)
            # Append optional features
            _append_node_type(out, "robot")
            _append_ego_robot(out, is_ego)
            
            return out

        # ====================================================================
        # TASK FEATURES
        # ====================================================================
        # Structure: [release_time, waiting_time, est_travel_time,
        #             pickup_x, pickup_y, (pickup_dx, pickup_dy),
        #             drop_x, drop_y, is_obsolete, is_assigned, ...]
        
        elif node_type == "task":
            _ = _normalize_rid(obj_a)  # Robot ID (context)
            t = _resolve_task(obj_b)
            if t is None:
                return out  # Return zeros if task not found

            # Timing features
            release_time    = float(t.get("release_time", 0.0))
            est_travel_time = float(t.get("est_travel_time", 0.0))
            waiting_time    = float(max(0.0, current_time - release_time))
                        
            if normalize_features:
                out[0] = release_time / time_scale
                out[1] = waiting_time / wait_scale
                out[2] = est_travel_time / travel_scale
            else:
                out[0] = release_time
                out[1] = waiting_time
                out[2] = est_travel_time

            # Location features
            px, py = _task_xy(str(obj_b), is_pickup=True)
            dx, dy = _task_xy(str(obj_b), is_pickup=False)
            
            if normalize_features:
                out[3], out[4] = px / pos_scale, py / pos_scale
            else:
                out[3], out[4] = px, py
            
            # Conditional relative pickup displacement
            if use_xy_pickup and not use_edge_rt:
                rx, ry = _robot_xy(_normalize_rid(obj_a))
                if normalize_features:
                    out[5] = float(px - rx) / pos_scale
                    out[6] = float(py - ry) / pos_scale
                    out[7], out[8] = dx / pos_scale, dy / pos_scale
                else:
                    out[5] = float(px - rx)
                    out[6] = float(py - ry)
                    out[7], out[8] = dx, dy
                
                # out[9] = 1.0 if bool(getattr(t, "is_obsolete", False)) else 0.0
                # out[10] = 1.0 if bool(getattr(t, "is_assigned", False)) else 0.0
                out[9] = 1.0 if bool(t.get("is_obsolete", False)) else 0.0
                out[10] = 1.0 if bool(t.get("is_assigned",  False)) else 0.0
            else:
                if normalize_features:
                    out[5], out[6] = dx / pos_scale, dy / pos_scale
                else:
                    out[5], out[6] = dx, dy
            
                out[7] = 1.0 if bool(t.get("is_obsolete", False)) else 0.0
                out[8] = 1.0 if bool(t.get("is_assigned",  False)) else 0.0
            # Append optional features
            _append_node_type(out, "task")
            _append_ego_robot(out, False)
            
            return out

        # ====================================================================
        # EDGE (ROUTE/TRAVEL) FEATURES
        # ====================================================================
        
        elif node_type == "edge_rt":
            rid = _normalize_rid(obj_a)
            task_id = str(obj_b) if obj_b is not None else None
            
            if rid is None or task_id is None:
                return np.zeros((len(edge_features),), dtype=np.float32)
            
            return _edge_rt_features(rid, task_id)

        # Fallback for unknown node types
        return out

    return feature_fn