"""
Graph construction utilities for batched ego-graphs in multi-agent task allocation.

build_padded_ego_batch(...) converts per-robot candidate task lists into a fixed-shape
batch of ego-graphs (robot + candidate tasks + optional competitor robots) suitable for 
GNN-based policies.

Graph structure per robot:
  - Node 0: ego robot
  - Nodes 1..K: candidate tasks for this robot
  - Nodes K+1..N: competitor robots (in 2-hop mode)
  
Edges:
  - Star edges: robot <-> candidate task (bidirectional)
  - Competitor edges (2-hop): candidate task <-> competitor robot
"""
from __future__ import annotations
from typing import Callable, Sequence, Any, List, Tuple, Optional, Dict
import numpy as np


def build_padded_ego_batchold(
    *,
    robots: Sequence[Optional[str]],
    tasks: Dict[str, Any],
    candidate_lists: Sequence[Sequence[str]],
    N_max: int,
    E_max: int,
    K_max: int,
    F: int,
    G: int,
    feature_fn: Callable[[Any, Any, str], np.ndarray],
    two_hop: bool = True,
    two_hop_directed: bool = False,
    normalize_features: bool = False,
    vicinity_m: float = 1000.0,
    use_edge_rt: bool = False,
    edge_feat_dim: int = 0,
    edge_features: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[List[Optional[str]]]]:
    """
    Build per-robot ego-graphs as batched observations for GNN policy.

    This function creates a batch of ego-centric graphs where each robot sees itself
    and its candidate tasks, optionally augmented with competing robots in vicinity
    (2-hop neighbors).

    Parameters
    ----------
    robots : Sequence[Optional[str]]
        List of robot IDs (strings). Length = R (number of robots).
        Can contain None for padded slots.

    tasks : Dict[str, Any]
        Dictionary mapping task_id -> task state object.
        Task state must have attributes accessible by feature_fn:
        - pickup_x, pickup_y (float): pickup location
        - dropoff_x, dropoff_y (float): dropoff location
        - release_time (float): when task was released
        - est_travel_time (float): estimated travel duration
        - is_assigned (bool): whether already assigned
        - is_obsolete (bool): whether task is obsolete

    candidate_lists : Sequence[Sequence[str]]
        For each robot, list of task IDs that are feasible candidates.
        Length = R. Inner lists can vary (0 .. K_max).

    N_max : int
        Maximum nodes per graph (robot + tasks + competitor robots).
        Node feature arrays padded to (R, N_max, F).

    E_max : int
        Maximum edges per graph.
        Edge arrays padded to (R, 2, E_max) or (R, E_max, edge_feat_dim).

    K_max : int
        Maximum candidate tasks per robot. Defines action space size.
        Candidate lists truncated to K_max.

    F : int
        Node feature dimension.

    G : int
        Global feature dimension (reserved for future use).

    feature_fn : Callable[[Any, Any, str], np.ndarray]
        Feature extraction function: feature_fn(robot_id_or_state, task_state_or_None, node_type)
        where node_type in {"robot", "robot_ego", "robot_other", "task", "edge_rt"}.
        Returns np.ndarray of shape (F,) for node types, or (edge_feat_dim,) for edges.

    two_hop : bool
        If True, include competitor robot nodes within vicinity_m of candidate tasks.

    two_hop_directed : bool
        If True, only add directed edges task -> competitor.
        If False, add bidirectional edges.

    normalize_features : bool
        Whether features are normalized (used for distance calculations).

    vicinity_m : float
        Distance threshold in meters for defining 2-hop competitors.
        Also used as position scale for normalization.

    use_edge_rt : bool
        Whether to compute edge features (e.g., travel time, distance).

    edge_feat_dim : int
        Dimension of edge features. 0 if use_edge_rt=False.

    edge_features : Optional[List[str]]
        Names of edge features (e.g., ["dx", "dy", "eta"]).
        Used to identify special features like "is_ego_edge".

    Returns
    -------
    obs : Dict[str, np.ndarray]
        Dictionary with keys:
        - "x": node features (R, N_max, F)
        - "node_mask": valid node indicator (R, N_max) [0=padded, 1=valid]
        - "edge_index": edge connectivity (R, 2, E_max)
        - "edge_mask": valid edge indicator (R, E_max)
        - "edge_attr": edge features (R, E_max, edge_feat_dim) [if use_edge_rt]
        - "cand_idx": node indices of candidate tasks (R, K_max)
        - "cand_mask": valid candidate indicator (R, K_max)

    cand_task_ids : List[List[Optional[str]]]
        Parallel list mapping graph slots to external task IDs.
        cand_task_ids[r][k] = task_id or None (for padding).
        Used by controller to map policy actions back to tasks.

    Notes
    -----
    Graph structure for each robot i:
      node 0: robot i (ego)
      nodes 1..m: candidate tasks (where m = min(K_max, # candidates))
      nodes m+1..n: competitor robots (only in 2-hop mode)

    Edge structure:
      Base edges (always):
        robot -> task, task -> robot (star topology)

      2-hop edges (if two_hop=True):
        task -> competitor_robot
        competitor_robot -> task (if not two_hop_directed)
    """
    R = len(robots)

    # Initialize output arrays
    x = np.zeros((R, N_max, F), dtype=np.float32)
    node_mask = np.zeros((R, N_max), dtype=np.uint8)
    edge_index = np.zeros((R, 2, E_max), dtype=np.int64)
    edge_mask = np.zeros((R, E_max), dtype=np.uint8)
    edge_attr = (
        np.zeros((R, E_max, edge_feat_dim), dtype=np.float32)
        if edge_feat_dim > 0
        else None
    )

    edge_features_list = list(edge_features or [])
    ego_edge_idx = (
        edge_features_list.index("is_ego_edge")
        if "is_ego_edge" in edge_features_list
        else None
    )

    # Candidate slot mapping (R, K_max)
    cand_idx = np.zeros((R, K_max), dtype=np.int64)
    cand_mask = np.zeros((R, K_max), dtype=np.uint8)

    # External task ID mapping (for controller)
    cand_task_ids: List[List[Optional[str]]] = [
        [None] * K_max for _ in range(R)
    ]

    vicinity_threshold = float(vicinity_m)
    pos_scale = max(1.0, float(vicinity_m))

    def _to_meters(xy: Tuple[float, float]) -> Tuple[float, float]:
        """Convert from normalized to meter coordinates if needed."""
        if normalize_features:
            return xy[0] * pos_scale, xy[1] * pos_scale
        return xy

    # Pre-compute robot positions for 2-hop distance checks
    robot_xy_cache: List[Tuple[float, float]] = []
    for rid in robots:
        try:
            if rid is None:
                robot_xy_cache.append((0.0, 0.0))
            else:
                rf = feature_fn(rid, None, "robot_other")
                robot_xy_cache.append(_to_meters((float(rf[0]), float(rf[1]))))
        except Exception:
            robot_xy_cache.append((0.0, 0.0))

    # ========================================================================
    # Build per-robot ego-graphs
    # ========================================================================

    for i in range(R):
        rid = robots[i]

        # Node 0: Ego robot
        try:
            if rid is not None:
                x[i, 0, :] = feature_fn(rid, None, "robot_ego")
            # else: leave as zeros for padded robot
        except Exception:
            pass
        node_mask[i, 0] = 1 if rid is not None else 0

        # Extract candidate task indices/IDs for this robot
        cands = list(candidate_lists[i]) if i < len(candidate_lists) else []

        # Cap candidates: leave space for robot (index 0) + competitors (2-hop)
        max_tasks_here = max(0, N_max - 1)  # conservative; competitors get added later
        cands = cands[: min(K_max, max_tasks_here)]

        # ====================================================================
        # Add task nodes and star edges
        # ====================================================================

        e_ptr = 0  # Edge counter
        next_node_id = 1 + len(cands)  # Next available node for competitors
        competitor_nodes: Dict[str, int] = {}  # Map competitor_rid -> node_id

        for local_slot, task_id in enumerate(cands):
            node_id = 1 + local_slot  # Node index for this task

            if node_id >= N_max:
                break  # Out of node space

            # Get task state
            t = tasks.get(task_id)
            print(f"Robot {rid} candidate task {task_id} state: {t}")
            if t is None:
                continue

            # Extract task features
            try:
                if rid is not None:
                    x[i, node_id, :] = feature_fn(rid, t, "task")
                # else: leave as zeros
            except Exception:
                pass

            node_mask[i, node_id] = 1

            # Cache task position for 2-hop competitor detection
            task_xy = None
            if two_hop:
                print(f"Robot {rid} candidate task {task_id} position extraction for 2-hop")
                try:
                    if rid is not None:
                        # tf = feature_fn(rid, t, "task")
                        # print(f"Extracted task features for 2-hop: {tf}")
                        # task_xy = _to_meters((float(tf[3]), float(tf[4])))
                        task_xy = (float(t["pickup_x"]), float(t["pickup_y"]))
                        print(f"Robot {rid} candidate task {task_id} position for 2-hop: {task_xy}")
                except Exception:
                    task_xy = None

            # Map candidate slot to node index
            cand_idx[i, local_slot] = node_id
            cand_mask[i, local_slot] = 1

            # Store external task ID for downstream controller
            try:
                # cand_task_ids[i][local_slot] = str(getattr(t, "id", None)) or None
                cand_task_ids[i][local_slot] = task_id
            
            except Exception:
                cand_task_ids[i][local_slot] = None

            # ================================================================
            # Add undirected star edges: robot <-> task
            # ================================================================

            if e_ptr + 2 <= E_max:
                # robot -> task
                edge_index[i, 0, e_ptr] = 0
                edge_index[i, 1, e_ptr] = node_id
                edge_mask[i, e_ptr] = 1

                if edge_attr is not None and use_edge_rt:
                    try:
                        if rid is not None:
                            edge_attr[i, e_ptr, :] = feature_fn(
                                rid, t, "edge_rt"
                            )
                            if ego_edge_idx is not None:
                                edge_attr[i, e_ptr, ego_edge_idx] = 1.0
                    except Exception:
                        pass

                e_ptr += 1

                # task -> robot
                edge_index[i, 0, e_ptr] = node_id
                edge_index[i, 1, e_ptr] = 0
                edge_mask[i, e_ptr] = 1

                if edge_attr is not None and use_edge_rt:
                    try:
                        if rid is not None:
                            edge_attr[i, e_ptr, :] = feature_fn(
                                rid, t, "edge_rt"
                            )
                            if ego_edge_idx is not None:
                                edge_attr[i, e_ptr, ego_edge_idx] = 1.0
                    except Exception:
                        pass

                e_ptr += 1

            # ================================================================
            # Add 2-hop edges: task <-> competitor robots in vicinity
            # ================================================================

            if two_hop and task_xy is not None:
                for j, other_rid in enumerate(robots):
                    if other_rid is None or j == i:
                        continue  # Skip None or self

                    # Check if competitor is within vicinity
                    rx, ry = robot_xy_cache[j]
                    dx = rx - task_xy[0]
                    dy = ry - task_xy[1]

                    dist_sq = dx * dx + dy * dy
                    if dist_sq > (vicinity_threshold * vicinity_threshold):
                        continue  # Out of vicinity

                    # Get or create competitor node
                    other_key = str(other_rid)
                    if other_key in competitor_nodes:
                        other_node_id = competitor_nodes[other_key]
                    else:
                        if next_node_id >= N_max:
                            continue  # Out of node space

                        other_node_id = next_node_id
                        next_node_id += 1
                        competitor_nodes[other_key] = other_node_id

                        # Add competitor robot node
                        try:
                            x[i, other_node_id, :] = feature_fn(
                                other_rid, None, "robot_other"
                            )
                        except Exception:
                            pass

                        node_mask[i, other_node_id] = 1

                    # Add task -> competitor edge
                    if e_ptr + 1 <= E_max:
                        edge_index[i, 0, e_ptr] = node_id
                        edge_index[i, 1, e_ptr] = other_node_id
                        edge_mask[i, e_ptr] = 1

                        if edge_attr is not None and use_edge_rt:
                            try:
                                if other_rid is not None:
                                    edge_attr[i, e_ptr, :] = feature_fn(
                                        other_rid, t, "edge_rt"
                                    )
                                    if ego_edge_idx is not None:
                                        edge_attr[
                                            i, e_ptr, ego_edge_idx
                                        ] = 0.0
                            except Exception:
                                pass

                        e_ptr += 1

                    # Add competitor -> task edge (if undirected)
                    if (not two_hop_directed) and (e_ptr + 1 <= E_max):
                        edge_index[i, 0, e_ptr] = other_node_id
                        edge_index[i, 1, e_ptr] = node_id
                        edge_mask[i, e_ptr] = 1

                        if edge_attr is not None and use_edge_rt:
                            try:
                                if other_rid is not None:
                                    edge_attr[i, e_ptr, :] = feature_fn(
                                        other_rid, t, "edge_rt"
                                    )
                                    if ego_edge_idx is not None:
                                        edge_attr[
                                            i, e_ptr, ego_edge_idx
                                        ] = 0.0
                            except Exception:
                                pass

                        e_ptr += 1

    # ========================================================================
    # Package observation
    # ========================================================================

    obs = dict(
        x=x,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_mask=edge_mask,
        cand_idx=cand_idx,
        cand_mask=cand_mask,
    )

    if edge_attr is not None:
        obs["edge_attr"] = edge_attr

    return obs, cand_task_ids

def build_padded_ego_batch(
    *,
    robots: Sequence[Optional[int]],
    robots_dict: Dict[int, Any],          # ← new: env's self.robots
    tasks: Dict[int, Any],
    candidate_lists: Sequence[Sequence[int]],
    N_max: int,
    E_max: int,
    K_max: int,
    F: int,
    G: int,
    feature_fn,
    two_hop: bool = False,
    two_hop_directed: bool = False,
    normalize_features: bool = False,
    vicinity_m: float = 1000.0,
    use_edge_rt: bool = False,
    edge_feat_dim: int = 0,
    edge_features=None,
):
    R = len(robots)

    x          = np.zeros((R, N_max, F),               dtype=np.float32)
    node_mask  = np.zeros((R, N_max),                   dtype=np.uint8)
    edge_index = np.zeros((R, 2, E_max),                dtype=np.int64)
    edge_mask  = np.zeros((R, E_max),                   dtype=np.uint8)
    edge_attr  = (np.zeros((R, E_max, edge_feat_dim),   dtype=np.float32)
                  if edge_feat_dim > 0 else None)

    edge_features_list = list(edge_features or [])
    ego_edge_idx = (edge_features_list.index("is_ego_edge")
                    if "is_ego_edge" in edge_features_list else None)

    cand_idx      = np.zeros((R, K_max), dtype=np.int64)
    cand_mask     = np.zeros((R, K_max), dtype=np.uint8)
    cand_task_ids = [[None] * K_max for _ in range(R)]

    # ── Pre-compute robot positions from actual robot state ───────────────
    robot_xy_cache = []
    for rid in robots:
        if rid is None:
            robot_xy_cache.append((0.0, 0.0))
        else:
            r = robots_dict.get(rid, {})
            robot_xy_cache.append((float(r.get("x", 0.0)), float(r.get("y", 0.0))))

    vicinity_sq = float(vicinity_m) ** 2

    # ── Per-robot ego-graph ───────────────────────────────────────────────
    for i in range(R):
        rid = robots[i]

        # Node 0: ego robot
        if rid is not None:
            try:
                x[i, 0, :] = feature_fn(rid, None, "robot_ego")
            except Exception:
                pass
            node_mask[i, 0] = 1

        cands = list(candidate_lists[i])[:K_max] if i < len(candidate_lists) else []

        e_ptr       = 0
        next_node   = 1                      # next free node slot
        competitor_nodes: Dict[str, int] = {}

        for local_slot, task_id in enumerate(cands):
            if next_node >= N_max:
                break

            t = tasks.get(task_id)
            if t is None:
                continue

            node_id     = next_node
            next_node  += 1

            # Task node features
            try:
                if rid is not None:
                    x[i, node_id, :] = feature_fn(rid, t, "task")
            except Exception:
                pass
            node_mask[i, node_id] = 1

            # Candidate slot
            cand_idx[i, local_slot]      = node_id
            cand_mask[i, local_slot]     = 1
            cand_task_ids[i][local_slot] = task_id

            # Star edges: ego ↔ task
            for src, dst in [(0, node_id), (node_id, 0)]:
                if e_ptr >= E_max:
                    break
                edge_index[i, 0, e_ptr] = src
                edge_index[i, 1, e_ptr] = dst
                edge_mask[i, e_ptr]     = 1
                if edge_attr is not None and use_edge_rt and rid is not None:
                    try:
                        edge_attr[i, e_ptr, :] = feature_fn(rid, t, "edge_rt")
                        if ego_edge_idx is not None:
                            edge_attr[i, e_ptr, ego_edge_idx] = 1.0
                    except Exception:
                        pass
                e_ptr += 1

            # 2-hop: competitor robots near this task's pickup
            if two_hop:
                tx = float(t["pickup_x"])
                ty = float(t["pickup_y"])

                for j, other_rid in enumerate(robots):
                    if other_rid is None or j == i:
                        continue

                    rx, ry = robot_xy_cache[j]
                    if (rx - tx)**2 + (ry - ty)**2 > vicinity_sq:
                        continue

                    # Get or create competitor node
                    other_key = str(other_rid)
                    if other_key not in competitor_nodes:
                        if next_node >= N_max:
                            continue
                        comp_node             = next_node
                        next_node            += 1
                        competitor_nodes[other_key] = comp_node

                        try:
                            x[i, comp_node, :] = feature_fn(other_rid, None, "robot_other")
                        except Exception:
                            pass
                        node_mask[i, comp_node] = 1
                    else:
                        comp_node = competitor_nodes[other_key]

                    # task → competitor (always)
                    if e_ptr < E_max:
                        edge_index[i, 0, e_ptr] = node_id
                        edge_index[i, 1, e_ptr] = comp_node
                        edge_mask[i, e_ptr]     = 1
                        if edge_attr is not None and use_edge_rt:
                            try:
                                edge_attr[i, e_ptr, :] = feature_fn(other_rid, t, "edge_rt")
                                if ego_edge_idx is not None:
                                    edge_attr[i, e_ptr, ego_edge_idx] = 0.0
                            except Exception:
                                pass
                        e_ptr += 1

                    # competitor → task (if undirected)
                    if not two_hop_directed and e_ptr < E_max:
                        edge_index[i, 0, e_ptr] = comp_node
                        edge_index[i, 1, e_ptr] = node_id
                        edge_mask[i, e_ptr]     = 1
                        if edge_attr is not None and use_edge_rt:
                            try:
                                edge_attr[i, e_ptr, :] = feature_fn(other_rid, t, "edge_rt")
                                if ego_edge_idx is not None:
                                    edge_attr[i, e_ptr, ego_edge_idx] = 0.0
                            except Exception:
                                pass
                        e_ptr += 1

    obs = dict(
        x=x, node_mask=node_mask,
        edge_index=edge_index, edge_mask=edge_mask,
        cand_idx=cand_idx, cand_mask=cand_mask,
    )
    if edge_attr is not None:
        obs["edge_attr"] = edge_attr

    return obs, cand_task_ids