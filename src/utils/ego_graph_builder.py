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