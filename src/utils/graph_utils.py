import numpy as np
def get_edge_idx_graphmain(attributes_matrix, n_tasks, n_robots, radius=20, use_true_id=True, previous_connected_dict={}):
    # print(attributes_matrix, 'attributes matrix in get_edge_idx_graph')
    list_ego_graphs = {}
    dict_tid_2_robots = {}

    all_coords = attributes_matrix[:, 1:3]
    all_r_coords = all_coords[:n_robots]
    all_t_coords = all_coords[n_robots:].reshape(n_tasks, 1, 2)
    # --- NEW: remove assigned tasks from consideration ---
    task_assigned_flags = attributes_matrix[n_robots:, -1]  # assuming last column is is_assigned
    available_task_idx = np.where(task_assigned_flags == 0)[0]  # only unassigned tasks
    # print('[debug] available_task_idx:', available_task_idx)
    list_idx_in_matrix = np.arange(len(attributes_matrix))
    list_true_ids = attributes_matrix[:, 0]
    # print('[debug] list_idx_in_matrix', list_idx_in_matrix)
    # print('[debug] list_true_ids', list_true_ids)
    list_mapping = [list_idx_in_matrix, list_true_ids]
    # distance_m = within_distance(all_t_coords, all_r_coords, radius=radius)
    distance_m = within_distance(all_t_coords[available_task_idx], all_r_coords, radius=radius)
    
    # print('[debug] distance m', distance_m)
    tid_matrix_offset0, rid_matrix = np.nonzero(distance_m)
    tid_matrix = tid_matrix_offset0 + n_robots
    unique_tid, n_connected_r = np.unique(tid_matrix, return_counts=True)
    # print('[debug] unique_tid:', unique_tid , 'n_connected_r:', n_connected_r)
    tid_true = list_true_ids[unique_tid]
    rid_true = list_true_ids[rid_matrix]
    previous_num_n =0
    for idx, tid_m in enumerate(unique_tid):
        # print('[debug] idx:', idx , 'tid_m:', tid_m)
        num_n = n_connected_r[idx]
        
        n_r_offset = idx + num_n
        tid_t = tid_true[idx]
        # rid_m = rid_matrix[idx:n_r_offset]
        # rid_t = rid_true[idx:n_r_offset]
        rid_m = rid_matrix[previous_num_n:previous_num_n + num_n]
        rid_t = rid_true[previous_num_n:previous_num_n + num_n]
        dict_tid_2_robots[tid_t] = rid_t
        # print('[debug] rid_m:', rid_m, 'rid_t:', rid_t)
        # print('[debug] dict_tid_2_robots:', dict_tid_2_robots)
        if use_true_id:
            tid = tid_t
            rid = rid_t
        else:
            tid = tid_m
            rid = rid_m
        tids_dup = np.repeat(tid, num_n)
        pairs = np.column_stack((rid, tids_dup)).astype(int)
        for _id in rid:
            if _id in list_ego_graphs:
                list_ego_graphs[_id].append(pairs)
            else:
                list_ego_graphs[_id] = []
                list_ego_graphs[_id].append(pairs)
        previous_num_n += num_n
    return list_ego_graphs, dict_tid_2_robots, list_mapping


def within_distancemain(coord_a, coord_b, radius):
    d = (coord_a - coord_b)
    # print('[debug] coord_a shape:', coord_a.shape, 'coord_b shape:', coord_b.shape)
    # print('[debug] d :''d:', d)
    dist = np.linalg.norm(d, axis=-1)
    added2robot = (np.abs(dist) < radius)
    return added2robot.astype(int)


def update_shared_attribute_matrixmain(attribute_array, current_robots_array_padded, current_task_array):
    old_array_true_ids = attribute_array[:, 0]
    current_task_ids = current_task_array[:, 0]
    current_robot_ids = current_robots_array_padded[:, 0]
    common_task_ids = list(set(old_array_true_ids) & set(current_task_ids))
    common_task_ids_index = np.where(np.isin(attribute_array[:, 0], common_task_ids))[0]
  
    attribute_array[:len(current_robots_array_padded), :] = current_robots_array_padded
    attribute_array[common_task_ids_index, :] = current_task_array[np.where(np.isin(current_task_array[:, 0], common_task_ids))[0], :]

    new_tasks_true_ids = list(set(current_task_ids) - set(old_array_true_ids))
    new_tasks_rows = current_task_array[np.where(np.isin(current_task_array[:, 0], new_tasks_true_ids))[0], :]
    attribute_array = np.vstack([attribute_array, new_tasks_rows])
    mapping_list = [np.arange(len(attribute_array)), attribute_array[:, 0]]

    return attribute_array, mapping_list


def delete_taskid_in_graphmain(list_ego_graphs, list_tid2remove):
    for rid, ego_graph in list_ego_graphs.items():
        # print(list_tid2remove, 'list of task ids to remove')
        # print('[debug] Before deletion, ego_g for rid', rid, ':', ego_graph)
        ego_graph = [
            g for g in ego_graph
            if g[0, 1] not in list_tid2remove
        ]
        list_ego_graphs[rid] = ego_graph

def remove_robot_edges(ego_graphs, full_capacity_robot_ids):
    """
    Remove all edges for robots that have reached their maximum capacity.

    Args:
        ego_graphs (dict): A dictionary where keys are robot IDs and values are lists of ego edges.
        full_capacity_robot_ids (list): List of robot IDs that have reached their maximum capacity.
    """
    # Convert robot IDs to native Python integers for compatibility
    full_capacity_robot_ids = [int(robot_id) for robot_id in full_capacity_robot_ids]

    for robot_id in full_capacity_robot_ids:
        if robot_id in ego_graphs:
            # Set the ego graph for the robot to an empty list
            ego_graphs[robot_id] = []
            
def pad_matrixmain(matrix, n_features=11):
    """Pad a matrix to ensure it has a specific number of features."""
    num, feature_dim = matrix.shape
    padding_size = n_features - feature_dim
    padding = np.zeros((num, padding_size))
    return np.column_stack((matrix, padding))

import numpy as np
def get_edge_idx_graph(attributes_matrix, n_tasks, n_robots, radius=20, use_true_id=True, previous_connected_dict={}):
    # (unchanged)
    list_ego_graphs = {}
    dict_tid_2_robots = {}

    all_coords = attributes_matrix[:, 1:3]
    all_r_coords = all_coords[:n_robots]
    all_t_coords = all_coords[n_robots:].reshape(n_tasks, 1, 2)
    # --- NEW: remove assigned tasks from consideration ---
    # task_assigned_flags = attributes_matrix[n_robots:, -1]  # assuming last column is is_assigned
    # available_task_idx = np.where(task_assigned_flags == 0)[0]  # only unassigned tasks
    # print('[debug] available_task_idx:', available_task_idx)
    list_idx_in_matrix = np.arange(len(attributes_matrix))
    list_true_ids = attributes_matrix[:, 0]
    list_mapping = [list_idx_in_matrix, list_true_ids]
    # distance_m = within_distance(all_t_coords[available_task_idx], all_r_coords, radius=radius)
    distance_m = within_distance(all_t_coords, all_r_coords, radius=radius)
    
    # print('[debug] distance m', distance_m) 
    tid_matrix_offset0, rid_matrix = np.nonzero(distance_m)
    # print('[debug] tid_matrix_offset0:', tid_matrix_offset0 , 'rid_matrix:', rid_matrix)
    tid_matrix = tid_matrix_offset0 + n_robots
    unique_tid, n_connected_r = np.unique(tid_matrix, return_counts=True)
    tid_true = list_true_ids[unique_tid]
    rid_true = list_true_ids[rid_matrix]
    # print('[debug] unique_tid:', unique_tid , 'n_connected_r:', n_connected_r)
    # print('[debug] tid_true:', tid_true , 'rid_true:', rid_true)
    previous_num_n =0
    for idx, tid_m in enumerate(unique_tid):
        num_n = n_connected_r[idx]
        n_r_offset = idx + num_n
        tid_t = tid_true[idx]
        rid_m = rid_matrix[previous_num_n:previous_num_n + num_n]
        rid_t = rid_true[previous_num_n:previous_num_n + num_n]
        dict_tid_2_robots[tid_t] = rid_t
        if use_true_id:
            tid = tid_t
            rid = rid_t
        else:
            tid = tid_m
            rid = rid_m
        tids_dup = np.repeat(tid, num_n)
        pairs = np.column_stack((rid, tids_dup)).astype(int)
        for _id in rid:
            if _id in list_ego_graphs:
                list_ego_graphs[_id].append(pairs)
            else:
                list_ego_graphs[_id] = []
                list_ego_graphs[_id].append(pairs)
        previous_num_n += num_n
    return list_ego_graphs, dict_tid_2_robots, list_mapping


def within_distance(coord_a, coord_b, radius):
    d = (coord_a - coord_b)
    dist = np.linalg.norm(d, axis=-1)
    added2robot = (np.abs(dist) < radius)
    return added2robot.astype(int)


def update_shared_attribute_matrix(attribute_array, current_robots_array_padded, current_task_array):
    """
    Robustly rebuild the shared attributes matrix from current robot rows and current task rows.

    - Pads robot/task rows to a common feature dimension if needed.
    - Stacks robots then tasks, returning the new attribute array and a mapping list
      [np.arange(n_rows), true_id_array].
    This avoids fragile index-based in-place updates which can fail if the old
    matrix contains duplicate IDs or different ordering.
    """
    # Ensure arrays are numpy arrays
    robots = np.asarray(current_robots_array_padded)
    tasks = np.asarray(current_task_array)
    # print('[debug] Current robots and tasks before padding:', robots, tasks)
    # If tasks is a 1-D empty array (e.g., np.zeros((0,))), convert to (0, feat)
    if tasks.ndim == 1 and tasks.size == 0:
        tasks = tasks.reshape(0, robots.shape[1])

    # Determine final feature dimension and pad if needed
    n_feat = max(robots.shape[1], tasks.shape[1] if tasks.ndim > 1 else 0)

    def _pad(arr, target_cols):
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = arr.shape[1]
        if cols < target_cols:
            pad = np.zeros((arr.shape[0], target_cols - cols), dtype=arr.dtype)
            return np.hstack((arr, pad))
        return arr

    robots_p = _pad(robots, n_feat)
    tasks_p = _pad(tasks, n_feat) if tasks.size != 0 else np.zeros((0, n_feat), dtype=robots_p.dtype)
    # print('[debug] Padded robots and padded tasks:', robots_p, tasks_p)
    # Stack robots then tasks
    attribute_array_new = np.vstack((robots_p, tasks_p)) if tasks_p.shape[0] > 0 else robots_p.copy()

    # Build mapping: indices and true ids (first column)
    mapping_list = [np.arange(len(attribute_array_new)), attribute_array_new[:, 0]]
    # print('[debug] Updated attribute array:', attribute_array_new, mapping_list)
    return attribute_array_new, mapping_list


def delete_taskid_in_graph(list_ego_graphs, list_tid2remove):
    for rid, ego_graph in list_ego_graphs.items():
        ego_graph = [
            g for g in ego_graph
            if g[0, 1] not in list_tid2remove
        ]
        list_ego_graphs[rid] = ego_graph

def delete_taskid_in_graph2(list_ego_graphs, list_tid2remove):
    """
    Remove edges corresponding to assigned tasks from ego graphs.
    list_ego_graphs[rid] is a list of (N_i x 2) arrays.
    """
    for rid, ego_g_list in list_ego_graphs.items():
        new_ego_g = []

        for edge_block in ego_g_list:
            if edge_block.size == 0:
                continue

            # keep rows whose task id is NOT removed
            mask = ~np.isin(edge_block[:, 1], list_tid2remove)
            filtered = edge_block[mask]

            if filtered.shape[0] > 0:
                new_ego_g.append(filtered)

        list_ego_graphs[rid] = new_ego_g

def pad_matrix(matrix, n_features=11):
    """Pad a matrix to ensure it has a specific number of features."""
    num, feature_dim = matrix.shape
    padding_size = n_features - feature_dim
    padding = np.zeros((num, padding_size))
    return np.column_stack((matrix, padding))