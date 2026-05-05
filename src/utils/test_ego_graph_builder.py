import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

# Import your code
from src.utils.ego_graph_builder import build_padded_ego_batch
from src.utils.feature_fn import make_feature_fn, get_feature_names

# =============================================================================
# 1. Load generated data
# =============================================================================

data_dir = Path("data")

agents_np = np.load(data_dir / "agents.npy")
tasks_np = np.load(data_dir / "tasks_batch_0.npy", allow_pickle=True)
print(agents_np)
print(tasks_np)
# =============================================================================
# 2. Build minimal env_state compatible with feature_fn
# =============================================================================

class Robot:
    def __init__(self, rid, x, y):
        self.id = rid
        self.x = float(x)
        self.y = float(y)
        self.capacity = 2
        self.assigned_tasks = []
        self.planned_route = []


class Task:
    def __init__(self, row):
        self.id = str(int(row[0]))
        self.pickup_x = float(row[1])
        self.pickup_y = float(row[2])
        self.dropoff_x = float(row[3])
        self.dropoff_y = float(row[4])
        self.release_time = float(row[5])
        self.pickup_deadline = float(row[6])
        self.est_travel_time = float(row[7])
        self.drop_off_deadline = float(row[8])
        self.is_assigned = False
        self.is_obsolete = False


class EnvState:
    def __init__(self):
        self.robots = {}
        self.tasks = {}
        self.current_time = 0.0


env = EnvState()

# --- populate robots ---
robots = []
for a in agents_np:
    rid = f"r{int(a[0])}"
    robots.append(rid)
    env.robots[rid] = Robot(rid, a[1], a[2])

# --- populate tasks ---
tasks_dict = {}
for row in tasks_np:
    t = Task(row)
    tasks_dict[t.id] = t
    env.tasks[t.id] = t
robot_names, task_names = get_feature_names(
    use_node_type=True,
    use_ego_robot=True
)

robot_idx = {name: i for i, name in enumerate(robot_names)}
task_idx = {name: i for i, name in enumerate(task_names)}
# print(robots)
# =============================================================================
# 3. Candidate lists (simple baseline)
# =============================================================================

task_ids = list(tasks_dict.keys())
candidate_lists = [task_ids for _ in robots]

# =============================================================================
# 4. Create feature function (YOUR REAL ONE)
# =============================================================================

feature_fn = make_feature_fn(
    env_state=env,
    normalize_features=False,
    use_node_type=True,
    use_ego_robot=True,
    use_edge_rt=True,
)

# Compute feature dimension automatically
F = feature_fn(robots[0], None, "robot_ego").shape[0]

print(f"Feature dimension F = {F}")

# =============================================================================
# 5. Build graph batch
# =============================================================================

obs, cand_ids = build_padded_ego_batch(
    robots=robots,
    tasks=tasks_dict,
    candidate_lists=candidate_lists,
    N_max=30,
    E_max=100,
    K_max=10,
    F=F,
    G=0,
    feature_fn=feature_fn,
    two_hop=True,
    vicinity_m=50.0,
)

print("Graph built successfully.")
print("Node mask shape:", obs["node_mask"].shape)
print("Edge mask shape:", obs["edge_mask"].shape)

# =============================================================================
# 6. Visualization (with node types!)
# =============================================================================



# =============================================================================
# 6. Visualization (with node types!)
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

from src.utils.feature_fn import get_feature_names

robot_names, task_names = get_feature_names(
    use_node_type=True,
    use_ego_robot=True
)

print(robot_names)
print(task_names)

def plot_graph(obs, graph_idx, jitter=0.8):
    x = obs["x"][graph_idx]
    node_mask = obs["node_mask"][graph_idx]

    plt.figure(figsize=(7, 7))

    for i in range(len(node_mask)):
        if not node_mask[i]:
            continue

        # detect type robustly
        is_robot = x[i][robot_idx["is_robot"]]
        is_task  = x[i][task_idx["is_task"]]
        is_ego   = x[i][robot_idx["is_ego_robot"]]

        # =========================
        # TASK NODE
        # =========================
        if is_task == 1:
            px = x[i][3]
            py = x[i][4]

            px += np.random.uniform(-jitter, jitter)
            py += np.random.uniform(-jitter, jitter)

            color = "blue"
            size = 80
            label = "task"

        # =========================
        # ROBOT NODE
        # =========================
        else:
            px = x[i][0]
            py = x[i][1]

            px += np.random.uniform(-jitter, jitter)
            py += np.random.uniform(-jitter, jitter)

            if is_ego == 1:
                color = "red"
                size = 160
                label = "ego"
            else:
                color = "orange"
                size = 120
                label = "robot"

        plt.scatter(px, py, c=color, s=size)

        plt.text(px, py, str(i), fontsize=8)

    plt.title(f"Ego Graph {graph_idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

for i in range(min(10, obs["x"].shape[1])):
    print(i, obs["x"][0, i, -3:], "task_flag=", obs["x"][0, i, -2])
# Plot a few graphs
for i in range(min(3, len(robots))):
    plot_graph(obs, i)

# =============================================================================
# 7. Sanity checks (VERY IMPORTANT)
# =============================================================================

for i in range(len(robots)):
    assert obs["node_mask"][i, 0] == 1, "Ego robot missing!"
    assert obs["cand_mask"][i].sum() > 0, "No candidates!"

print("All sanity checks passed ✅")



robot_names, task_names = get_feature_names(
    use_node_type=True,
    use_ego_robot=True
)

task_idx = {name: i for i, name in enumerate(task_names)}

x = obs["x"][0]   # pick graph 0
node_mask = obs["node_mask"][0]

for i in range(len(node_mask)):
    if not node_mask[i]:
        continue

    is_task = x[i][task_idx["is_task"]]
    is_robot = x[i][task_idx["is_robot"]]

    if is_task == 1:
        print(f"\nNODE {i} (TASK)")
        print("is_task:", is_task)
        print("is_robot:", is_robot)
        print("full vector:", x[i])