

import unittest
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.graph_utils import get_edge_idx_graph, update_shared_attribute_matrix, pad_matrix

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


from PIL import Image
def label_point(x, y, text, dy=0.5):
    plt.text(
        x,
        y - dy,            # move label slightly above the point
        text,
        fontsize=9,
        ha="center",
        va="bottom",
        color="black",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
    )

def plot_ego_graphs(ego_graphs, attributes_matrix, batch_index, map_path, origin_x, origin_y, map_resolution):
    """
    Plot ego graphs for visualization with distinct colors for each robot's ego graph and a map background.
    """
    # Load the map image
    map_img = Image.open(map_path)

    # Get the map dimensions in meters
    map_width, map_height = map_img.size
    map_width_in_meters = map_width * map_resolution
    map_height_in_meters = map_height * map_resolution
    map_extent = [
        origin_x,  # Left (x_min)
        origin_x + map_width_in_meters,  # Right (x_max)
        origin_y,  # Bottom (y_min)
        origin_y + map_height_in_meters  # Top (y_max)
    ]

    plt.figure(figsize=(10, 10))

    # Display the map as the background with origin at the top-left
    plt.imshow(map_img, extent=map_extent, origin="lower", cmap="gray")
    plt.gca().invert_yaxis()

    # Generate a colormap with as many colors as there are robots
    cmap = get_cmap("tab10", len(ego_graphs))  # Use a colormap with 10 distinct colors

    for i, (robot_id, ego_graph) in enumerate(ego_graphs.items()):  # Iterate over robots and their ego graphs
        # Find the row index of the robot in the attributes matrix
        robot_row_index = np.where(attributes_matrix[:, 0] == robot_id)[0][0]
        robot_coords = attributes_matrix[robot_row_index, 1:3]  # Robot coordinates (w, h)

        # Assign a unique color to this robot's ego graph
        color = cmap(i)

        # Plot the robot (circle shape)
        plt.scatter(*robot_coords, color=color, marker='o', label=f"Robot {int(robot_id)}")
        label_point(
            robot_coords[0],
            robot_coords[1],
            f"R{int(robot_id)}")

        # Plot edges and tasks connected to this robot
        for edge_array in ego_graph:
            for edge in edge_array:
                task_id = int(edge[1])  # Extract task ID from the edge
                # Find the row index of the task in the attributes matrix
                task_row_index = np.where(attributes_matrix[:, 0] == task_id)[0][0]
                task_coords = attributes_matrix[task_row_index, 1:3]  # Task coordinates (w, h)
                # plt.scatter(*task_coords, color=color, marker='^', edgecolor="black")
                plt.scatter(*task_coords, color=color, marker='^', edgecolor="black")
                label_point(
                    task_coords[0],
                    task_coords[1],
                    f"T{task_id}"
)

                plt.plot([robot_coords[0], task_coords[0]], [robot_coords[1], task_coords[1]], color=color, linestyle='--')

    plt.title(f"Ego Graphs for Batch {batch_index}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()  # Only robots will appear in the legend
    plt.grid()
    plt.show()

def test_ego_graph_generation():
    # Define the data directory
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # Load agents and tasks
    agents = np.load(data_dir / "agents.npy")
    tasks = np.load(data_dir / "tasks_batch_0.npy")  # Load the first batch of tasks
    print("tasks:", tasks)
    print("agents:", agents)
    n_features = max(agents.shape[1], tasks.shape[1])  # Determine the maximum number of features
    agents = pad_matrix(agents, n_features=n_features)
    tasks = pad_matrix(tasks, n_features=n_features)
    # Combine agents and tasks into the attributes matrix
    attributes_matrix = np.vstack((agents, tasks))
    n_robots = len(agents)
    n_tasks = len(tasks)
    radius = 100  # Define the radius for ego graph construction
    print(attributes_matrix)
    # Generate ego graphs
    list_ego_graphs, dict_tid_2_robots, list_mapping = get_edge_idx_graph(
        attributes_matrix, n_tasks, n_robots, radius
    )

    # print("list_ego_graphs:", list_ego_graphs)

    # Assertions to validate the ego graph structure
    assert len(list_ego_graphs) > 0, "Ego graphs should not be empty."
    assert len(dict_tid_2_robots) > 0, "Task-to-robot mapping should not be empty."

    # Plot ego graphs for visualization
    # plot_ego_graphs(list_ego_graphs, attributes_matrix, batch_index=0)
    map_path = Path(__file__).resolve().parent.parent / "env" / "osaka2d.png"
    plot_ego_graphs(list_ego_graphs, attributes_matrix, batch_index=0, map_path=map_path,
                    origin_x=0, origin_y=0, map_resolution=0.05)

if __name__ == "__main__":
    test_ego_graph_generation()