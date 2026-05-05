import numpy as np
from heapq import heappop, heappush
import math
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation
from PIL import Image
def preprocess_image(image):
    """Convert grayscale image to binary (black/white)."""
    return image.point(lambda p: 0 if p < 255 else 255)

def is_exact(trajectory, task):
    """Check if the last point in the trajectory is close to the task location."""
    return np.sqrt((trajectory[-1][0] - task[0]) ** 2 + (trajectory[-1][1] - task[1]) ** 2) < 1

def calculate_distance(point_a, point_b):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point_a) - np.array(point_b))

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def astar(grid, start, goal):
    # print('Starting A* algorithm i n utils...')
    rows ,cols= grid.shape
    # print('Grid shape: row, col', grid.shape)
    open_set = [(0, start)]  # (cost, (x, y))
    g_cost = {start: 0}
    came_from = {}

    while open_set:
        _, current = heappop(open_set)
        # print('Current node:', current)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            # print('Path found:', path[::-1])
            return True, path[::-1]  # Return reversed path

        x, y = current
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (x+dx, y+dy)
            # print('grid[neighbor]' , grid[neighbor])
            # print('Checking neighbor:', neighbor)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and not grid[neighbor]:
                # print('Valid neighbor:', neighbor)
                new_cost = g_cost[current] + 1
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priority = new_cost + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])  # Heuristic
                    heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
    return False, None

def smooth_astar_path(path, Planning_resolution, origin_x, origin_y, average_velocity):
    """Smooths the A* path using cubic splines and computes heading."""
    if len(path) < 3:  # Not enough points to smooth
        return [(x, y, 0) for x, y in path]

    # Convert (x, y) grid indices to real-world coordinates
    real_path = [from_index_to_real_point(x, y, Planning_resolution, origin_x, origin_y) for y,x in path]

    # Extract x and y separately
    x_vals = np.array([p[0] for p in real_path])
    y_vals = np.array([p[1] for p in real_path])

    # Parameter t for interpolation
    t = np.linspace(0, 1, len(real_path))

    # Fit cubic splines
    cs_x = CubicSpline(t, x_vals)
    cs_y = CubicSpline(t, y_vals)

    # Compute path length and determine the number of points
    path_length = np.sum(np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2))
    num_points = int(path_length / average_velocity) + 2  # More points for smoother motion

    # Generate smooth waypoints
    t_fine = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_fine)
    y_smooth = cs_y(t_fine)

    # Compute smooth heading (θ)
    smooth_path = []
    for i in range(len(t_fine) - 1):
        x1, y1 = x_smooth[i], y_smooth[i]
        x2, y2 = x_smooth[i + 1], y_smooth[i + 1]
        heading = math.atan2(y2 - y1, x2 - x1)  # Compute heading

        # Convert back to grid indices
        x_index, y_index = from_real_point_to_index(x1, y1, Planning_resolution, origin_x, origin_y)
        smooth_path.append([x1, y1, heading])

    # Add the last point
    x_last, y_last = x_smooth[-1], y_smooth[-1]
    heading_last = smooth_path[-1][2]  # Use last computed heading
    smooth_path.append([x_last, y_last, heading_last])

    return smooth_path

def discritized_path(solution_path_lines, average_velocity, resolution, origin_x, origin_y):
    """
    # generate all the cells that are on the path to compute the cust further
    """
    real_path = []
    for line in solution_path_lines:
        if line.strip():
            x, y, theta = [float(x) for x in line.strip().split()]
            real_x, real_y = from_index_to_real_point(x, y, resolution, origin_x, origin_y)
            real_path.append((real_x, real_y, theta))

    # Extract x, y, and headings
    x = np.array([p[0] for p in real_path])
    y = np.array([p[1] for p in real_path])
    headings = np.array([p[2] for p in real_path])

    # Parameter t
    t = np.linspace(0, 1, len(real_path))

    # Create cubic splines
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    path_length = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Calculate number of points based on average velocity
    num_points = int(path_length / average_velocity) + 2

    # Generate fine-grained trajectory
    t_fine = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_fine)
    y_smooth = cs_y(t_fine)
    crossed_cells_grid = []
    crossed_cells_real = []

    unique_cells = set()

    for i in range(len(t_fine) - 1):
        x1, y1 = x_smooth[i], y_smooth[i]
        x2, y2 = x_smooth[i + 1], y_smooth[i + 1]

        if i < len(t_fine) - 1:
            heading = math.atan2(y2 - y1, x2 - x1)
        else:
            heading = headings[-1]

        # Convert to grid indices
        x_index1, y_index1 = from_real_point_to_index(x1, y1, resolution, origin_x, origin_y)
        x_index2, y_index2 = from_real_point_to_index(x2, y2, resolution, origin_x, origin_y)

        # Use Bresenham to get cells in grid indices
        cells = bresenham(x_index1, y_index1, x_index2, y_index2)
        for cell in cells:
            if tuple(cell) not in unique_cells:
                # Convert the grid indices back to real coordinates
                real_x, real_y = from_index_to_real_point(cell[0], cell[1], resolution, origin_x, origin_y)
                crossed_cells_real.append([real_x, real_y, heading])
                crossed_cells_grid.append([cell[0], cell[1], heading])
                unique_cells.add(tuple(cell))

        # interpolated_path.append([x1, y1, heading])

    # Process the last point
    x_last, y_last = x_smooth[-1], y_smooth[-1]
    heading_last = headings[-1]
    x_index_last, y_index_last = from_real_point_to_index(x_last, y_last, resolution, origin_x, origin_y)
    crossed_cells_real.append([x_last, y_last, heading_last])
    crossed_cells_grid.append([x_index_last, y_index_last, heading_last])

    return crossed_cells_real, crossed_cells_grid

def from_index_to_real_point(x, y, Planning_resolution, origin_x, origin_y):
    """
    convert index of grid map to real value
    """
    real_x = float(x) * Planning_resolution + origin_x
    real_y = origin_y - float(y) * Planning_resolution
    return real_x, real_y


def from_real_point_to_index(x, y, Planning_resolution, origin_x, origin_y):
    """
    convert real value to index(int) of grid map to
    """

    grid_x = abs(round((float(x) - origin_x) / Planning_resolution))
    grid_y = abs(round((float(y) - origin_y) / Planning_resolution))
    return grid_x, grid_y


def bresenham(x1, y1, x2, y2):
    cells = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        cells.append([x1, y1])
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return cells

def enlarge_obstacles(image, dilation_radius):
    """
    Enlarge obstacles on the map by dilating the obstacle regions.

    :param image: Input grayscale image with obstacles.
    :param dilation_radius: Radius of the dilation operation.
    :return: Image with enlarged obstacles.
    """
    binary_image = preprocess_image(image)
    binary_array = np.array(binary_image)
    # Black (obstacles) are now True, white (free space) is False
    inverted_array = binary_array == 0

    # Create a structure element for dilation (a disk with the given radius)
    structure_element = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))

    # Perform dilation
    dilated_array = binary_dilation(inverted_array, structure=structure_element).astype(np.uint8) * 255

    # Convert the numpy array back to a PIL image
    return Image.fromarray(dilated_array)
