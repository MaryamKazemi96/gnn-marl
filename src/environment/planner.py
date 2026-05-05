from ompl import base as ob
from ompl import geometric as og
import numpy as np
from PIL import Image
from utils import enlarge_obstacles, astar

class Planner(og.SimpleSetup):
    def __init__(self, map_image_path, dilation_radius, static_obstacle_threshold):
        super().__init__(ob.SE2StateSpace())
        self.map = Image.open(map_image_path).convert('L')
        self.dilation_radius = dilation_radius
        self.static_obstacle_threshold = static_obstacle_threshold
        self.width, self.height = self.map.size
        self.obstacle_grid = self.get_obstacle_grid()

        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, 0)
        bounds.setHigh(0, self.width)
        bounds.setLow(1, 0)
        bounds.setHigh(1, self.height)
        self.getStateSpace().setBounds(bounds)
        self.getSpaceInformation().setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

    def get_obstacle_grid(self):
        enlarged_obstacle_map = enlarge_obstacles(self.map, self.dilation_radius)
        obstacle_grid = np.zeros((self.width, self.height), dtype=bool)
        for x in range(self.width):
            for y in range(self.height):
                if enlarged_obstacle_map.getpixel((x, y)) < self.static_obstacle_threshold:
                    obstacle_grid[x, y] = True
        return obstacle_grid

    def is_point_valid(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return not self.obstacle_grid[x, y]

    def is_state_valid(self, state):
        x, y = state[0], state[1]
        return self.is_point_valid(x, y)

    def get_plan(self, start, goal):
        start_state = ob.State(self.getStateSpace())
        start_state.setX(start[0])
        start_state.setY(start[1])
        goal_state = ob.State(self.getStateSpace())
        goal_state.setX(goal[0])
        goal_state.setY(goal[1])

        self.setStartAndGoalStates(start_state, goal_state)
        self.setPlanner(og.AITstar(self.getSpaceInformation()))
        solved = self.solve(2.0)

        if solved:
            path = self.getSolutionPath()
            return path
        else:
            return None

    def get_plan_astar(self, start, goal):
        grid = np.array(self.obstacle_grid)
        solved, path = astar(grid, start, goal)
        return solved, path