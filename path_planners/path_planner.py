from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

import path_planners.utils.math_utils as MathUtils


class PathPlanner:
    def __init__(
        self,
        goal_reach_dist_threshold: float = 0.5,
        goal_reach_angle_threshold: float = 0.1 * np.pi,
    ):
        self.goal_reach_dist_threshold = goal_reach_dist_threshold
        self.goal_reach_angle_threshold = goal_reach_angle_threshold

        self.occupancy_map = None
        self.occupancy_map_resolution = None
        self.occupancy_map_origin = None
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None

        self.path = None

    def set_occupancy_map(
        self,
        occupancy_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float, float],
    ):
        """
        Parameters:
        - occupancy_map: 2D numpy array representing the occupancy grid
        - resolution: The resolution of the map (meters per pixel)
        - origin: Origin of the map in world coordinates (x, y, yaw)
        """
        self.occupancy_map = occupancy_map
        self.occupancy_map_resolution = resolution
        self.occupancy_map_origin = origin
        self.x_min = origin[0]
        self.y_min = origin[1]
        self.x_max = origin[0] + resolution * occupancy_map.shape[1]
        self.y_max = origin[1] + resolution * occupancy_map.shape[0]

    def plan_global_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        render=False,
    ) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Plan a global path from start to goal pose
        :param start_pose: start pose
        :param goal_pose: goal pose
        :param render: render the path
        :return: success, path, number of nodes sampled
        """
        path, num_nodes_sampled = self._plan_path(start_pose, goal_pose)

        if render:
            self._render_path(path, start_pose, goal_pose)

        success = path is not None
        if path is None:
            path = []

        return success, path, num_nodes_sampled

    def _render_path(
        self,
        path,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ):
        # Create a figure and axis
        plt.figure(figsize=(8, 8))

        # Define a color map for occupancy
        # - Free space (0) -> white
        # - Occupied space (100) -> black
        # - Unknown space (-1) -> gray
        cmap = plt.cm.gray
        cmap.set_under(color="white")  # Color for free space
        cmap.set_over(color="black")  # Color for occupied space
        cmap.set_bad(color="lightgray")  # Color for unknown space

        # Mask unknown values (-1) so they appear as "bad" (gray) values in the plot
        masked_grid = np.ma.masked_where(self.occupancy_map == -1, self.occupancy_map)

        # Visualize the grid with the specified colormap
        plt.imshow(
            masked_grid,
            cmap=cmap,
            origin="lower",
            interpolation="none",
            extent=[
                self.occupancy_map_origin[0],
                self.occupancy_map_origin[0]
                + self.occupancy_map.shape[1] * self.occupancy_map_resolution,
                self.occupancy_map_origin[1],
                self.occupancy_map_origin[1]
                + self.occupancy_map.shape[0] * self.occupancy_map_resolution,
            ],
        )

        # plot path
        if path is not None:
            plt.plot([x[1] for x in path], [x[0] for x in path])
        # plot start and goal position
        plt.plot(start_pose[0], start_pose[1], "go")
        plt.plot(goal_pose[0], goal_pose[1], "ro")

        # Add labels and title
        plt.title("Occupancy Grid Map")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")

        # Display color bar for reference
        cbar = plt.colorbar()
        cbar.set_label("Occupancy Value")
        cbar.set_ticks([0, 50, 100])
        cbar.set_ticklabels(["Free", "Unknown", "Occupied"])

        # Show the plot
        plt.grid(False)
        plt.show()

    def _plan_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Plan a path from start to goal pose
        :param start_pose: start pose
        :param goal_pose: goal pose
        :return: path, number of nodes sampled
        """
        # this method should be implemented in the child class
        raise NotImplementedError

    def _check_collision(
        self,
        path: List[Tuple[float, float, float]],
    ) -> bool:
        """
        Return True if there is a collision between the near node and the new node. Otherwise, return False.
        """
        return False

    def _check_goal_reached(
        self,
        new_node_pos: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> bool:
        """
        Return True if the goal is reached. Otherwise, return False.
        """
        dist = np.linalg.norm(np.array(new_node_pos[:2]) - np.array(goal_pose[:2]))
        angle_diff = MathUtils.normalize_angle(new_node_pos[2] - goal_pose[2])
        return False
