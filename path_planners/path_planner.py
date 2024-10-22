from typing import List, Tuple, Optional

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.gridmap_utils as GridmapUtils
import path_planners.utils.plot_utils as PlotUtils


class PathPlanner:
    def __init__(
        self,
        terminate_on_goal_reached: bool = True,
        goal_reach_dist_threshold: float = 0.5,
        goal_reach_angle_threshold: float = 0.1 * np.pi,
        occupancy_map_obstacle_padding_dist: float = 0.5,
    ):
        self._terminate_on_goal_reached = terminate_on_goal_reached
        self._goal_reach_dist_threshold = goal_reach_dist_threshold
        self._goal_reach_angle_threshold = goal_reach_angle_threshold
        self._occupancy_map_obstacle_padding_dist = occupancy_map_obstacle_padding_dist

        self._occupancy_map = None
        self._occupancy_map_resolution = None
        self._occupancy_map_origin = None
        self._x_min = None
        self._y_min = None
        self._x_max = None
        self._y_max = None

        self._path = None

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
        self._occupancy_map = occupancy_map
        self._occupancy_map_resolution = resolution
        self._occupancy_map_origin = origin
        self._x_min = origin[0]
        self._y_min = origin[1]
        self._x_max = origin[0] + resolution * occupancy_map.shape[1]
        self._y_max = origin[1] + resolution * occupancy_map.shape[0]

        self._padded_occupancy_map = GridmapUtils.get_padded_occupancy_map(
            occupancy_map, self._occupancy_map_obstacle_padding_dist, resolution
        )

    def plan_global_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        render=False,
        save_to_file=False,
        plot_file_name: str = "path.png",
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
            PlotUtils.plot_global_path(
                self._occupancy_map,
                self._occupancy_map_resolution,
                self._occupancy_map_origin,
                start_pose,
                goal_pose,
                path,
            )

        if save_to_file:
            PlotUtils.save_global_path_to_file(
                self._occupancy_map,
                self._occupancy_map_resolution,
                self._occupancy_map_origin,
                start_pose,
                goal_pose,
                path,
                plot_file_name,
            )

        success = path is not None
        if path is None:
            path = []

        return success, path, num_nodes_sampled

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
        del start_pose, goal_pose
        # this method should be implemented in the child class
        raise NotImplementedError

    def _check_collision(
        self,
        path: List[Tuple[float, float, float]],
    ) -> bool:
        """
        Return True if there is a collision between the near node and the new node. Otherwise, return False.
        """
        return GridmapUtils.check_collision_for_path(
            self._padded_occupancy_map,
            self._occupancy_map_resolution,
            self._occupancy_map_origin,
            path,
        )

    def _check_goal_reached(
        self,
        new_node_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> bool:
        """
        Return True if the goal is reached. Otherwise, return False.
        """
        return GeometryUtils.check_if_dist_and_angle_diff_are_below_threshold(
            new_node_pose,
            goal_pose,
            self._goal_reach_dist_threshold,
            self._goal_reach_angle_threshold,
        )

    def _sample_random_pos(self) -> Tuple[float, float, float]:
        return (
            np.random.uniform(self._x_min, self._x_max),
            np.random.uniform(self._y_min, self._y_max),
        )
