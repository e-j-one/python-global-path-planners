from typing import List, Tuple, Optional

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.gridmap_utils as GridmapUtils
import path_planners.utils.plot_utils as PlotUtils
import path_planners.utils.math_utils as MathUtils


class PathPlanner:
    def __init__(
        self,
        terminate_on_goal_reached: bool = True,
        goal_reach_dist_threshold: float = 0.5,
        goal_reach_angle_threshold: float = 0.1 * np.pi,
        occupancy_map_obstacle_padding_dist: float = 0.5,
        interpolate_path: bool = False,
        d_s: float = 0.25,
    ):
        self._terminate_on_goal_reached = terminate_on_goal_reached
        self._goal_reach_dist_threshold = goal_reach_dist_threshold
        self._goal_reach_angle_threshold = goal_reach_angle_threshold
        self._occupancy_map_obstacle_padding_dist = occupancy_map_obstacle_padding_dist
        self._interpolate_path = interpolate_path
        self._d_s = d_s

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
        save_plot_to_file=False,
        plot_file_name: str = "path.png",
    ) -> Tuple[bool, List[Tuple[float, float, float]], int]:
        """
        Plan a global path from start to goal pose

        Parameters
        ----------
        start_pose : Tuple[float, float, float]
            start pose (x, y, yaw)
        goal_pose : Tuple[float, float, float]
            goal pose (x, y, yaw)
        render : bool, optional
            Whether to render the path, by default False
        save_plot_to_file : bool, optional
            Whether to save the path to a file, by default False
        plot_file_name : str, optional
            The name of the file to save the path to, by default "path.png"

        Returns
        -------
        success : bool
        path : List[Tuple[float, float, float]]
        number of nodes sampled : int
        """
        path, num_nodes_sampled = self._plan_path(start_pose, goal_pose)

        if path is not None:
            if self._interpolate_path:
                path = self._interpolate_poses_on_path(path)
            if self._check_collision(path):
                print(f"Path is in collision! path: {path}")
                path = None

        success = path is not None
        if path is None:
            path = []

        if render:
            PlotUtils.plot_global_path(
                self._occupancy_map,
                self._occupancy_map_resolution,
                self._occupancy_map_origin,
                start_pose,
                goal_pose,
                path,
            )

        if save_plot_to_file:
            PlotUtils.save_global_path_to_file(
                self._occupancy_map,
                self._occupancy_map_resolution,
                self._occupancy_map_origin,
                start_pose,
                goal_pose,
                path,
                plot_file_name,
            )

        return success, path, num_nodes_sampled

    def sample_start_goal_poses(
        self, min_dist: float = 1.0
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Sample start and goal poses.
        Poses are sampled in collision free region considering the padding distance.
        Yaw is sampled from (-pi, pi].
        """
        if min_dist >= np.linalg.norm(
            [self._x_max - self._x_min, self._y_max - self._y_min]
        ):
            raise ValueError(
                "The minimum distance between start and goal poses should be less than the map size!"
            )

        while True:
            start_pos = self._sample_random_collision_free_pos()
            goal_pos = self._sample_random_collision_free_pos()
            if MathUtils.calculate_dist(start_pos[:2], goal_pos[:2]) >= min_dist:
                break
        start_pose = (*start_pos, -np.random.uniform(-np.pi, np.pi))
        goal_pose = (*goal_pos, -np.random.uniform(-np.pi, np.pi))
        return start_pose, goal_pose

    def _plan_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], int]:
        """
        Plan a path from start to goal pose

        Parameters
        ----------
        start_pose : Tuple[float, float, float]
            start pose (x, y, yaw)
        goal_pose : Tuple[float, float, float]
            goal pose (x, y, yaw)

        Returns
        -------
        path : Optional[List[Tuple[float, float, float]]]
            The planned path. None if the path is not found.
        num_nodes_sampled : int
            The number of nodes sampled during the planning
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

    def _sample_random_collision_free_pos(self) -> Tuple[float, float, float]:
        """
        Sample a random position that is obstacle free.
        """
        while True:
            pos = self._sample_random_pos()
            if GridmapUtils.check_if_pos_is_free(
                self._padded_occupancy_map,
                self._occupancy_map_resolution,
                self._occupancy_map_origin,
                pos,
            ):
                return pos

    def _interpolate_poses_on_path(
        self, path: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Interpolate the path by adding or removing points in the path.
        - The distance between two consecutive points should be self._d_s.
        """
        raise NotImplementedError
