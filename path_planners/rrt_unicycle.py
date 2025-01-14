from typing import Tuple, Optional, List
import time

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.kinematic_utils as KinematicUtils
import path_planners.rrt_utils.rrt_utils as RrtUtils
import path_planners.utils.debug_utils as DebugUtils

from path_planners.path_planner import PathPlanner
from path_planners.rrt_utils.rrt_trees import RrtTree


class RrtUnicyclePlanner(PathPlanner):
    """
    Plan a path from start to goal using rrt star algorithm for wheeled vehicle.
    - Start pose orientation (yaw) is considered but goal pose orientation is ignored.
    - The vehicle have minimum turning ladius.
    - Angular acceleration limits are not considered.
    """

    def __init__(
        self,
        terminate_on_goal_reached: bool = True,
        goal_reach_dist_threshold: float = 0.5,
        goal_reach_angle_threshold: float = 0.1 * np.pi,
        occupancy_map_obstacle_padding_dist: float = 0.5,
        interpolate_path: bool = False,
        d_s: float = 0.25,
        print_log: bool = False,
        goal_sample_rate: float = 0.2,
        max_iter: int = 10000,
        max_drive_dist: float = 0.5,
        linear_velocity: float = 1.0,
        max_angular_velocity: float = 1.0,
        render_tree_during_planning: bool = False,
    ):
        super().__init__(
            terminate_on_goal_reached,
            goal_reach_dist_threshold,
            goal_reach_angle_threshold,
            occupancy_map_obstacle_padding_dist,
            interpolate_path,
            d_s,
            print_log,
        )
        self._goal_sample_rate = goal_sample_rate
        self._max_iter = max_iter
        self._max_drive_dist = max_drive_dist
        self._linear_velocity = linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._render_tree_during_planning = render_tree_during_planning

        self._update_min_turning_radius()
        self._tree = RrtTree()

    def _update_min_turning_radius(self):
        if self._max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity should be positive!")
        if self._linear_velocity <= 0.0:
            raise ValueError("linear_velocity should be positive!")
        self._min_turning_radius = self._linear_velocity / self._max_angular_velocity

    def set_occupancy_map(self, occupancy_map, resolution, origin):
        return super().set_occupancy_map(occupancy_map, resolution, origin)

    def plan_global_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        render=False,
        save_plot_to_file=False,
        plot_file_name: str = "rrt_unicycle_path.png",
    ) -> Tuple[bool, List[Tuple[float, float, float]], int]:
        return super().plan_global_path(
            start_pose, goal_pose, render, save_plot_to_file, plot_file_name
        )

    def _plan_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], int]:
        """
        Plan a path from start to goal using rrt star algorithm for wheeled vehicle.

        Until goal is reached or max_iter is reached:
        1. Sample a random point (x, y).
        2. Find the nearest node (by position) in the tree to the random point.
        3. Drive from the nearest node towards the random point and get new node position.
        4. Check collision and reachability.
            - If collision or unreachable, continue.
        5. Add the new node to the tree with the parent node.
            - The heading of the new node is assigned by the path from the parent node chosen for that node.
        """
        path_planning_start_time = time.time()
        # Initialize the tree with the start node
        self._tree.reset_tree()
        self._tree.add_root(start_pose)

        path_found = False

        edge_paths = []

        for sample_iter in range(self._max_iter):
            if self._render_tree_during_planning and sample_iter % 1000 == 0:
                print("sample_iter: ", sample_iter)
                self._plot_tree_with_edge_paths(start_pose, goal_pose, edge_paths)

            # 1. Sample a random point
            random_pos = self._sample_position(goal_pose)

            # 2. Find the nearest node (by position) in the tree to the random point
            nearest_node = self._tree.find_nearest_node(random_pos)
            nearest_node_pos = nearest_node.get_pos()
            nearest_node_pose = nearest_node.get_pose()

            # 3. Drive from the nearest node towards the random point and get new node position
            new_node_pos = RrtUtils.drive_pos(
                nearest_node_pos, random_pos, self._max_drive_dist
            )

            # 4. Check collision and reachability
            # - Get arc path from the nearest node to the new node
            if not KinematicUtils.check_unicycle_reachability(
                nearest_node_pose,
                new_node_pos,
                self._linear_velocity,
                self._max_angular_velocity,
            ):
                continue

            path_to_new_node = KinematicUtils.get_unicycle_path(
                nearest_node_pose,
                new_node_pos,
                d_s=self._occupancy_map_resolution,  # = collision check resolution
            )

            if self._check_collision(path_to_new_node):
                continue

            edge_paths.append(path_to_new_node)

            new_node_yaw = KinematicUtils.calculate_final_yaw_of_arc_path(
                nearest_node_pose, new_node_pos
            )
            # print("\tnew_node_yaw: ", new_node_yaw)

            new_node_idx = self._tree.add_node(
                (new_node_pos[0], new_node_pos[1], new_node_yaw),
                nearest_node.get_idx(),
            )

            # Check if the goal is reached
            if self._check_goal_reachable(new_node_pos, goal_pose):
                if not path_found:
                    if self._print_log:
                        print("Goal is reached !!! sample_iter: ", sample_iter)
                    path_found = True
                if not self._tree.check_if_pose_in_tree(goal_pose):
                    self._tree.add_node(goal_pose, new_node_idx)
                if self._terminate_on_goal_reached:
                    break

        path_planning_end_time = time.time()
        if self._print_log:
            print(
                f"Path planning time: {path_planning_end_time - path_planning_start_time:.2f} sec"
            )
        self._plot_tree_with_edge_paths(start_pose, goal_pose, edge_paths)

        if path_found:
            if self._print_log:
                print("Path found")
            self._path = self._tree.get_path_from_tree(goal_pose)
            # interpolated_unicycle_path = KinematicUtils.interpolate_path_using_arc(
            #     self._path, d_s=self._occupancy_map_resolution
            # )
            # self._path = interpolated_unicycle_path
            return self._path, sample_iter
        else:
            if self._print_log:
                print("Max iteration reached !!!")
            return None, sample_iter

    def _check_goal_reachable(
        self,
        new_node_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> bool:
        """
        Check if the goal is reachable from the new node pose.
        """
        # TODO: Implement this method
        return GeometryUtils.check_if_dist_is_below_threshold(
            new_node_pose[:2], goal_pose[:2], self._goal_reach_dist_threshold
        )

    def _sample_position(
        self, goal_pose: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        if np.random.rand() < self._goal_sample_rate:
            return (goal_pose[0], goal_pose[1])
        else:
            return self._sample_random_pos()

    def _check_collision(self, path: List[Tuple[float, float, float]]):
        return super()._check_collision(path)

    def _check_goal_reached(
        self, new_node_pos: Tuple[float, float], goal_pos: Tuple[float, float]
    ) -> bool:
        return super()._check_goal_reached(new_node_pos, goal_pos)

    def _sample_random_pos(self):
        return super()._sample_random_pos()

    def _interpolate_poses_on_path(
        self, path: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        return KinematicUtils.interpolate_path_using_arc(path, self._d_s)

    def _plot_tree(
        self,
        start_pose=Tuple[float, float, float],
        goal_pose=Tuple[float, float, float],
    ):
        nodes = self._tree.get_nodes()

        edge_paths = []
        for node in nodes:
            parent_idx = node.get_parent()
            if parent_idx is not None:
                parent_node = nodes[parent_idx]
                unicycle_edge_path = KinematicUtils.get_unicycle_path(
                    parent_node.get_pose(),
                    node.get_pos(),
                    d_s=self._occupancy_map_resolution,
                )
                edge_paths.append(unicycle_edge_path)

        RrtUtils.plot_tree_with_edge_paths(
            nodes,
            self._occupancy_map,
            self._occupancy_map_resolution,
            self._occupancy_map_origin,
            start_pose,
            goal_pose,
            edge_paths,
        )

    def _plot_tree_with_edge_paths(
        self,
        start_pose=Tuple[float, float, float],
        goal_pose=Tuple[float, float, float],
        edge_paths: List[List[Tuple[float, float, float]]] = [],
    ):
        nodes = self._tree.get_nodes()
        RrtUtils.plot_tree_with_edge_paths(
            nodes,
            self._occupancy_map,
            self._occupancy_map_resolution,
            self._occupancy_map_origin,
            start_pose,
            goal_pose,
            edge_paths,
        )
