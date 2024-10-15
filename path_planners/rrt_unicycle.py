from typing import Tuple, Optional, List

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
        goal_reach_dist_threshold: float = 0.5,
        goal_reach_angle_threshold: float = 0.1 * np.pi,
        goal_sample_rate: float = 0.2,
        max_iter: int = 10000,
        max_drive_dist: float = 0.5,
        linear_velocity: float = 1.0,
        max_angular_velocity: float = 1.0,
    ):
        super().__init__(goal_reach_dist_threshold, goal_reach_angle_threshold)
        self._goal_sample_rate = goal_sample_rate
        self._max_iter = max_iter
        self._max_drive_dist = max_drive_dist
        self._linear_velocity = linear_velocity
        self._max_angular_velocity = max_angular_velocity

        self._tree = RrtTree()

    def set_occupancy_map(self, occupancy_map, resolution, origin):
        return super().set_occupancy_map(occupancy_map, resolution, origin)

    def plan_global_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        render=False,
    ) -> Tuple[bool, List[Tuple[int, int]], int]:
        return super().plan_global_path(start_pose, goal_pose, render)

    def _plan_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> Tuple[Optional[List[Tuple[int, int]]], int]:
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

        # Initialize the tree with the start node
        self._tree.reset_tree()
        self._tree.add_root(start_pose)

        path_found = False

        edge_paths = []

        for sample_iter in range(self._max_iter):
            if sample_iter % 1000 == 0:
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
            print(f"new_node_pos: {new_node_pos[0]:.2f}, {new_node_pos[1]:.2f}")
            print(
                f"\tnearest_node_pose: {nearest_node_pose[0]:.2f}, {nearest_node_pose[1]:.2f}, {nearest_node_pose[2]:.2f}"
            )
            # 4. Check collision and reachability
            # - Get arc path from the nearest node to the new node
            if not KinematicUtils.check_unicycle_reachability(
                nearest_node_pose,
                new_node_pos,
                self._linear_velocity,
                self._max_angular_velocity,
            ):
                print("\tUnreachable")
                continue

            path_to_new_node = KinematicUtils.get_unicycle_path(
                nearest_node_pose,
                new_node_pos,
                d_s=self._occupancy_map_resolution,  # = collision checkresolution
            )
            print("\t----------------")
            DebugUtils.print_path(path=path_to_new_node)
            print("\t----------------")

            if self._check_collision(path_to_new_node):
                continue
            edge_paths.append(path_to_new_node)

            new_node_yaw = KinematicUtils.calculate_unicycle_final_yaw(
                nearest_node_pose, new_node_pos
            )
            print("\tnew_node_yaw: ", new_node_yaw)

            new_node_idx = self._tree.add_node(
                (new_node_pos[0], new_node_pos[1], new_node_yaw),
                nearest_node.get_idx(),
            )

            # Check if the goal is reached
            if self._check_goal_reachable(new_node_pos, goal_pose):
                path_found = True
                if not self._tree.check_if_pose_in_tree(goal_pose):
                    self._tree.add_node(goal_pose, new_node_idx)
                break

        self._plot_tree_with_edge_paths(start_pose, goal_pose, edge_paths)

        if path_found:
            print("Goal is reached !!!")
            self.path = self._tree.get_path_from_tree(goal_pose)
            return self.path, sample_iter
        else:
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

    def _check_collision(self, path):
        return super()._check_collision(path)

    def _check_goal_reached(
        self, new_node_pos: Tuple[float, float], goal_pos: Tuple[float, float]
    ) -> bool:
        return super()._check_goal_reached(new_node_pos, goal_pos)

    def _sample_random_pos(self):
        return super()._sample_random_pos()

    def _plot_tree(
        self,
        start_pose=Tuple[float, float, float],
        goal_pose=Tuple[float, float, float],
    ):
        nodes = self._tree.get_nodes()
        RrtUtils.plot_tree(
            nodes,
            self._occupancy_map,
            self._occupancy_map_resolution,
            self._occupancy_map_origin,
            start_pose,
            goal_pose,
        )
        print("plot tree")

    def _plot_tree_with_edge_paths(
        self,
        start_pose=Tuple[float, float, float],
        goal_pose=Tuple[float, float, float],
        edge_paths: List[List[Tuple[float, float, float]]] = [],
    ):
        nodes = self._tree.get_nodes()
        RrtUtils._plot_tree_with_edge_paths(
            nodes,
            self._occupancy_map,
            self._occupancy_map_resolution,
            self._occupancy_map_origin,
            start_pose,
            goal_pose,
            edge_paths,
        )
        print("plot tree with paths")
