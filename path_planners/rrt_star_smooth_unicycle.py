from typing import Tuple, Optional, List

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.kinematic_utils as KinematicUtils
import path_planners.rrt_utils.rrt_utils as RrtUtils
import path_planners.utils.debug_utils as DebugUtils

from path_planners.rrt_unicycle import RrtUnicyclePlanner
from path_planners.rrt_utils.rrt_trees import RrtStarTree


class RrtStarSmoothUnicyclePlanner(RrtUnicyclePlanner):
    """
    Plan a path from start to goal using rrt star algorithm for wheeled vehicle.
    - Start pose orientation (yaw) is considered but goal pose orientation is ignored.
    - The vehicle can only drive at a constant velocity.
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
        render_tree_during_planning: bool = False,
        near_node_dist_threshold: float = 0.5,
    ):
        super().__init__(
            goal_reach_dist_threshold,
            goal_reach_angle_threshold,
            goal_sample_rate,
            max_iter,
            max_drive_dist,
            linear_velocity,
            max_angular_velocity,
            render_tree_during_planning,
        )

        self._tree = RrtStarTree(near_node_dist_threshold=near_node_dist_threshold)

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
        1. Sample a random point
        2. Find the nearest node (by position) in the tree to the random point
        3. Drive from the nearest node towards the random point and get new node position
        4. Find the nodes withing near_node_threshold distance from the new node position
        5. Choose the parent node with minimum cost from the near nodes
            - For each near node:
                - Get v (const), w from the near node to the new node
                - Calculate the cost
                - If cost is smaller than minimum cost,
                  check collision between near node and new node.
                  If no collision, update the minimum cost and parent node
        6. Add the new node to the tree with the parent node
        7. Rewire the tree
            - For each near node:
                - Get path and pose_stopover from the near node to the new node which connects the nodes with two smooth arcs.
                - If no collision and the cost is smaller than the previous cost, update the parent node.
        """
        # Initialize the tree with the start node
        self._tree.reset_tree()
        self._tree.add_root(start_pose)

        path_found = False

        for sample_iter in range(self._max_iter):
            if self._render_tree_during_planning and sample_iter % 1000 == 0:
                print("sample_iter: ", sample_iter)

            # 1. Sample a random point
            random_pos = self._sample_position(goal_pose)

            # 2. Find the nearest node (by position) in the tree to the random point
            nearest_node = self._tree.find_nearest_node(random_pos)
            nearest_node_pos = nearest_node.get_pos()

            # 3. Drive from the nearest node towards the random point and get new node position
            new_node_pos = RrtUtils.drive_pos(
                nearest_node_pos, random_pos, self._max_drive_dist
            )

            # 4. Find the nodes withing near_node_threshold distance from the new node position
            near_nodes_idx = self._tree.find_near_nodes(new_node_pos)

            # 5. Choose the parent node with minimum cost from the near nodes
            (
                new_node_poses_by_collision_free_near_nodes,
                collision_free_near_nodes_idx,
                cost_to_collision_free_near_nodes,
            ) = self._get_collision_free_near_nodes(new_node_pos, near_nodes_idx)

            if len(collision_free_near_nodes_idx) == 0:
                continue

            # 6. Add the new node to the tree with the parent node
            new_node_idx = self._tree.find_optimal_parent_and_add_node_to_tree(
                new_node_poses_by_collision_free_near_nodes,
                collision_free_near_nodes_idx,
                cost_to_collision_free_near_nodes,
            )

            # 7. Rewire the tree
            self._rewire_tree(new_node_idx, near_nodes_idx)

            # Check if the goal is reached
            if self._check_goal_reachable(new_node_pos, goal_pose):
                path_found = True
                if not self._tree.check_if_pose_in_tree(goal_pose):
                    new_node_cost = self._tree.get_cost_of_node_idx(new_node_idx)
                    cost_to_goal = self._get_cost_of_path([new_node_pos, goal_pose])
                    self._tree.add_node(
                        goal_pose,
                        new_node_idx,
                        new_node_cost + cost_to_goal,
                        cost_to_goal,
                    )
                # break

        if path_found:
            self._plot_tree(start_pose, goal_pose)
            print("Goal is reached !!!")
            self._path = self._tree.get_path_from_tree(goal_pose)
            return self._path, sample_iter
        else:
            print("Max iteration reached !!!")
            return None, sample_iter

    def _check_goal_reachable(
        self,
        new_node_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
    ) -> bool:
        return super()._check_goal_reachable(new_node_pose, goal_pose)

    def _sample_position(
        self, goal_pose: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        return super()._sample_position(goal_pose)

    def _get_collision_free_near_nodes(
        self, new_node_pos: Tuple[float, float], near_nodes_idx: List[int]
    ) -> Tuple[List[Tuple[float, float, float]], List[int], List[float]]:
        """
        Returns:
        - new_node_poses: List of new node poses derived from each near nodes
        - collision_free_near_nodes_idx: List of near nodes which can reach the new node without collision
        - cost_to_collision_free_near_nodes: List of costs to the each path from near node to new node
        """
        collision_free_near_nodes_idx = []
        cost_to_collision_free_near_nodes = []
        new_node_poses_by_collision_free_near_nodes = []

        for near_node_idx in near_nodes_idx:
            near_node_pose = self._tree.get_pose_of_node(near_node_idx)
            if not KinematicUtils.check_unicycle_reachability(
                near_node_pose,
                new_node_pos,
                self._linear_velocity,
                self._max_angular_velocity,
            ):
                continue

            path_to_new_node = KinematicUtils.get_unicycle_path(
                near_node_pose,
                new_node_pos,
                d_s=self._occupancy_map_resolution,
            )

            if self._check_collision(path_to_new_node):
                continue

            new_node_yaw = KinematicUtils.calculate_final_yaw_of_arc_path(
                near_node_pose, new_node_pos
            )

            cost_to_near_node = self._get_cost_of_path(path_to_new_node)

            collision_free_near_nodes_idx.append(near_node_idx)
            cost_to_collision_free_near_nodes.append(cost_to_near_node)
            new_node_poses_by_collision_free_near_nodes.append(
                (new_node_pos[0], new_node_pos[1], new_node_yaw)
            )

        return (
            new_node_poses_by_collision_free_near_nodes,
            collision_free_near_nodes_idx,
            cost_to_collision_free_near_nodes,
        )

    def _check_collision(self, path: List[Tuple[float, float, float]]):
        return super()._check_collision(path)

    def _get_cost_of_path(self, path: List[Tuple[float, float, float]]):
        return GeometryUtils.get_path_length(path)

    def _rewire_tree(self, new_node_idx, near_nodes_idx):
        # new_node_pose = self._tree.get_pose_of_node(new_node_idx)
        # for near_node_idx in near_nodes_idx:
        #     near_node_pose = self._tree.get_pose_of_node(near_node_idx)

        #     path_1, path_2, pose_stopover = (
        #         KinematicUtils.get_pose_to_connect_poses_by_two_arcs(
        #             new_node_pose, near_node_pose
        #         )
        #     )

        #     if self._check_collision(path_to_new_node):
        #         continue

        #     new_node_cost = self._tree.get_cost_of_node_idx(new_node_idx)
        #     cost_to_new_node = self._get_cost_of_path(path_to_new_node)

        #     if new_node_cost + cost_to_new_node < self._tree.get_cost_of_node_idx(
        #         near_node_idx
        #     ):
        #         self._tree.update_parent_of_node(near_node_idx, new_node_idx)
        pass
