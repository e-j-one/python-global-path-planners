from typing import Tuple, Optional, List

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.kinematic_utils as KinematicUtils

from path_planners.path_planner import PathPlanner
from path_planners.rrt_utils.rrt_trees import RrtStarTree


class UnicycleRrtStarPlanner(PathPlanner):
    """
    Plan a path from start to goal using rrt star algorithm for wheeled vehicle.
    - Start pose orientation (yaw) is considered but goal pose orientation is ignored.
    - The vehicle can only drive at a constant velocity.
    - Angular acceleration limits are not considered.
    """

    def __init__(
        self,
        goal_sample_rate: float = 0.05,
        goal_reach_threshold: float = 0.5,
        max_iter: int = 10000,
        max_drive_dist: float = 0.5,
        near_node_threshold: float = 0.5,
        linear_velocity: float = 1.0,
        max_angular_velocity: float = 1.0,
    ):
        super().__init__()
        self.goal_sample_rate = goal_sample_rate
        self.goal_reach_threshold = goal_reach_threshold
        self.max_iter = max_iter
        self.max_drive_dist = max_drive_dist
        self.linear_velocity = linear_velocity
        self.max_angular_velocity = max_angular_velocity

        self.tree = RrtStarTree(near_node_threshold=near_node_threshold)

    def _plan_path(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        print_iter=False,
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
        """
        # Initialize the tree with the start node
        self.tree.reset_tree()
        self.tree.add_root(start_pose)

        path_found = False

        for sample_iter in range(self.max_iter):
            # if sample_iter % 1000 == 0 and print_iter:
            #     print("sample_iter: ", sample_iter)

            # 1. Sample a random point
            random_pos = self._sample_position(goal_pose)

            # 2. Find the nearest node (by position) in the tree to the random point
            nearest_node = self.tree.find_nearest_node(random_pos)
            nearest_node_pos = nearest_node.get_pos()

            # 3. Drive from the nearest node towards the random point and get new node position
            new_node_pos = self._drive(nearest_node_pos, random_pos)

            # 4. Find the nodes withing near_node_threshold distance from the new node position
            near_nodes_idx = self.tree.find_near_nodes(new_node_pos)

            # 5. Choose the parent node with minimum cost from the near nodes
            (
                collision_free_near_nodes_idx,
                cost_to_collision_free_near_nodes,
                yaw_by_near_nodes,
            ) = self._get_collision_free_near_nodes(new_node_pos, near_nodes_idx)

            # 6. Add the new node to the tree with the parent node
            new_node_idx = self.tree.find_optimal_parent_and_add_node_to_tree(
                new_node_pos,
                collision_free_near_nodes_idx,
                cost_to_collision_free_near_nodes,
            )

            # 7. Rewire the tree
            self._rewire_tree(new_node_idx, near_nodes_idx)

            # Check if the goal is reached
            if self._check_goal_reached(new_node_pos, goal_pose):
                path_found = True
                if not self.tree.check_if_pos_in_tree(goal_pose):
                    self.tree.add_node(goal_pose, new_node_idx)
                break

        if path_found:
            self.path = self.tree.get_path_from_tree(goal_pose)
            return self.path, sample_iter
        else:
            return None, sample_iter

    def _sample_position(
        self, goal_pose: Tuple[float, float, float]
    ) -> Tuple[float, float]:
        if np.random.rand() < self.goal_sample_rate:
            return (goal_pose[0], goal_pose[1])
        else:
            return self._sample_random_pos()

    def _sample_random_pos(self) -> Tuple[float, float, float]:
        return (
            np.random.uniform(self.x_min, self.x_max),
            np.random.uniform(self.y_min, self.y_max),
        )

    def _drive(
        self,
        nearest_node_pos: Tuple[float, float],
        random_pos: Tuple[float, float],
    ) -> Tuple[float, float]:
        dist = np.linalg.norm(np.array(nearest_node_pos) - np.array(random_pos))
        if dist < self.max_drive_dist:
            return random_pos
        else:
            direction = np.array(random_pos) - np.array(nearest_node_pos)
            direction = direction / np.linalg.norm(direction)
            new_node_pos = (
                nearest_node_pos[0] + self.max_drive_dist * direction[0],
                nearest_node_pos[1] + self.max_drive_dist * direction[1],
            )
            return new_node_pos

    def _get_collision_free_near_nodes(
        self, new_node_pos: Tuple[float, float], near_nodes_idx: List[int]
    ) -> Tuple[List[int], List[float], List[float]]:
        collision_free_near_nodes_idx = []
        cost_to_collision_free_near_nodes = []
        yaw_by_near_nodes = []

        for near_node_idx in near_nodes_idx:
            near_node_pose = self.tree.get_pose_of_node(near_node_idx)
            if not KinematicUtils.check_unicycle_reachability(
                near_node_pose,
                new_node_pos,
                self.linear_velocity,
                self.max_angular_velocity,
            ):
                continue

            if self._check_collision(near_node_pose, new_node_pos):
                continue

            final_yaw = KinematicUtils.calculate_unicycle_w_yaw(
                near_node_pose, new_node_pos, self.linear_velocity
            )

            cost = self._get_cost(near_node_idx, v, w)

            if cost < float("inf"):
                if not self._check_collision(near_node_pos, new_node_pos):
                    collision_free_near_nodes_idx.append(near_node_idx)
                    cost_to_collision_free_near_nodes.append(cost)
                    yaw_by_near_nodes.append(
                        np.arctan2(
                            new_node_pos[1] - near_node_pos[1],
                            new_node_pos[0] - near_node_pos[0],
                        )
                    )

        return (
            collision_free_near_nodes_idx,
            cost_to_collision_free_near_nodes,
            yaw_by_near_nodes,
        )

    def _check_collision(
        self,
        near_node_pos: Tuple[float, float],
        new_node_pos: Tuple[float, float],
        angular_velocity: float,
    ) -> bool:
        """
        Return True if there is a collision between the near node and the new node. Otherwise, return False.
        """
        return False
