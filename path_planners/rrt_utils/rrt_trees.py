from typing import List, Tuple, Optional

from scipy.spatial import KDTree
import numpy as np

from path_planners.rrt_utils.rrt_nodes import RrtNode, RrtStarNode


class RrtTree:
    def __init__(self) -> None:
        self._nodes: List[RrtNode] = []
        self._pose_to_idx_map = {}
        self._kd_pos_tree: KDTree = None

    def reset_tree(self) -> None:
        self._nodes: List[RrtNode] = []
        self._pose_to_idx_map = {}

        self._kd_pos_tree: KDTree = None

    def add_node(self, node_pose: Tuple[float, float, float], parent_idx: int) -> int:
        idx = len(self._nodes)
        node = RrtNode(idx, node_pose, parent_idx)
        self._nodes.append(node)
        if parent_idx is not None:
            self._nodes[parent_idx].add_child(idx)
        self._pose_to_idx_map[node_pose] = idx
        self._kd_pos_tree = KDTree([node.get_pos() for node in self._nodes])
        return idx

    def add_root(self, root_pose: Tuple[float, float, float]) -> int:
        if len(self._nodes) > 0:
            raise ValueError("Root node already exists!")
        self.add_node(root_pose, None)

    def get_pose_of_node(self, idx: int) -> Tuple[float, float, float]:
        return self._nodes[idx].get_pose()

    def find_nearest_node(self, pos: Tuple[float, float]) -> RrtNode:
        distances, nearest_node_idx = self._kd_pos_tree.query(np.array([pos]))
        return self._nodes[nearest_node_idx[0]]

    def check_if_pose_in_tree(self, pose: Tuple[float, float]) -> bool:
        return pose in self._pose_to_idx_map

    def get_idx_from_pos(self, pose: Tuple[float, float, float]) -> Optional[int]:
        if pose not in self._pose_to_idx_map:
            return -1
        return self._pose_to_idx_map[pose]

    def get_num_nodes(self) -> int:
        return len(self._nodes)

    def get_path_from_tree(
        self, goal_pose: Tuple[float, float, float]
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Get the path from the root to the goal pose. Return None if the goal pose is not in the tree
        """
        if goal_pose not in self._pose_to_idx_map:
            return None
        path = [goal_pose]
        goal_idx = self._pose_to_idx_map[goal_pose]
        node_on_path_idx = self._nodes[goal_idx].get_parent()
        while node_on_path_idx is not None:
            path.append(self._nodes[node_on_path_idx].get_pose())
            node_on_path_idx = self._nodes[node_on_path_idx].get_parent()

        return path[::-1]

    def get_nodes(self) -> List[RrtNode]:
        return self._nodes

    def get_path_idx_from_tree(
        self, goal_pose: Tuple[float, float, float]
    ) -> Optional[List[int]]:
        if goal_pose not in self._pose_to_idx_map:
            return None
        goal_idx = self._pose_to_idx_map[goal_pose]
        path = [goal_idx]
        node_on_path_idx = self._nodes[goal_idx].get_parent()
        while node_on_path_idx is not None:
            path.append(node_on_path_idx)
            node_on_path_idx = self._nodes[node_on_path_idx].get_parent()

        return path[::-1]


class RrtStarTree(RrtTree):
    def __init__(self, near_node_dist_trheshold: float):
        self.near_node_dist_trheshold = near_node_dist_trheshold

        self._nodes: List[RrtStarNode] = []
        self._pose_to_idx_map = {}

        self._kd_pos_tree: KDTree = None

    def reset_tree(self):
        self._nodes: List[RrtStarNode] = []
        self._pose_to_idx_map = {}

        self._kd_pos_tree: KDTree = None

    def add_node(
        self,
        node_pose: Tuple[float, float, float],
        parent_idx: int,
        cost: float,
        cost_from_parent: float,
    ) -> int:
        idx = len(self._nodes)
        node = RrtStarNode(idx, node_pose, parent_idx, cost, cost_from_parent)
        self._nodes.append(node)
        if parent_idx is not None:
            self._nodes[parent_idx].add_child(idx)
        self._pose_to_idx_map[node_pose] = idx
        self._kd_pos_tree = KDTree([node.get_pos() for node in self._nodes])
        return idx

    def add_root(self, root_pose) -> int:
        if len(self._nodes) > 0:
            raise ValueError("Root node already exists")
        return self.add_node(root_pose, None, 0, 0)

    def propagate_cost_to_children(self, node_idx: int):
        for child_idx in self._nodes[node_idx].children:
            updated_cost = (
                self._nodes[node_idx].get_cost()
                + self._nodes[child_idx].get_cost_from_parent()
            )

            self._nodes[child_idx].update_cost(updated_cost)
            self.propagate_cost_to_children(child_idx)

    def find_near_nodes(self, new_node_pos: Tuple[float, float]) -> List[int]:
        near_nodes_idx = self._kd_pos_tree.query_ball_point(
            np.array([new_node_pos]), self.near_node_dist_trheshold
        )
        return near_nodes_idx[0]

    def get_cost_of_node_pos(self, pose: Tuple[float, float, float]) -> float:
        if pose not in self._pose_to_idx_map:
            return float("inf")
        return self._nodes[self._pose_to_idx_map[pose]].get_cost()

    def get_cost_of_node_idx(self, idx: int) -> float:
        return self._nodes[idx].get_cost()

    def update_parent_and_cost(
        self,
        node_idx: int,
        new_parent_idx: int,
        new_cost: float,
        new_cost_from_parent: float,
    ):
        original_parent_idx = self._nodes[node_idx].get_parent()
        self._nodes[original_parent_idx].remove_child(node_idx)
        self._nodes[node_idx].update_parent_and_cost(
            new_parent_idx, new_cost, new_cost_from_parent
        )
        self._nodes[new_parent_idx].add_child(node_idx)

    def find_optimal_parent_and_add_node_to_tree(
        self,
        new_node_pose: Tuple[float, float, float],
        collision_free_near_nodes_idx: List[int],
        cost_to_collision_free_near_nodes: List[float],
    ) -> Tuple[int, float]:
        optimal_parent = None
        min_cost = float("inf")
        cost_to_parent = None
        for iter_idx, near_node_idx in enumerate(collision_free_near_nodes_idx):
            cost_via_near_node = (
                self._nodes[near_node_idx].get_cost()
                + cost_to_collision_free_near_nodes[iter_idx]
            )
            if cost_via_near_node < min_cost:
                min_cost = cost_via_near_node
                cost_to_parent = cost_to_collision_free_near_nodes[iter_idx]
                optimal_parent = near_node_idx

        if optimal_parent is None:
            raise ValueError("No optimal parent found")

        return self.add_node(new_node_pose, optimal_parent, min_cost, cost_to_parent)
