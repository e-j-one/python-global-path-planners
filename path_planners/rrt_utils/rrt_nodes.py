from typing import Tuple


class RrtNode:
    def __init__(self, idx: int, pose: Tuple[float, float, float], parent: int) -> None:
        """
        Parameters:
        - idx: index of the node
        - pose: (x, y, yaw) of the node
        - parent: index of the parent node
        """
        self.idx = idx
        self.pose = pose
        self.parent = parent
        self.children = []

    def add_child(self, child_idx: int) -> None:
        self.children.append(child_idx)

    def get_idx(self) -> int:
        return self.idx

    def get_pos(self) -> Tuple[float, float]:
        """
        Return the (x, y) position of the node
        """
        return (self.pose[0], self.pose[1])

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Return the (x, y, yaw) pose of the node
        """
        return self.pose

    def get_parent(self) -> int:
        "Return the index of the parent node"
        return self.parent

    def get_children(self) -> int:
        return self.children.copy()


class RrtStarNode(RrtNode):
    def __init__(
        self,
        idx: int,
        pose: Tuple[float, float, float],
        parent: int,
        cost: float,
        cost_from_parent: float,
    ) -> None:
        """
        Parameters:
        - idx: index of the node
        - pose: (x, y, yaw) of the node
        - parent: index of the parent node
        - cost: cost of the node
        - cost_from_parent: cost from the parent node to this node (used to propagate cost)
        """
        self.idx = idx
        self.pose = pose
        self.parent = parent
        self.cost = cost
        self.cost_from_parent = cost_from_parent
        self.children = []

    def update_parent_and_cost(
        self, new_parent: int, new_cost: float, new_cost_from_parent: float
    ) -> None:
        """
        Parameters:
        - new_parent: new parent index
        - new_cost: new cost of the node
        - new_cost_from_parent: new cost from the parent node to this node
        """
        self.parent = new_parent
        self.cost = new_cost
        self.cost_from_parent = new_cost_from_parent

    def update_cost(self, updated_cost: float) -> None:
        self.cost = updated_cost

    def remove_child(self, child_idx: int) -> None:
        self.children.remove(child_idx)

    def get_cost_from_parent(self) -> float:
        return self.cost_from_parent

    def get_cost(self) -> float:
        return self.cost
