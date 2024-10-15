from typing import Tuple


class RrtNode:
    def __init__(self, idx: int, pose: Tuple[float, float, float], parent: int) -> None:
        """
        Parameters:
        - idx: index of the node
        - pose: (x, y, yaw) of the node
        - parent: index of the parent node
        """
        self._idx = idx
        self._pose = pose
        self._parent = parent
        self._children = []

    def add_child(self, child_idx: int) -> None:
        self._children.append(child_idx)

    def get_idx(self) -> int:
        return self._idx

    def get_pos(self) -> Tuple[float, float]:
        """
        Return the (x, y) position of the node
        """
        return (self._pose[0], self._pose[1])

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Return the (x, y, yaw) pose of the node
        """
        return self._pose

    def get_parent(self) -> int:
        "Return the index of the parent node"
        return self._parent

    def get_children(self) -> int:
        return self._children.copy()


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
        self._idx = idx
        self._pose = pose
        self._parent = parent
        self._cost = cost
        self._cost_from_parent = cost_from_parent
        self._children = []

    def update_parent_and_cost(
        self, new_parent: int, new_cost: float, new_cost_from_parent: float
    ) -> None:
        """
        Parameters:
        - new_parent: new parent index
        - new_cost: new cost of the node
        - new_cost_from_parent: new cost from the parent node to this node
        """
        self._parent = new_parent
        self._cost = new_cost
        self._cost_from_parent = new_cost_from_parent

    def update_cost(self, updated_cost: float) -> None:
        self._cost = updated_cost

    def remove_child(self, child_idx: int) -> None:
        self._children.remove(child_idx)

    def get_cost_from_parent(self) -> float:
        return self._cost_from_parent

    def get_cost(self) -> float:
        return self._cost
