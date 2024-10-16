from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from path_planners.rrt_utils.rrt_nodes import RrtNode
from path_planners.utils.plot_utils import add_occupancy_grid_to_plot


def drive_pos(
    nearest_node_pos: Tuple[float, float],
    random_pos: Tuple[float, float],
    max_drive_dist: float,
) -> Tuple[float, float]:
    dist = np.linalg.norm(np.array(nearest_node_pos) - np.array(random_pos))
    if dist < max_drive_dist:
        return random_pos
    else:
        direction = np.array(random_pos) - np.array(nearest_node_pos)
        direction = direction / np.linalg.norm(direction)
        new_node_pos = (
            nearest_node_pos[0] + max_drive_dist * direction[0],
            nearest_node_pos[1] + max_drive_dist * direction[1],
        )
        return new_node_pos


def plot_tree(
    nodes: List[RrtNode],
    occupancy_map: np.ndarray,
    occupancy_map_resolution: float,
    occupancy_map_origin: Tuple[float, float, float],
    start_pose: Tuple[float, float, float],
    goal_pose: Tuple[float, float, float],
):
    # Extract node positions and draw edges
    node_poses = [node.get_pose() for node in nodes]
    edges = []
    for node in nodes:
        parent_idx = node.get_parent()
        if parent_idx is not None:
            parent_node = nodes[parent_idx]
            edges.append((node.get_pose(), parent_node.get_pose()))

    # Set up the figure
    plt.figure()

    add_occupancy_grid_to_plot(
        occupancy_map, occupancy_map_resolution, occupancy_map_origin
    )
    # Plot edges
    for edge in edges:
        (x1, y1, _), (x2, y2, _) = edge
        plt.plot([x1, x2], [y1, y2], "b-", linewidth=1)

    # Plot nodes
    for x, y, _ in node_poses:
        plt.plot(x, y, "bo", markersize=2)

    # Plot start and goal poses
    plt.plot(start_pose[0], start_pose[1], "go", markersize=8, label="Start")
    plt.plot(goal_pose[0], goal_pose[1], "ro", markersize=8, label="Goal")

    plt.title("RRT Tree on Occupancy Map")
    plt.legend()

    plt.grid(False)
    plt.show()


def plot_tree_with_edge_paths(
    nodes: List[RrtNode],
    occupancy_map: np.ndarray,
    occupancy_map_resolution: float,
    occupancy_map_origin: Tuple[float, float, float],
    start_pose: Tuple[float, float, float],
    goal_pose: Tuple[float, float, float],
    edge_paths: List[List[Tuple[float, float, float]]],
):
    # Extract node positions and draw edges
    node_poses = [node.get_pose() for node in nodes]

    # Set up the figure
    plt.figure()

    add_occupancy_grid_to_plot(
        occupancy_map, occupancy_map_resolution, occupancy_map_origin
    )
    # Plot edges
    for edge_path in edge_paths:
        for i in range(len(edge_path) - 1):
            (x1, y1, _), (x2, y2, _) = edge_path[i], edge_path[i + 1]
            plt.plot([x1, x2], [y1, y2], "b-", linewidth=1)

    # Plot nodes
    for x, y, _ in node_poses:
        plt.plot(x, y, "bo", markersize=2)

    # Plot start and goal poses
    plt.plot(start_pose[0], start_pose[1], "go", markersize=8, label="Start")
    plt.plot(goal_pose[0], goal_pose[1], "ro", markersize=8, label="Goal")

    plt.title("RRT Tree on Occupancy Map")
    plt.legend()

    plt.grid(False)
    plt.show()
