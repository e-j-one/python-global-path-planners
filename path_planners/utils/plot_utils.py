from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def add_occupancy_grid_to_plot(
    occupancy_map: np.ndarray,
    occupancy_map_resolution: float,
    occupancy_map_origin: Tuple[float, float, float],
):
    # Plot the occupancy map
    cmap = plt.cm.binary
    cmap.set_under(color="white")  # Color for free space
    cmap.set_over(color="black")  # Color for occupied space
    cmap.set_bad(color="lightgray")  # Color for unknown space

    # Mask unknown values (-1) so they appear as "bad" (gray) values in the plot
    masked_grid = np.ma.masked_where(occupancy_map == -1, occupancy_map)

    # Visualize the grid with the specified colormap
    plt.imshow(
        masked_grid,
        cmap=cmap,
        origin="lower",
        interpolation="none",
        extent=[
            occupancy_map_origin[0],
            occupancy_map_origin[0] + occupancy_map.shape[1] * occupancy_map_resolution,
            occupancy_map_origin[1],
            occupancy_map_origin[1] + occupancy_map.shape[0] * occupancy_map_resolution,
        ],
        # vmin=0,
        # vmax=100,
    )

    # Add labels and title
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")


def _add_display_color_bar():
    cbar = plt.colorbar()
    cbar.set_label("Occupancy Value")
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(["Free", "Unknown", "Occupied"])


def plot_occupancy_grid(
    occupancy_map: np.ndarray,
    occupancy_map_resolution: float,
    occupancy_map_origin: Tuple[float, float, float],
):
    """
    Visualizes the occupancy grid using matplotlib.

    Parameters:
    - occupancy_map: 2D numpy array representing the occupancy grid
    - occupancy_map_resolution: The resolution of the map (meters per pixel)
    - occupancy_map_origin: Origin of the map in world coordinates (x, y, theta)
    """
    # plt.figure(figsize=(8, 8))
    plt.figure()

    add_occupancy_grid_to_plot(
        occupancy_map, occupancy_map_resolution, occupancy_map_origin
    )
    plt.title("Occupancy Grid Map")

    plt.grid(False)
    plt.show()


def plot_global_path(
    occupancy_map: np.ndarray,
    occupancy_map_resolution: float,
    occupancy_map_origin: Tuple[float, float, float],
    start_pose: Tuple[float, float, float],
    goal_pose: Tuple[float, float, float],
    path: List[Tuple[float, float, float]],
):
    # Create a figure and axis
    plt.figure(figsize=(8, 8))

    add_occupancy_grid_to_plot(
        occupancy_map, occupancy_map_resolution, occupancy_map_origin
    )
    plt.title("Occupancy Grid Map")

    # plot path
    if path is not None:
        plt.plot([x[0] for x in path], [x[1] for x in path])
    # plot start and goal position
    plt.plot(start_pose[0], start_pose[1], "go")
    plt.plot(goal_pose[0], goal_pose[1], "ro")

    # Show the plot
    plt.grid(False)
    plt.show()
