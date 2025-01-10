import os
from typing import Tuple, List, Optional
import warnings

import cv2
import yaml
import numpy as np
from scipy.ndimage import distance_transform_edt


# Load YAML configuration for ROS map
def load_yaml_config(yaml_path) -> dict:
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Load and process occupancy map using ROS conventions
def load_occupancy_map(
    config: dict, map_img_dir: Optional[str] = None, flip_y_axis=False
) -> Tuple[np.ndarray, float, List[float]]:
    # image_path = os.path.join(
    #     os.path.dirname(__file__), "../occupancy_maps", config["image"]
    # )
    if map_img_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        image_path = os.path.join(project_root, "occupancy_maps", config["image"])
    else:
        image_path = os.path.join(map_img_dir, config["image"])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Normalize the image to 0-1 scale
    normalized_image = (255.0 - image) / 255.0

    # Create an empty occupancy map with ROS conventions
    occupancy_map = np.full_like(normalized_image, -1)  # Initialize with -1 (unknown)

    # Assign values according to ROS occupancy conventions
    occupancy_map[normalized_image >= config["occupied_thresh"]] = 100  # Occupied
    occupancy_map[normalized_image <= config["free_thresh"]] = 0  # Free space

    # If negate flag is set, invert the values
    if config["negate"]:
        occupancy_map = 100 - occupancy_map
        occupancy_map[occupancy_map == -101] = -1  # Handle unknown values

    # Flip the map along the Y-axis if needed
    if flip_y_axis:
        occupancy_map = np.flipud(
            occupancy_map
        )  # Flip along the Y-axis (vertical flip)

    # Handle map resolution and origin
    resolution = config["resolution"]
    origin = config["origin"]
    print(f"Map loaded with resolution: {resolution} m/px and origin at: {origin}")
    if flip_y_axis:
        print("Map has been flipped along the Y-axis")

    return occupancy_map, resolution, origin


def load_occupancy_map_by_config_path(
    config_file_path: str, map_img_dir: Optional[str] = None, flip_y_axis: bool = False
):
    map_config = load_yaml_config(config_file_path)
    return load_occupancy_map(map_config, map_img_dir, flip_y_axis)


def pos_to_grid_cell_idx(
    pos: Tuple[float, float], resolution: float, origin: Tuple[float, float]
) -> Tuple[int, int]:
    gridmap_idx_i = int((pos[1] - origin[1]) / resolution)
    gridmap_idx_j = int((pos[0] - origin[0]) / resolution)
    return gridmap_idx_i, gridmap_idx_j


def grid_cell_idx_to_pos(
    cell: Tuple[int, int], resolution: float, origin: Tuple[float, float]
) -> Tuple[float, float]:
    pos_x = origin[0] + cell[1] * resolution + 0.5 * resolution
    pos_y = origin[1] + cell[0] * resolution + 0.5 * resolution
    return pos_x, pos_y


# check if pos is inside the occupancy map
def check_if_pos_inside_map(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    pos: Tuple[float, float],
) -> bool:
    """
    Check if the given position is inside the map boundaries.
    x_min <= pos_x <= x_max
    y_min <= pos_y <= y_max
    """

    if pos[0] < x_min or pos[0] >= x_max:
        return False
    if pos[1] < y_min or pos[1] >= y_max:
        return False
    return True


def check_if_pos_is_free(
    occupancy_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    pos: Tuple[float, float],
) -> bool:
    """
    Check if the given position is free in the occupancy map.
    """
    idx_i, idx_j = pos_to_grid_cell_idx(pos, resolution, origin)
    if occupancy_map[idx_i, idx_j] == 0:
        return True
    return False


def check_if_cell_is_free(occupancy_map: np.ndarray, cell: Tuple[int, int]) -> bool:
    """
    Check if the given cell is free in the occupancy map.
    """
    # Check if the cell is inside the map
    if cell[0] < 0 or cell[0] >= occupancy_map.shape[0]:
        return False
    if cell[1] < 0 or cell[1] >= occupancy_map.shape[1]:
        return False

    if occupancy_map[cell[0], cell[1]] == 0:
        return True
    return False


def check_path_segments_are_not_larger_than_threshold(
    path: List[Tuple[float, float, float]],
    threshold: float,
) -> bool:
    """
    Check if the path segments are larger than the threshold.
    """
    if len(path) == 1:
        return False

    for pos_idx in range(len(path) - 1):
        pos = path[pos_idx][:2]
        next_pos = path[pos_idx + 1][:2]
        if abs(pos[0] - next_pos[0]) + abs(pos[1] - next_pos[1]) <= threshold:
            continue
        if np.linalg.norm(np.array(pos) - np.array(next_pos)) > threshold:
            return True
    return False


def check_collision_for_path(
    occupancy_map: np.ndarray,
    resolution: float,
    origin: float,
    path: List[Tuple[float, float, float]],
) -> bool:
    """
    Return True if there is a collision.
    """

    if check_path_segments_are_not_larger_than_threshold(path, resolution):
        warnings.warn(
            "Some path segments are larger than the resolution. Collision check may not be accurate."
        )

    x_min = origin[0]
    y_min = origin[1]
    x_max = origin[0] + resolution * occupancy_map.shape[1]
    y_max = origin[1] + resolution * occupancy_map.shape[0]

    for pose_on_path in path:
        if not check_if_pos_inside_map(x_min, y_min, x_max, y_max, pose_on_path[:2]):
            return True
        if not check_if_pos_is_free(
            occupancy_map, resolution, origin, pose_on_path[:2]
        ):
            return True
    return False


def get_padded_occupancy_map(occupancy_map, distance, resolution):
    """
    Pads the occupancy map by marking cells within a given distance from occupied cells as occupied.

    Parameters:
        occupancy_map (np.array): The original occupancy map with 0 as free, 100 as occupied, and -1 as unknown.
        distance (float): The distance (in meters) to pad around occupied cells.
        resolution (float): The resolution of the occupancy map (meters per cell).

    Returns:
        np.array: The padded occupancy map.
    """
    # Calculate the radius of padding in terms of cells
    radius_cells = round(distance / resolution, 6)
    # Create a binary mask of occupied cells (100)
    occupied_mask = occupancy_map == 100

    # Compute the distance transform of the inverse of the occupied mask
    distance_map = distance_transform_edt(~occupied_mask)

    # Create the new padded occupancy map
    padded_occupancy_map = occupancy_map.copy()
    padded_occupancy_map[distance_map <= radius_cells] = 100

    return padded_occupancy_map


def line_of_sight(
    cell_i: Tuple[int, int], cell_f: Tuple[int, int], occupancy_map: np.ndarray
):
    """
    Supercovor implementation of Bresenham's line algorithm.
    Ref: http://eugen.dedu.free.fr/projects/bresenham/
    """
    if not check_if_cell_is_free(occupancy_map, cell_i) or not check_if_cell_is_free(
        occupancy_map, cell_f
    ):
        return False

    x = cell_i[0]
    y = cell_i[1]

    d_x = cell_f[0] - cell_i[0]
    d_y = cell_f[1] - cell_i[1]

    x_step = 1
    y_step = 1

    if d_y < 0:
        d_y = -d_y
        y_step = -1

    if d_x < 0:
        d_x = -d_x
        x_step = -1

    dd_y = 2 * d_y
    dd_x = 2 * d_x
    if dd_x >= dd_y:
        error_prev = d_x
        error = d_x

        for i in range(d_x):
            x += x_step
            error += dd_y
            if error > dd_x:
                y += y_step
                error -= dd_x

                if error + error_prev < dd_x:
                    if not check_if_cell_is_free(occupancy_map, (x, y - y_step)):
                        return False
                elif error + error_prev > dd_x:
                    if not check_if_cell_is_free(occupancy_map, (x - x_step, y)):
                        return False
                else:
                    if not check_if_cell_is_free(
                        occupancy_map, (x, y - y_step)
                    ) or not check_if_cell_is_free(occupancy_map, (x - x_step, y)):
                        return False
            if not check_if_cell_is_free(occupancy_map, (x, y)):
                return False
            error_prev = error
    else:
        error_prev = d_y
        error = d_y

        for i in range(d_y):
            y += y_step
            error += dd_x
            if error > dd_y:
                x += x_step
                error -= dd_y

                if error + error_prev < dd_y:
                    if not check_if_cell_is_free(occupancy_map, (x - x_step, y)):
                        return False
                elif error + error_prev > dd_y:
                    if not check_if_cell_is_free(occupancy_map, (x, y - y_step)):
                        return False
                else:
                    if not check_if_cell_is_free(
                        occupancy_map, (x - x_step, y)
                    ) or not check_if_cell_is_free(occupancy_map, (x, y - y_step)):
                        return False
            if not check_if_cell_is_free(occupancy_map, (x, y)):
                return False
            error_prev = error
    assert y == cell_f[1] and x == cell_f[0]
    return True


def bresenham_line_algorithm(
    cell_i: Tuple[int, int],
    cell_f: Tuple[int, int],
    occupancy_map: np.ndarray,
):
    x_0 = cell_i[0]
    y_0 = cell_i[1]
    x_1 = cell_f[0]
    y_1 = cell_f[1]

    d_x = abs(x_1 - x_0)
    d_y = -abs(y_1 - y_0)

    s_x = -1
    s_y = -1

    if x_0 < x_1:
        s_x = 1
    if y_0 < y_1:
        s_y = 1
    e = d_x + d_y

    while True:
        if not check_if_cell_is_free(occupancy_map, (x_0, y_0)):
            return False
        if (x_0, y_0) == cell_f:
            print(f"{(x_0, y_0)} == {cell_f}")
            return True
        e_2 = 2 * e
        if e_2 >= d_y:
            if x_0 == x_1:
                return True
            e += d_y
            x_0 += s_x
        if e_2 <= d_x:
            if y_0 == y_1:
                return True
            e += d_x
            y_0 += s_y
