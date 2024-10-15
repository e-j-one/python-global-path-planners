from typing import Tuple, Optional, List
from enum import Enum

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.math_utils as MathUtils


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
