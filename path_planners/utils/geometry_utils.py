from typing import Tuple

import numpy as np

import path_planners.utils.math_utils as MathUtils


def calculate_delta_i_f(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float]
) -> float:
    """
    Return angle difference between heading direction of pose_i and vector_i_f in -pi to pi
    Normalize the angle difference to -pi to pi.
    If the pose_i and pos_f are in the same position, return 0.

    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pos_f: (x, y) of the final position
    """
    vector_i_f = (pos_f[0] - pose_i[0], pos_f[1] - pose_i[1])
    if np.linalg.norm(vector_i_f) == 0:
        return 0
    delta_i_f = np.arctan2(vector_i_f[1], vector_i_f[0]) - pose_i[2]

    return MathUtils.normalize_angle(delta_i_f)
