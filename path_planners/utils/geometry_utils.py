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


def calculate_arc_path_radius(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float]
) -> float:
    """
    Calculate the radius of the arc path from pose_i to pos_f
    - r = 0.5 * dist_i_f / sin(delta_i_f)
    - r is positive if pos_f is in the left side of the heading direction of pose_i

    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pos_f: (x, y) of the final position

    Returns:
    - radius of the arc path. 0 if the pose_i and pos_f are in the same position. np.inf if the pose_i and pos_f are in the same direction.
    """
    dist_i_f = np.linalg.norm((pos_f[0] - pose_i[0], pos_f[1] - pose_i[1]))
    # delta_i_f: angle difference between heading direction of pose_i and vector_i_f in -pi to pi
    if dist_i_f == 0:
        return 0.0
    delta_i_f = calculate_delta_i_f(pose_i, pos_f)
    if abs(np.sin(delta_i_f)) < 1e-12:
        return np.inf
    return 0.5 * dist_i_f / np.sin(delta_i_f)


def check_if_dist_is_below_threshold(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float], threshold: float
):
    """
    Check if the pose_i is near the pos_f within the threshold (dist < threshold)

    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pos_f: (x, y) of the final position
    - threshold: threshold distance to check if the pose_i is near pos_f
    """
    return np.linalg.norm(np.array(pose_i[:2]) - np.array(pos_f)) < threshold


def check_if_pos_in_same_side_of_heading(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float]
) -> bool:
    vector_i_f = (pos_f[0] - pose_i[0], pos_f[1] - pose_i[1])
    vector_i_heading = (np.cos(pose_i[2]), np.sin(pose_i[2]))
    dot_product = np.dot(vector_i_f, vector_i_heading)
    return dot_product > 1e-12  # floating point error
