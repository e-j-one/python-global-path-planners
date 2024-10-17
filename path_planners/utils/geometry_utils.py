from typing import Tuple
import warnings

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


def get_path_length(path: Tuple[float, float, float]) -> float:
    """
    Calculate the length of the path
    """
    if len(path) == 0:
        warnings.warn("Path is empty.")
        return 0.0

    length = 0.0
    for i in range(len(path) - 1):
        length += np.linalg.norm(np.array(path[i][:2]) - np.array(path[i + 1][:2]))
    return length


def check_if_dist_is_below_threshold(
    pos_i: Tuple[float, float], pos_f: Tuple[float, float], threshold: float
):
    """
    Check if the pos_i is near the pos_f within the threshold (dist < threshold)

    Parameters:
    - pos_i: (x, y) of the initial pose
    - pos_f: (x, y) of the final position
    - threshold: threshold distance to check if the pos_i is near pos_f
    """
    return np.linalg.norm(np.array(pos_i[:2]) - np.array(pos_f)) < threshold


def check_if_angle_diff_is_below_threshold(
    angle_i: float, angle_f: float, threshold: float
):
    """
    Check if the angle_i is near the angle_f within the threshold (abs(angle_i - angle_f) < threshold)

    Parameters:
    - angle_i: initial angle
    - angle_f: final angle
    - threshold: threshold angle difference to check if the angle_i is near angle_f
    """
    return abs(MathUtils.normalize_angle(angle_i - angle_f)) < threshold


def check_if_dist_and_angle_diff_are_below_threshold(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
    dist_threshold: float,
    angle_threshold: float,
):
    """
    Check if the pose_i is near the pose_f within the threshold (dist < dist_threshold) and (angle_i - angle_f < angle_threshold)
    """
    return check_if_dist_is_below_threshold(
        pose_i[:2], pose_f[:2], dist_threshold
    ) and check_if_angle_diff_is_below_threshold(pose_i[2], pose_f[2], angle_threshold)


def check_if_pos_in_same_side_of_heading(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float]
) -> bool:
    vector_i_f = (pos_f[0] - pose_i[0], pos_f[1] - pose_i[1])
    vector_i_heading = (np.cos(pose_i[2]), np.sin(pose_i[2]))
    dot_product = np.dot(vector_i_f, vector_i_heading)
    return dot_product > 1e-12  # floating point error


def check_if_yaws_and_direction_of_poses_align(
    pose_i: Tuple[float, float, float], pose_f: Tuple[float, float, float]
):
    """
    Check if yaw_i and yaw_f are aligned with direction of the vector from pos_i to pos_f
    """
    # Check if yaw_i and yaw_f are aligned
    if not check_if_angle_diff_is_below_threshold(pose_i[2], pose_f[2], 1e-12):
        return False

    if check_if_dist_is_below_threshold(pose_i[:2], pose_f[:2], 1e-12):
        return True

    # Check if direction of the vector from pos_i to pos_f is aligned with yaw_i
    vector_i_f = (pose_f[0] - pose_i[0], pose_f[1] - pose_i[1])
    vector_i_f_yaw = np.atan2(vector_i_f[1], vector_i_f[0])
    vector_f_i_yaw = np.atan2(-vector_i_f[1], -vector_i_f[0])
    print("vector_i_f_yaw", vector_i_f_yaw, "vector_f_i_yaw", vector_f_i_yaw)

    print("pose_i", pose_i, "pose_f", pose_f)

    if check_if_angle_diff_is_below_threshold(
        vector_i_f_yaw, pose_i[2], 1e-12
    ) or check_if_angle_diff_is_below_threshold(vector_f_i_yaw, pose_i[2], 1e-12):
        return True
    return False
