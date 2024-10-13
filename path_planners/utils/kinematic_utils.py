# TODO: separate files by functionality or robot type

from typing import Tuple, Optional, List
from enum import Enum

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.math_utils as MathUtils


def check_unicycle_reachability(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
    min_linear_velocity: float,
    max_angular_velocity: float,
) -> bool:
    """
    If robot can reach pos_f from pose_i with the given linear (v_x) and angular velocity (w) constraints return True
    - Check if the pos_f is in the unicycle's reachable set from pose_i
    - The unreachable set of unicycle with constant linear velocity is a circle with r=w/v
        with center at the +-r in the y direction from the pose_i
    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pos_f: (x, y) of the final position
    - min_linear_velocity: minimum linear velocity of the robot
    - max_angular_velocity: maximum angular velocity of the robot (>=0)
    """
    delta_i_f = GeometryUtils.calculate_delta_i_f(pose_i, pos_f)

    if max_angular_velocity < 0:
        raise ValueError("max_angular_velocity should be non-negaitve!")
    elif max_angular_velocity == 0:
        return delta_i_f == 0

    if min_linear_velocity <= 0:
        # If min_linear_velocity is not positive, robot can go to any position
        return True

    if min_linear_velocity > 0 and delta_i_f == np.pi:
        # If min_linear_velocity is positive and delta_i_f is pi, robot can't reach the position
        # - Because it's in the opposite direction of the heading direction (path_length is infinite)
        return False

    unreachable_set_radius = min_linear_velocity / max_angular_velocity
    unreachable_set_left_center = (
        pose_i[0] - unreachable_set_radius * np.sin(pose_i[2]),
        pose_i[1] + unreachable_set_radius * np.cos(pose_i[2]),
    )
    unreachable_set_right_center = (
        pose_i[0] + unreachable_set_radius * np.sin(pose_i[2]),
        pose_i[1] - unreachable_set_radius * np.cos(pose_i[2]),
    )

    if (
        np.linalg.norm(np.array(pos_f) - np.array(unreachable_set_left_center))
        >= unreachable_set_radius
        and np.linalg.norm(np.array(pos_f) - np.array(unreachable_set_right_center))
        >= unreachable_set_radius
    ):
        return True
    return False


def get_circle_radius_candidates(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
) -> List[float]:
    """
    Return radius candidates of two arcs that can connects pose_i and pose_f.
    Positive for left turn first, negative for right turn first.

    pose_i = (x_i, y_i, yaw_i)
    pose_f = (x_f, y_f, yaw_f)

    pos_i = (x_i, y_i)
    pos_f = (x_f, y_f)
    perp_i = (cos(yaw_i + 0.5*pi), sin(yaw_i + 0.5*pi))
    perp_f = (cos(yaw_f + 0.5*pi), sin(yaw_f + 0.5*pi))

    |(pos_f - r * perp_f) - (pos_i + r * perp_i)| = |2 * r|
    |(pos_f - pos_i) - r * (perp_f + perp_i)| = |2 * r|
    """
    # |pos_diff - r * perp_sum| = |2 * r|
    pos_diff = (pose_f[0] - pose_i[0], pose_f[1] - pose_i[1])
    perp_sum = (
        np.cos(pose_i[2] + 0.5 * np.pi) + np.cos(pose_f[2] + 0.5 * np.pi),
        np.sin(pose_i[2] + 0.5 * np.pi) + np.sin(pose_f[2] + 0.5 * np.pi),
    )

    # (perp_sum ⋅ perp_sum - 4) * r^2 + perp_sum ⋅ pos_dff * r + pos_diff ⋅ pos_diff = 2
    denominator = perp_sum[0] ** 2 + perp_sum[1] ** 2 - 4
    # print("denominator", denominator)

    pos_diff_dot_perp_sum = pos_diff[0] * perp_sum[0] + pos_diff[1] * perp_sum[1]
    pos_diff_dot_pos_diff = pos_diff[0] ** 2 + pos_diff[1] ** 2

    if denominator == 0:
        if pos_diff_dot_perp_sum == 0:
            # Could be w = 0 !
            # print("Headings are same and perp_sum ⋅ pos_dff = 0")
            return []
        turning_radius = 0.5 * pos_diff_dot_pos_diff / pos_diff_dot_perp_sum
        # print("Headings are same but perp_sum ⋅ pos_dff != 0")
        # print("turning_radius", turning_radius)
        return [turning_radius]

    discriminant = (
        4 * pos_diff_dot_pos_diff
        - (perp_sum[0] * pos_diff[1] - perp_sum[1] * pos_diff[0]) ** 2
    )

    turning_radius_candidates = [
        (pos_diff_dot_perp_sum + np.sqrt(discriminant)) / denominator,
        (pos_diff_dot_perp_sum - np.sqrt(discriminant)) / denominator,
    ]

    return turning_radius_candidates


def get_pose_connecting_arc_by_radius(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
    radius: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Return the pose connecting pose_i and pose_f by the two arc with the same given radius
    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pose_f: (x, y, yaw) of the final pose
    - radius: radius of the arc (non-zero, positive for left turn first, negative for right turn first)
    """
    # Get the center of the circle
    pos_i = (pose_i[0], pose_i[1])
    pos_f = (pose_f[0], pose_f[1])
    perp_i = (np.cos(pose_i[2] + 0.5 * np.pi), np.sin(pose_i[2] + 0.5 * np.pi))
    perp_f = (np.cos(pose_f[2] + 0.5 * np.pi), np.sin(pose_f[2] + 0.5 * np.pi))

    # Get the centers of the circle
    center_i = (pos_i[0] + radius * perp_i[0], pos_i[1] + radius * perp_i[1])
    center_f = (pos_f[0] - radius * perp_f[0], pos_f[1] - radius * perp_f[1])

    # Get the midpoint of two centers
    center_mid = (
        0.5 * (center_i[0] + center_f[0]),
        0.5 * (center_i[1] + center_f[1]),
    )

    # Get the vector from center_i to center_mid
    vector_center_i_mid = (center_mid[0] - center_i[0], center_mid[1] - center_i[1])

    # Get the yaw of the mid pose
    # r > 0
    if radius > 0:
        yaw = np.arctan2(vector_center_i_mid[1], vector_center_i_mid[0]) + 0.5 * np.pi
    else:
        yaw = np.arctan2(vector_center_i_mid[1], vector_center_i_mid[0]) - 0.5 * np.pi

    print("pose_i", pose_i, "pose_f", pose_f, "center_mid", center_mid, "yaw", yaw)
    print("center_i", center_i, "center_f", center_f)

    return (center_mid[0], center_mid[1], yaw)


def get_pose_to_connect_poses_by_arcs(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
    min_linear_velocity: float,
    max_angular_velocity: float,
):
    """
    Return the pose to connect pose_i and pose_f by two arcs with the given constraints
    """

    # 1. Get candidates turning radius of arcs
    radius_candidates = get_circle_radius_candidates(pose_i, pose_f)

    # 2. Validate & choose the valid radius
    if len(radius_candidates) == 0:
        return None

    # Get the pose connecting two arcs
    for radius in radius_candidates:
        pass

    return


def calculate_unicycle_final_yaw(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
) -> float:
    """
    Given the pose_i, pos_f and linear velocity, return the angular velocity (w) and yaw to reach pos_f from pose_i
    1. Calculate radius of the circle with center at +-r in the y direction from the pose_i
    2. Calculate the yaw difference between the pose_i and the final pose (delta theta)
    """
    delta_i_f = GeometryUtils.calculate_delta_i_f(pose_i, pos_f)
    pose_f_yaw = pose_i[2] + 2.0 * delta_i_f

    return MathUtils.normalize_angle(pose_f_yaw)


def calculate_unicycle_path_angular_velocity(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
    linear_velocity: float,
) -> float:
    """
    Given the pose_i, pos_f and linear velocity, return the angular velocity (w) and yaw to reach pos_f from pose_i
    1. Calculate radius (=r) of the path
        - r = 0.5 * dist_i_f / sin(delta_i_f)
        - r is positive if pos_f is in the left side of the heading direction of pose_i
    2. w = v_x / r
    """
    # 1. Calculate radius of the circle with center at +-r in the y direction from the pose_i
    vector_i_f = np.array(pos_f) - np.array([pose_i[0], pose_i[1]])
    dist_i_f = np.linalg.norm(vector_i_f)
    if dist_i_f < 1e-12:
        return 0.0

    # delta_i_f: angle difference between heading direction of pose_i and vector_i_f in -pi to pi
    delta_i_f = GeometryUtils.calculate_delta_i_f(pose_i, pos_f)
    if abs(np.sin(delta_i_f)) < 1e-12:
        return 0.0

    turning_radius = 0.5 * dist_i_f / np.sin(delta_i_f)
    angular_velocity = linear_velocity / turning_radius

    return angular_velocity


def calculate_unicycle_w_yaw(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
    linear_velocity: float,
) -> Tuple[float, float]:
    """
    Return angular velocity and yaw of final pos to reach pos_f from pose_i with the given linear velocity
    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pos_f: (x, y) of the final position
    - linear_velocity: linear velocity of the robot
    Returns:
    - angular_velocity: angular velocity to reach pos_f from pose_i
    - yaw: yaw of the final position
    """
    return calculate_unicycle_path_angular_velocity(
        pose_i, pos_f, linear_velocity
    ), calculate_unicycle_final_yaw(pose_i, pos_f)
