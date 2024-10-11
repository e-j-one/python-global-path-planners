# TODO: separate files by functionality or robot type

from typing import Tuple, Optional, List

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
