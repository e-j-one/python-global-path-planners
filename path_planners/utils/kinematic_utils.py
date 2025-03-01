# TODO: separate files by functionality or robot type

import warnings
from typing import Tuple, Optional, List

import numpy as np

import path_planners.utils.geometry_utils as GeometryUtils
import path_planners.utils.math_utils as MathUtils

MAX_PATH_LENGTH = 1e6


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


def check_if_pose_can_be_connected_by_arc(
    pose_i: Tuple[float, float, float], pose_f: Tuple[float, float, float]
):
    """
    Check if there exists a valid arc path connecting pose_i and pose_f
    Return false if the poses are in the same position or the same direction
    """
    # Check if yaw_i and yaw_f are aligned
    if GeometryUtils.check_if_angle_diff_is_below_threshold(
        pose_i[2], pose_f[2], 1e-12
    ):
        return False

    if GeometryUtils.check_if_dist_is_below_threshold(pose_i[:2], pose_f[:2], 1e-12):
        return False

    pos_f_yaw_from_i_along_arc = calculate_final_yaw_of_arc_path(pose_i, pose_f[:2])
    return GeometryUtils.check_if_angle_diff_is_below_threshold(
        pose_f[2], pos_f_yaw_from_i_along_arc, 1e-12
    )


def _get_turning_radius_candidates_to_connect_pose_with_two_arcs(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
) -> List[float]:
    """
    Return turning radius candidates of two arcs that can connects pose_i and pose_f.
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


def _get_pose_connecting_arc_paths_by_radius(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
    radius: float,
) -> Tuple[float, float, float]:
    """
    Return the pose connecting pose_i and pose_f by the two arc with the same given radius

    Parameters:
    - pose_i: (x, y, yaw) of the initial pose
    - pose_f: (x, y, yaw) of the final pose
    - radius: radius of the arc (non-zero, positive for left turn first, negative for right turn first)
    """
    # Get the center of the circle
    perp_i = (np.cos(pose_i[2] + 0.5 * np.pi), np.sin(pose_i[2] + 0.5 * np.pi))
    perp_f = (np.cos(pose_f[2] + 0.5 * np.pi), np.sin(pose_f[2] + 0.5 * np.pi))

    # Get the centers of the circle
    center_i = (pose_i[0] + radius * perp_i[0], pose_i[1] + radius * perp_i[1])
    center_f = (pose_f[0] - radius * perp_f[0], pose_f[1] - radius * perp_f[1])

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

    # print("pose_i", pose_i, "pose_f", pose_f, "center_mid", center_mid, "yaw", yaw)
    # print("center_i", center_i, "center_f", center_f)

    return (center_mid[0], center_mid[1], yaw)


def _get_arc_path_length(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
) -> float:
    """
    Given two poses, return the path length of the arc connecting the two poses

    Returns:
    - path_length: path length of the arc
    """

    # 1. Calculate radius of the circle with center at +-r in the y direction from the pose_i
    vector_i_f = np.array(pos_f) - np.array([pose_i[0], pose_i[1]])
    dist_i_f = np.linalg.norm(vector_i_f)
    if dist_i_f < 1e-12:
        return 0.0

    # delta_i_f: angle difference between heading direction of pose_i and vector_i_f in -pi to pi
    delta_i_f = GeometryUtils.calculate_delta_i_f(pose_i, pos_f)
    if abs(delta_i_f) < 1e-12:
        return dist_i_f
    elif abs(np.sin(delta_i_f)) < 1e-12:
        # goal pos is in the opposite direction of the heading direction
        return np.inf

    turning_radius = 0.5 * dist_i_f / np.sin(delta_i_f)

    pos_f_yaw = calculate_final_yaw_of_arc_path(pose_i, pos_f)

    angle_diff = pos_f_yaw - pose_i[2]
    # Normalize the angle difference to [0, 2pi)
    if delta_i_f > 0:  # left turn
        angle_diff = MathUtils.normalize_angle_positive(angle_diff)
        return turning_radius * angle_diff
    else:
        angle_diff = MathUtils.normalize_angle_positive(-angle_diff)
        return -turning_radius * angle_diff


def get_pose_to_connect_poses_by_two_arcs(
    pose_i: Tuple[float, float, float],
    pose_f: Tuple[float, float, float],
    min_turning_radius: float,
) -> Optional[Tuple[Tuple[float, float, float], float, float]]:
    """
    NOTE: This function only connects two poses with two arcs with the same radius or the straight line.
    - There can be better path with different radius of the arcs which is not implemented in this function.

    Get two arcs with the same radius connecting pose_i and pose_f
    - that has the minimum path length
    - while satisfying the velocity constraints.

    Returns: None if there is no valid path or path can be connected by a straight line
    - pose_stopover: pose to stopover
    - path_length: path length of the arcs
    - radius: radius of the arc
    """

    # 0. Check if poses can be connected by straight line or single arc
    if GeometryUtils.check_if_yaws_and_direction_of_poses_align(pose_i, pose_f):
        return None

    if check_if_pose_can_be_connected_by_arc(pose_i, pose_f):
        return None

    # 1. Get candidates turning radius of arcs
    radius_candidates = _get_turning_radius_candidates_to_connect_pose_with_two_arcs(
        pose_i, pose_f
    )

    # print("radius_candidates", radius_candidates)

    # 2. Validate & choose the valid radius
    if len(radius_candidates) == 0:
        return None

    optimal_path_length = np.inf
    optimal_stopover_pose = None

    for radius in radius_candidates:
        if abs(radius) < min_turning_radius:
            continue

        # Get the pose connecting two arcs
        pose_stopover = _get_pose_connecting_arc_paths_by_radius(pose_i, pose_f, radius)
        path_length_of_candidate = _get_arc_path_length(
            pose_i, pose_stopover[:2]
        ) + _get_arc_path_length(pose_stopover, pose_f[:2])

        if path_length_of_candidate < optimal_path_length:
            optimal_path_length = path_length_of_candidate
            optimal_stopover_pose = pose_stopover

    if optimal_stopover_pose is None:
        return None
    return optimal_stopover_pose, optimal_path_length


def calculate_final_yaw_of_arc_path(
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


def calculate_angular_velocity_to_reach_pos(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
    linear_velocity: float,
) -> float:
    """
    Given the pose_i, pos_f and linear velocity, return the angular velocity (w) and yaw to reach pos_f from pose_i
    1. Calculate radius (=r) of the path
    2. w = v_x / r
    """
    turning_radius = GeometryUtils.calculate_arc_path_radius(pose_i, pos_f)
    if turning_radius == 0:  # Straight line
        return 0.0
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
    return calculate_angular_velocity_to_reach_pos(
        pose_i, pos_f, linear_velocity
    ), calculate_final_yaw_of_arc_path(pose_i, pos_f)


def get_straingt_path(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float], d_s: float
) -> List[Tuple[float, float, float]]:
    """
    Get the straight line path from pose_i to pos_f where s
    """
    vector_i_f = (pos_f[0] - pose_i[0], pos_f[1] - pose_i[1])
    dist_i_f = np.linalg.norm(vector_i_f)
    if dist_i_f == 0:
        return [pose_i]
    unit_i_f = (vector_i_f[0] / dist_i_f, vector_i_f[1] / dist_i_f)

    if not GeometryUtils.check_if_pos_in_same_side_of_heading(pose_i, pos_f):
        raise ValueError(
            "The pose_i and pos_f are not in the same side of the heading direction"
        )

    num_steps = int(dist_i_f // d_s)
    path = [pose_i]

    for i in range(num_steps):
        s = (i + 1) * d_s
        x = pose_i[0] + s * unit_i_f[0]
        y = pose_i[1] + s * unit_i_f[1]
        path.append((x, y, pose_i[2]))

    if path[-1][:2] != pos_f:
        path.append((*pos_f, pose_i[2]))

    return path


def get_unicycle_path(
    pose_i: Tuple[float, float, float], pos_f: Tuple[float, float], d_s: float
) -> Optional[List[Tuple[float, float, float]]]:
    """
    Get the arc path from pose_i to pos_f.
    1. Calculate the radius of the arc
    2. Calculate the angle difference per each step
    3. Generate the path with the angle difference until reaching the pos_f

    Distance between two consecutive poses is d_s but the last pose is guaranteed to be pos_f
    """

    turning_radius = GeometryUtils.calculate_arc_path_radius(pose_i, pos_f)
    if turning_radius == np.inf:  # Straight line
        return get_straingt_path(pose_i, pos_f, d_s)

    if turning_radius == 0:  # Same position
        return [pose_i]

    turning_center = (
        pose_i[0] + turning_radius * np.cos(pose_i[2] + 0.5 * np.pi),
        pose_i[1] + turning_radius * np.sin(pose_i[2] + 0.5 * np.pi),
    )

    theta_goal = np.arctan2(pos_f[1] - turning_center[1], pos_f[0] - turning_center[0])
    d_theta = d_s / turning_radius

    radius_dir_sign = np.sign(turning_radius)

    theta_start = MathUtils.normalize_angle(pose_i[2] - radius_dir_sign * 0.5 * np.pi)
    theta_diff = MathUtils.normalize_angle_positive(
        radius_dir_sign * (theta_goal - theta_start)
    )
    num_steps = int(theta_diff // (radius_dir_sign * d_theta))

    if num_steps > MAX_PATH_LENGTH:
        warnings.warn(f"Too many steps ({num_steps}) on the path. Returning None.")
        return None

    path = [pose_i]

    theta = theta_start
    for i in range(num_steps):
        theta = theta_start + (i + 1) * d_theta

        next_pose = (
            turning_center[0] + radius_dir_sign * turning_radius * np.cos(theta),
            turning_center[1] + radius_dir_sign * turning_radius * np.sin(theta),
            MathUtils.normalize_angle(theta + radius_dir_sign * 0.5 * np.pi),
        )
        path.append(next_pose)

    if not MathUtils.check_if_angle_diff_is_within_threshold(theta, theta_goal, 1e-12):
        path.append(
            (
                *pos_f,
                MathUtils.normalize_angle(theta_goal + radius_dir_sign * 0.5 * np.pi),
            )
        )
    # print("================================")
    # print(f"pose_i: {pose_i} pose_f: {pos_f}")
    # print("turning_radius", turning_radius)
    # print("turning_center", turning_center)
    # print("radius_dir_sign", radius_dir_sign)
    # print(
    #     f"theta start: {theta_start:.2f}, theta goal: {theta_goal:.2f}, theta_diff: {theta_diff:.2f} d_theta: {d_theta:.2f}"
    # )
    # print(f"num_steps: {num_steps}")
    # print("path: ")
    # DebugUtils.print_path(path)

    return path


def interpolate_path_between_arc(path: List[Tuple[float, float, float]], d_s: float):
    """
    Interpolate the path using the arc path with the given resolution
    - For each two consecutive poses, get the arc path between the poses and interpolate the path
    """
    interpolated_path = []
    for i in range(len(path) - 1):
        pose_i = path[i]
        pos_f = path[i + 1][:2]
        interpolated_path += get_unicycle_path(pose_i, pos_f, d_s)[:-1]

    interpolated_path.append(path[-1])

    return interpolated_path


def interpolate_path_using_arc(path: List[Tuple[float, float, float]], d_s: float):
    """
    Interpolate the path using the arc path with the given resolution
    - The pose in original path may not be in the interpolated path
    """
    interpolated_path = []
    interpolated_path.append(path[0])

    distance_to_interpolate_from_last_original_pose = d_s

    for i in range(len(path) - 1):
        pose_arc_start = path[i]
        pos_arc_end = path[i + 1][:2]

        arc_length = _get_arc_path_length(pose_arc_start, pos_arc_end)

        if distance_to_interpolate_from_last_original_pose > arc_length:
            distance_to_interpolate_from_last_original_pose -= arc_length
            continue

        while True:
            interpolated_pose = _get_pose_on_arc(
                pose_arc_start,
                pos_arc_end,
                distance_to_interpolate_from_last_original_pose,
            )

            if interpolated_pose is None:
                raise ValueError("The interpolated pose is None")

            interpolated_path.append(interpolated_pose)

            distance_to_interpolate_from_last_original_pose += d_s
            if distance_to_interpolate_from_last_original_pose > arc_length:
                distance_to_interpolate_from_last_original_pose -= arc_length
                break

    return interpolated_path


def _get_pose_on_arc(
    pose_i: Tuple[float, float, float],
    pos_f: Tuple[float, float],
    distance_from_pose_i: float,
) -> Optional[Tuple[float, float, float]]:
    """
    Returns
    - pose: pose on the arc path from pose_i to pos_f with the given distance
    - is pose on the arc: True if the pose is on the arc path, False if the pose is beyond the pos_f
    """
    if _get_arc_path_length(pose_i, pos_f) < distance_from_pose_i:
        return None

    turning_radius = GeometryUtils.calculate_arc_path_radius(pose_i, pos_f)
    if turning_radius == np.inf:  # Straight line
        return (
            pose_i[0] + distance_from_pose_i * np.cos(pose_i[2]),
            pose_i[1] + distance_from_pose_i * np.sin(pose_i[2]),
            pose_i[2],
        )

    if turning_radius == 0:  # Same position
        raise ValueError("The pose_i and pos_f are in the same position")

    turning_center = (
        pose_i[0] + turning_radius * np.cos(pose_i[2] + 0.5 * np.pi),
        pose_i[1] + turning_radius * np.sin(pose_i[2] + 0.5 * np.pi),
    )
    d_theta = distance_from_pose_i / turning_radius
    radius_dir_sign = np.sign(turning_radius)

    theta_start = MathUtils.normalize_angle(pose_i[2] - radius_dir_sign * 0.5 * np.pi)
    theta = theta_start + d_theta

    return (
        turning_center[0] + radius_dir_sign * turning_radius * np.cos(theta),
        turning_center[1] + radius_dir_sign * turning_radius * np.sin(theta),
        MathUtils.normalize_angle(theta + radius_dir_sign * 0.5 * np.pi),
    )
