import numpy as np
import pytest

from path_planners.utils.kinematic_utils import (
    check_unicycle_reachability,
    get_circle_radius_candidates,
    get_pose_connecting_arc_by_radius,
    get_pose_to_connect_poses_by_arcs,
    calculate_unicycle_final_yaw,
    calculate_unicycle_path_angular_velocity,
    calculate_unicycle_w_yaw,
)


def test_check_unicycle_reachability():
    # Test along the axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert check_unicycle_reachability(pose_i, (0.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.5, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (1.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (2.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (4.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (8.0, 0.0), 1.0, 1.0) == True

    pose_i = (0.0, 0.0, np.pi)
    assert check_unicycle_reachability(pose_i, (-0.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-0.5, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-1.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-2.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-4.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-8.0, 0.0), 1.0, 1.0) == True

    pose_i = (0.0, 0, 0.5 * np.pi)
    assert check_unicycle_reachability(pose_i, (0.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 0.5), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 1.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 2.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 4.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 8.0), 1.0, 1.0) == True

    # Test along the axis perpendicular to the heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert check_unicycle_reachability(pose_i, (0.0, 0.5), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, 1.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, 1.9), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, 2.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 4.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 8.0), 1.0, 1.0) == True

    assert check_unicycle_reachability(pose_i, (0.0, -0.5), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, -1.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, -1.9), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, -2.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, -4.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, -8.0), 1.0, 1.0) == True

    pose_i = (0.0, 0.0, np.pi)
    assert check_unicycle_reachability(pose_i, (0.0, 0.5), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, 1.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, 1.9), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.0, 2.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 4.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 8.0), 1.0, 1.0) == True

    pose_i = (0.0, 0, 0.5 * np.pi)
    assert check_unicycle_reachability(pose_i, (0.5, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (1.0, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (1.9, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (2.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (4.0, 0.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (8.0, 0.0), 1.0, 1.0) == True

    # Test along the diagonal direction
    pose_i = (0.0, 0.0, 0.0)
    assert check_unicycle_reachability(pose_i, (0.1, 0.1), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.5, 0.5), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (0.9, 0.9), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (1.0, 1.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (1.1, 1.1), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (2.0, 2.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (4.0, 4.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (8.0, 8.0), 1.0, 1.0) == True

    assert check_unicycle_reachability(pose_i, (-0.1, -0.1), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-0.5, -0.5), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-0.9, -0.9), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-1.0, -1.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-1.1, -1.1), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-2.0, -2.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-4.0, -4.0), 1.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (-8.0, -8.0), 1.0, 1.0) == True

    # Test edge cases: max_linear_velocity < 0.0
    with pytest.raises(
        ValueError, match=r".*max_angular_velocity should be non-negaitve.*"
    ):
        check_unicycle_reachability(pose_i, (0.0, 0.0), 1.0, -1.0)

    # Test edge cases: min_linear_velocity = 0
    assert check_unicycle_reachability(pose_i, (0.0, 0.0), 0.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.1, 0.0), 0.0, 1.0) == True
    assert check_unicycle_reachability(pose_i, (0.0, 0.1), 0.0, 1.0) == True

    # Test edge cases: max_angular_velocity = 0
    assert check_unicycle_reachability(pose_i, (1.0, 0.0), 1.0, 0.0) == True
    assert check_unicycle_reachability(pose_i, (1.0, 0.125), 1.0, 0.0) == False
    assert check_unicycle_reachability(pose_i, (1.0, -0.125), 1.0, 0.0) == False

    # Test edge cases: min_linear_velocity > 0 and pos_f is in line with the heading direction in the opposite direction
    assert check_unicycle_reachability(pose_i, (-0.5, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-1.0, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-2.0, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-4.0, 0.0), 1.0, 1.0) == False
    assert check_unicycle_reachability(pose_i, (-8.0, 0.0), 1.0, 1.0) == False

    pose_i = (0.0, 0, 0.25 * np.pi)
    assert check_unicycle_reachability(pose_i, (-1.0, -1.0), 1.0, 1.0) == False
    pose_i = (0.0, 0, 0.5 * np.pi)
    assert check_unicycle_reachability(pose_i, (0.0, -1.0), 1.0, 1.0) == False
    pose_i = (0.0, 0, np.pi)
    assert check_unicycle_reachability(pose_i, (1.0, 0.0), 1.0, 1.0) == False
    pose_i = (0.0, 0, 1.25 * np.pi)
    assert check_unicycle_reachability(pose_i, (1.0, 1.0), 1.0, 1.0) == False


def test_get_circle_radius_candidates():
    # Test left turn -> right turn with r = 1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert pytest.approx(1.0) in get_circle_radius_candidates(
            (0, 0, 0), (2 + np.cos(theta), 1 + np.sin(theta), theta - 0.5 * np.pi)
        )

    # Test right turn -> left turn with r = -1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert pytest.approx(-1.0) in get_circle_radius_candidates(
            (0, 0, 0), (2 + np.cos(theta), -1 + np.sin(theta), theta + 0.5 * np.pi)
        )


def test_get_pose_connecting_arc_by_radius():
    # Test left turn -> right turn with r = 1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert pytest.approx(
            (1.0, 1.0, 0.5 * np.pi)
        ) == get_pose_connecting_arc_by_radius(
            (0, 0, 0), (2 + np.cos(theta), 1 + np.sin(theta), theta - 0.5 * np.pi), 1.0
        )

    # Test right turn -> left turn with r = -1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert pytest.approx(
            (1.0, -1.0, -0.5 * np.pi)
        ) == get_pose_connecting_arc_by_radius(
            (0, 0, 0),
            (2 + np.cos(theta), -1 + np.sin(theta), theta + 0.5 * np.pi),
            -1.0,
        )


def test_get_pose_to_connect_poses_by_arcs():
    assert False == True, "Not implemented yet"


def test_calculate_unicycle_final_yaw():
    _NON_ZERO_OFFSET = 0.01
    # Test same position
    for theta in np.linspace(-np.pi + _NON_ZERO_OFFSET, np.pi, 100):
        pose_i = (0.0, 0.0, theta)
        assert calculate_unicycle_final_yaw(pose_i, (0.0, 0.0)) == theta

    # Test along the axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_unicycle_final_yaw(pose_i, (0.5, 0.0)) == 0.0
    assert calculate_unicycle_final_yaw(pose_i, (1.0, 0.0)) == 0.0
    assert calculate_unicycle_final_yaw(pose_i, (2.0, 0.0)) == 0.0

    assert calculate_unicycle_final_yaw(pose_i, (-0.5, 0.0)) == 0.0
    assert calculate_unicycle_final_yaw(pose_i, (-1.0, 0.0)) == 0.0
    assert calculate_unicycle_final_yaw(pose_i, (-2.0, 0.0)) == 0.0

    pose_i = (0.0, 0.0, np.pi)
    assert calculate_unicycle_final_yaw(pose_i, (-0.0, 0.0)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (-0.5, 0.0)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (-1.0, 0.0)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (-2.0, 0.0)) == np.pi

    # Test along the axis perpendicular to the heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_unicycle_final_yaw(pose_i, (0.0, 0.5)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (0.0, 1.0)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (0.0, 2.0)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (0.0, -0.5)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (0.0, -1.0)) == np.pi
    assert calculate_unicycle_final_yaw(pose_i, (0.0, -2.0)) == np.pi

    pose_i = (0.0, 0.0, 0.5 * np.pi)
    assert calculate_unicycle_final_yaw(pose_i, (0.5, 0.0)) == -0.5 * np.pi
    assert calculate_unicycle_final_yaw(pose_i, (1.0, 0.0)) == -0.5 * np.pi
    assert calculate_unicycle_final_yaw(pose_i, (2.0, 0.0)) == -0.5 * np.pi
    assert calculate_unicycle_final_yaw(pose_i, (-0.5, 0.0)) == -0.5 * np.pi
    assert calculate_unicycle_final_yaw(pose_i, (-1.0, 0.0)) == -0.5 * np.pi
    assert calculate_unicycle_final_yaw(pose_i, (-2.0, 0.0)) == -0.5 * np.pi


def test_calculate_unicycle_path_angular_velocity():
    _NON_ZERO_OFFSET = 0.01
    # Test same position
    for theta in np.linspace(-np.pi + _NON_ZERO_OFFSET, np.pi, 100):
        pose_i = (0.0, 0.0, theta)
        assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, 0.0), 1.0) == 0.0

    # Test along the axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.5, 0.0), 1.0) == 0.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (1.0, 0.0), 1.0) == 0.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (2.0, 0.0), 1.0) == 0.0

    assert calculate_unicycle_path_angular_velocity(pose_i, (-0.5, 0.0), 1.0) == 0.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (-1.0, 0.0), 1.0) == 0.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (-2.0, 0.0), 1.0) == 0.0

    pose_i = (0.0, 0.0, 0.25 * np.pi)
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.5, 0.5), 1.0) == 0.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (1.0, 1.0), 1.0) == 0.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (2.0, 2.0), 1.0) == 0.0

    # Test along the axis perpendicular to the heading direction (w=v/r)
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, 0.5), 1.0) == 4.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, 1.0), 1.0) == 2.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, 2.0), 1.0) == 1.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, 4.0), 1.0) == 0.5

    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, -0.5), 1.0) == -4.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, -1.0), 1.0) == -2.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, -2.0), 1.0) == -1.0
    assert calculate_unicycle_path_angular_velocity(pose_i, (0.0, -4.0), 1.0) == -0.5

    # Test along the circle with radius 1
    pose_i = (0.0, 0.0, 0.5 * np.pi)
    center_of_turning_left = (-1.0, 0.0)
    center_of_turning_right = (1.0, 0.0)
    turning_radius = 1.0
    for theta in np.linspace(-np.pi + _NON_ZERO_OFFSET, np.pi - _NON_ZERO_OFFSET, 100):
        pos_f_right = (
            center_of_turning_right[0] + turning_radius * np.cos(theta),
            center_of_turning_right[1] + turning_radius * np.sin(theta),
        )
        pos_f_left = (
            center_of_turning_left[0] + turning_radius * np.cos(np.pi - theta),
            center_of_turning_left[1] + turning_radius * np.sin(np.pi - theta),
        )
        # Test positive linear velocity
        assert calculate_unicycle_path_angular_velocity(
            pose_i, pos_f_left, 1.0
        ) == pytest.approx(1.0)
        assert calculate_unicycle_path_angular_velocity(
            pose_i, pos_f_right, 1.0
        ) == pytest.approx(-1.0)

        # Test negative linear velocity
        assert calculate_unicycle_path_angular_velocity(
            pose_i, pos_f_left, -1.0
        ) == pytest.approx(-1.0)
        assert calculate_unicycle_path_angular_velocity(
            pose_i, pos_f_right, -1.0
        ) == pytest.approx(1.0)


def test_calculate_unicycle_w_yaw():
    assert calculate_unicycle_w_yaw((0, 0, 0), (1, 0), 1.0) == (0, 0)
    assert calculate_unicycle_w_yaw((0, 0, 0), (0, 2.0), 1.0) == (1.0, np.pi)
