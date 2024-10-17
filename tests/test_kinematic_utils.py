import numpy as np
import pytest

from path_planners.utils.kinematic_utils import (
    check_unicycle_reachability,
    check_if_pose_can_be_connected_by_arc,
    _get_turning_radius_candidates_to_connect_pose_with_two_arcs,
    _get_pose_connecting_arc_paths_by_radius,
    _get_arc_path_length,
    get_pose_to_connect_poses_by_two_arcs,
    calculate_final_yaw_of_arc_path,
    calculate_angular_velocity_to_reach_pos,
    calculate_unicycle_w_yaw,
    get_straingt_path,
    get_unicycle_path,
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

    pose_i = (0.0, 0.0, 0.5 * np.pi)
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

    pose_i = (0.0, 0.0, 0.5 * np.pi)
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

    pose_i = (0.0, 0.0, 0.25 * np.pi)
    assert check_unicycle_reachability(pose_i, (-1.0, -1.0), 1.0, 1.0) == False
    pose_i = (0.0, 0.0, 0.5 * np.pi)
    assert check_unicycle_reachability(pose_i, (0.0, -1.0), 1.0, 1.0) == False
    pose_i = (0.0, 0.0, np.pi)
    assert check_unicycle_reachability(pose_i, (1.0, 0.0), 1.0, 1.0) == False
    pose_i = (0.0, 0.0, 1.25 * np.pi)
    assert check_unicycle_reachability(pose_i, (1.0, 1.0), 1.0, 1.0) == False


def test_check_if_pose_can_be_connected_by_arc():
    # Arrange
    _NON_ZERO_OFFSET = 0.01
    pose_i = (0.0, 0.0, 0.0)
    pose_f_along_circle = []

    circle_radiuses = [0.5, 1.0, 2.0]
    theta_candidates_up = np.linspace(0.0 * np.pi, 2.0 * np.pi, 100)

    for radius in circle_radiuses:
        for theta in theta_candidates_up:
            if theta != 1.5 * np.pi:
                pose_f_along_circle.append(
                    (
                        0.0 + radius * np.cos(theta),
                        radius + radius * np.sin(theta),
                        theta + 0.5 * np.pi,
                    )
                )
            if theta != 0.5 * np.pi:
                pose_f_along_circle.append(
                    (
                        0.0 + radius * np.cos(theta),
                        -radius + radius * np.sin(theta),
                        theta - 0.5 * np.pi,
                    )
                )

    pose_f_along_line = [
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, np.pi),
        (0.0, 0.0, np.pi),
        (1.0, 0.0, np.pi),
    ]

    unreachable_pose_f = [
        (1.0, 1.0, 0.0),
        (1.0, 1.0, np.pi),
        (1.0, 1.0, -0.5 * np.pi),
        (-1.0, -1.0, -0.5 * np.pi),
    ]

    # Act & Assert
    for pose_f in pose_f_along_circle:
        assert check_if_pose_can_be_connected_by_arc(pose_i, pose_f) == True

    for pose_f in pose_f_along_line:
        assert check_if_pose_can_be_connected_by_arc(pose_i, pose_f) == False
    for pose_f in unreachable_pose_f:
        assert check_if_pose_can_be_connected_by_arc(pose_i, pose_f) == False


def test__get_turning_radius_candidates_to_connect_pose_with_two_arcs():
    # Test left turn -> right turn with r = 1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert pytest.approx(
            1.0
        ) in _get_turning_radius_candidates_to_connect_pose_with_two_arcs(
            (0.0, 0.0, 0.0),
            (2.0 + np.cos(theta), 1.0 + np.sin(theta), theta - 0.5 * np.pi),
        )

    # Test right turn -> left turn with r = -1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert pytest.approx(
            -1.0
        ) in _get_turning_radius_candidates_to_connect_pose_with_two_arcs(
            (0.0, 0.0, 0.0),
            (2.0 + np.cos(theta), -1.0 + np.sin(theta), theta + 0.5 * np.pi),
        )

    # Test pose that can be reached by single arc
    assert pytest.approx(
        1.0
    ) in _get_turning_radius_candidates_to_connect_pose_with_two_arcs(
        (0.0, 0.0, 0.0), (1.0, 1.0, 0.5 * np.pi)
    )


def test__get_pose_connecting_arc_paths_by_radius():
    # Test left turn -> right turn with r = 1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert _get_pose_connecting_arc_paths_by_radius(
            (0.0, 0.0, 0.0),
            (2.0 + np.cos(theta), 1.0 + np.sin(theta), theta - 0.5 * np.pi),
            1.0,
        ) == pytest.approx((1.0, 1.0, 0.5 * np.pi))

    # Test right turn -> left turn with r = -1.0
    for theta in np.linspace(-np.pi, np.pi, 100):
        assert _get_pose_connecting_arc_paths_by_radius(
            (0.0, 0.0, 0.0),
            (2.0 + np.cos(theta), -1.0 + np.sin(theta), theta + 0.5 * np.pi),
            -1.0,
        ) == pytest.approx((1.0, -1.0, -0.5 * np.pi))


def test__get_arc_path_length():
    # Test along left turn circle with r = 1.0
    pose_i = (0.0, 0.0, 0.0)
    assert _get_arc_path_length(pose_i, (0.0, 0.0)) == pytest.approx(0.0)
    assert _get_arc_path_length(pose_i, (1.0, 1.0)) == pytest.approx(0.5 * np.pi)
    assert _get_arc_path_length(pose_i, (0.0, 2.0)) == pytest.approx(np.pi)
    assert _get_arc_path_length(pose_i, (-1.0, 1.0)) == pytest.approx(1.5 * np.pi)
    # Test along right turn circle with r = 1.0
    pose_i = (0.0, 0.0, 0.0)
    assert _get_arc_path_length(pose_i, (1.0, -1.0)) == pytest.approx(0.5 * np.pi)
    assert _get_arc_path_length(pose_i, (0.0, -2.0)) == pytest.approx(np.pi)
    assert _get_arc_path_length(pose_i, (-1.0, -1.0)) == pytest.approx(1.5 * np.pi)

    # Test along axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert _get_arc_path_length(pose_i, (1.0, 0.0)) == pytest.approx(1.0)
    assert _get_arc_path_length(pose_i, (2.0, 0.0)) == pytest.approx(2.0)
    assert _get_arc_path_length(pose_i, (4.0, 0.0)) == pytest.approx(4.0)
    assert _get_arc_path_length(pose_i, (-1.0, 0.0)) == np.inf
    assert _get_arc_path_length(pose_i, (-2.0, 0.0)) == np.inf
    assert _get_arc_path_length(pose_i, (-4.0, 0.0)) == np.inf


def test_get_pose_to_connect_poses_by_two_arcs():
    # Arrange
    pose_i = (0.0, 0.0, 0.0)
    min_r = 1.0

    # Act
    label_pose_stopover, label_path_length = get_pose_to_connect_poses_by_two_arcs(
        pose_i, (2.0, 2.0, 0.0), min_r
    )

    # Assert
    assert label_pose_stopover == pytest.approx((1.0, 1.0, 0.5 * np.pi))
    assert label_path_length == pytest.approx(np.pi)

    assert get_pose_to_connect_poses_by_two_arcs(pose_i, (0.0, 0.0, 0.0), min_r) == None
    assert get_pose_to_connect_poses_by_two_arcs(pose_i, (0.0, 1.0, 0.0), min_r) == None
    assert (
        get_pose_to_connect_poses_by_two_arcs(pose_i, (0.0, -1.0, 0.0), min_r) == None
    )

    # Test along the axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    pose_f = (1.0, 0.0, 0.0)
    assert get_pose_to_connect_poses_by_two_arcs(pose_i, pose_f, min_r) == None

    pose_i = (0.0, 0.0, 0.25 * np.pi)
    pose_f = (1.0, 1.0, 0.25 * np.pi)
    assert get_pose_to_connect_poses_by_two_arcs(pose_i, pose_f, min_r) == None

    # Test along the path that can be reached by single arc
    pose_i = (0.0, 0.0, 0.0)
    pose_f = (1.0, 1.0, 0.5 * np.pi)
    assert get_pose_to_connect_poses_by_two_arcs(pose_i, pose_f, min_r) == None


def test_calculate_final_yaw_of_arc_path():
    _NON_ZERO_OFFSET = 0.01
    # Test same position
    for theta in np.linspace(-np.pi + _NON_ZERO_OFFSET, np.pi, 100):
        pose_i = (0.0, 0.0, theta)
        assert calculate_final_yaw_of_arc_path(pose_i, (0.0, 0.0)) == theta

    # Test along the axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_final_yaw_of_arc_path(pose_i, (0.5, 0.0)) == 0.0
    assert calculate_final_yaw_of_arc_path(pose_i, (1.0, 0.0)) == 0.0
    assert calculate_final_yaw_of_arc_path(pose_i, (2.0, 0.0)) == 0.0

    assert calculate_final_yaw_of_arc_path(pose_i, (-0.5, 0.0)) == 0.0
    assert calculate_final_yaw_of_arc_path(pose_i, (-1.0, 0.0)) == 0.0
    assert calculate_final_yaw_of_arc_path(pose_i, (-2.0, 0.0)) == 0.0

    pose_i = (0.0, 0.0, np.pi)
    assert calculate_final_yaw_of_arc_path(pose_i, (-0.0, 0.0)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (-0.5, 0.0)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (-1.0, 0.0)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (-2.0, 0.0)) == np.pi

    # Test along the axis perpendicular to the heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_final_yaw_of_arc_path(pose_i, (0.0, 0.5)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (0.0, 1.0)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (0.0, 2.0)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (0.0, -0.5)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (0.0, -1.0)) == np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (0.0, -2.0)) == np.pi

    pose_i = (0.0, 0.0, 0.5 * np.pi)
    assert calculate_final_yaw_of_arc_path(pose_i, (0.5, 0.0)) == -0.5 * np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (1.0, 0.0)) == -0.5 * np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (2.0, 0.0)) == -0.5 * np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (-0.5, 0.0)) == -0.5 * np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (-1.0, 0.0)) == -0.5 * np.pi
    assert calculate_final_yaw_of_arc_path(pose_i, (-2.0, 0.0)) == -0.5 * np.pi


def test_calculate_angular_velocity_to_reach_pos():
    _NON_ZERO_OFFSET = 0.01
    # Test same position
    for theta in np.linspace(-np.pi + _NON_ZERO_OFFSET, np.pi, 100):
        pose_i = (0.0, 0.0, theta)
        assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, 0.0), 1.0) == 0.0

    # Test along the axis of heading direction
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.5, 0.0), 1.0) == 0.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (1.0, 0.0), 1.0) == 0.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (2.0, 0.0), 1.0) == 0.0

    assert calculate_angular_velocity_to_reach_pos(pose_i, (-0.5, 0.0), 1.0) == 0.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (-1.0, 0.0), 1.0) == 0.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (-2.0, 0.0), 1.0) == 0.0

    pose_i = (0.0, 0.0, 0.25 * np.pi)
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.5, 0.5), 1.0) == 0.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (1.0, 1.0), 1.0) == 0.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (2.0, 2.0), 1.0) == 0.0

    # Test along the axis perpendicular to the heading direction (w=v/r)
    pose_i = (0.0, 0.0, 0.0)
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, 0.5), 1.0) == 4.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, 1.0), 1.0) == 2.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, 2.0), 1.0) == 1.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, 4.0), 1.0) == 0.5

    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, -0.5), 1.0) == -4.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, -1.0), 1.0) == -2.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, -2.0), 1.0) == -1.0
    assert calculate_angular_velocity_to_reach_pos(pose_i, (0.0, -4.0), 1.0) == -0.5

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
        assert calculate_angular_velocity_to_reach_pos(
            pose_i, pos_f_left, 1.0
        ) == pytest.approx(1.0)
        assert calculate_angular_velocity_to_reach_pos(
            pose_i, pos_f_right, 1.0
        ) == pytest.approx(-1.0)

        # Test negative linear velocity
        assert calculate_angular_velocity_to_reach_pos(
            pose_i, pos_f_left, -1.0
        ) == pytest.approx(-1.0)
        assert calculate_angular_velocity_to_reach_pos(
            pose_i, pos_f_right, -1.0
        ) == pytest.approx(1.0)


def test_calculate_unicycle_w_yaw():
    assert calculate_unicycle_w_yaw((0.0, 0.0, 0.0), (1, 0), 1.0) == (0.0, 0.0)
    assert calculate_unicycle_w_yaw((0.0, 0.0, 0.0), (0, 2.0), 1.0) == (1.0, np.pi)


def test_get_straingt_path():
    d_s = 1.0

    pose_i = (0.0, 0.0, 0.0)
    pos_f = (4.0, 0.0)
    label_path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    ]
    assert get_straingt_path(pose_i, pos_f, d_s) == label_path

    pos_f = (4.1, 0.0)
    label_path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.1, 0.0, 0.0),
    ]
    assert get_straingt_path(pose_i, pos_f, d_s) == label_path

    pos_f = (4.99, 0.0)
    label_path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.99, 0.0, 0.0),
    ]
    assert get_straingt_path(pose_i, pos_f, d_s) == label_path

    pose_i = (0.0, 0.0, 0.5 * np.pi)
    pos_f = (0.0, 4.0)
    label_path = [
        (0.0, 0.0, 0.5 * np.pi),
        (0.0, 1.0, 0.5 * np.pi),
        (0.0, 2.0, 0.5 * np.pi),
        (0.0, 3.0, 0.5 * np.pi),
        (0.0, 4.0, 0.5 * np.pi),
    ]
    assert get_straingt_path(pose_i, pos_f, d_s) == label_path

    # Test edge cases: pos_f is in the opposite direction of the heading direction
    pose_i = (0.0, 0.0, 0.0)
    pos_f = (-1.0, 0.0)
    with pytest.raises(ValueError):
        get_straingt_path(pose_i, pos_f, d_s)
    pos_f = (-0.1, 0.0)
    with pytest.raises(ValueError):
        get_straingt_path(pose_i, pos_f, d_s)

    # Test edge cases: pos_f is in the same position as pose_i
    pose_i = (0.0, 0.0, 0.0)
    pos_f = (0.0, 0.0)
    assert get_straingt_path(pose_i, pos_f, d_s) == [pose_i]
    pose_i = (0.0, 0.0, 0.5 * np.pi)
    pos_f = (0.0, 0.0)
    assert get_straingt_path(pose_i, pos_f, d_s) == [pose_i]


def test_get_unicycle_path():
    # ==================== Test arc with r = 1.0 ====================
    d_s = 0.25 * np.pi

    pose_i = (0.0, 0.0, 0.0)
    pos_f = (1.0, 1.0)
    label_path = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.5 * np.sqrt(2.0), 1.0 - 0.5 * np.sqrt(2.0), 0.25 * np.pi),
            (1.0, 1.0, 0.5 * np.pi),
        ]
    )
    answer_path = np.array(get_unicycle_path(pose_i, pos_f, d_s))
    assert answer_path == pytest.approx(label_path)

    # Test that last pos on path is guaranteed to be pos_f
    pos_f = (np.cos(0.1 * np.pi), 1.0 + np.sin(0.1 * np.pi))
    label_path = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.5 * np.sqrt(2.0), 1.0 - 0.5 * np.sqrt(2.0), 0.25 * np.pi),
            (1.0, 1.0, 0.5 * np.pi),
            (np.cos(0.1 * np.pi), 1.0 + np.sin(0.1 * np.pi), 0.6 * np.pi),
        ]
    )
    answer_path = np.array(get_unicycle_path(pose_i, pos_f, d_s))
    assert answer_path == pytest.approx(label_path)

    pos_f = (np.cos(1.4 * np.pi), 1.0 + np.sin(1.4 * np.pi))
    label_path = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.5 * np.sqrt(2.0), 1.0 - 0.5 * np.sqrt(2.0), 0.25 * np.pi),
            (1.0, 1.0, 0.5 * np.pi),
            (0.5 * np.sqrt(2.0), 1.0 + 0.5 * np.sqrt(2.0), 0.75 * np.pi),
            (0.0, 2.0, 1.0 * np.pi),
            (-0.5 * np.sqrt(2.0), 1.0 + 0.5 * np.sqrt(2.0), -0.75 * np.pi),
            (-1.0, 1.0, -0.5 * np.pi),
            (-0.5 * np.sqrt(2.0), 1.0 - 0.5 * np.sqrt(2.0), -0.25 * np.pi),
            (np.cos(1.4 * np.pi), 1.0 + np.sin(1.4 * np.pi), -0.1 * np.pi),
        ]
    )
    label_path_pi_to_minus_pi = np.where(label_path == np.pi, -np.pi, label_path)

    answer_path = np.array(get_unicycle_path(pose_i, pos_f, d_s))
    assert answer_path == pytest.approx(label_path) or answer_path == pytest.approx(
        label_path_pi_to_minus_pi
    )

    pose_i = (0.0, 0.0, np.pi)
    pos_f = (-1.0, -1.0)
    label_path = np.array(
        [
            (0.0, 0.0, np.pi),
            (-0.5 * np.sqrt(2.0), -1.0 + 0.5 * np.sqrt(2.0), -0.75 * np.pi),
            (-1.0, -1.0, -0.5 * np.pi),
        ]
    )
    answer_path = np.array(get_unicycle_path(pose_i, pos_f, d_s))
    assert answer_path == pytest.approx(label_path)

    # ==================== Test arc with r = -1.0 ====================
    d_s = 0.25 * np.pi

    pose_i = (0.0, 0.0, 0.0)
    pos_f = (1.0, -1.0)
    label_path = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.5 * np.sqrt(2.0), -1.0 + 0.5 * np.sqrt(2.0), -0.25 * np.pi),
            (1.0, -1.0, -0.5 * np.pi),
        ]
    )
    answer_path = np.array(get_unicycle_path(pose_i, pos_f, d_s))
    assert answer_path == pytest.approx(label_path)

    pose_i = (0.0, 0.0, np.pi)
    pos_f = (-1.0, 1.0)
    label_path = np.array(
        [
            (0.0, 0.0, np.pi),
            (-0.5 * np.sqrt(2.0), 1.0 - 0.5 * np.sqrt(2.0), 0.75 * np.pi),
            (-1.0, 1.0, 0.5 * np.pi),
        ]
    )
    answer_path = np.array(get_unicycle_path(pose_i, pos_f, d_s))
    assert answer_path == pytest.approx(label_path)

    # ==================== Test straight path ====================
    d_s = 1.0

    pose_i = (0.0, 0.0, 0.0)
    pos_f = (4.0, 0.0)
    label_path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
    ]
    assert get_unicycle_path(pose_i, pos_f, d_s) == label_path

    pos_f = (4.1, 0.0)
    label_path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.1, 0.0, 0.0),
    ]
    assert get_unicycle_path(pose_i, pos_f, d_s) == label_path

    pos_f = (4.99, 0.0)
    label_path = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.99, 0.0, 0.0),
    ]
    assert get_unicycle_path(pose_i, pos_f, d_s) == label_path

    pose_i = (0.0, 0.0, 0.5 * np.pi)
    pos_f = (0.0, 4.0)
    label_path = [
        (0.0, 0.0, 0.5 * np.pi),
        (0.0, 1.0, 0.5 * np.pi),
        (0.0, 2.0, 0.5 * np.pi),
        (0.0, 3.0, 0.5 * np.pi),
        (0.0, 4.0, 0.5 * np.pi),
    ]
    assert get_unicycle_path(pose_i, pos_f, d_s) == label_path

    # Test edge cases: pos_f is in the opposite direction of the heading direction
    pose_i = (0.0, 0.0, 0.0)
    pos_f = (-1.0, 0.0)
    with pytest.raises(ValueError):
        get_unicycle_path(pose_i, pos_f, d_s)
    pos_f = (-0.1, 0.0)
    with pytest.raises(ValueError):
        get_unicycle_path(pose_i, pos_f, d_s)

    # Test edge cases: pos_f is in the same position as pose_i
    pose_i = (0.0, 0.0, 0.0)
    pos_f = (0.0, 0.0)
    assert get_unicycle_path(pose_i, pos_f, d_s) == [pose_i]
    pose_i = (0.0, 0.0, 0.5 * np.pi)
    pos_f = (0.0, 0.0)
    assert get_unicycle_path(pose_i, pos_f, d_s) == [pose_i]
