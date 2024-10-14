import numpy as np
import pytest

from path_planners.utils.geometry_utils import (
    calculate_delta_i_f,
    calculate_arc_path_radius,
    check_if_dist_is_below_threshold,
    check_if_pos_in_same_side_of_heading,
)


def test_normalize_angle():

    offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    label_delta_for_0_theta = [
        0,
        0.25 * np.pi,
        0.5 * np.pi,
        0.75 * np.pi,
        np.pi,
        -0.75 * np.pi,
        -0.5 * np.pi,
        -0.25 * np.pi,
    ]
    label_delta_for_pi_theta = [
        np.pi,
        -0.75 * np.pi,
        -0.5 * np.pi,
        -0.25 * np.pi,
        0,
        0.25 * np.pi,
        0.5 * np.pi,
        0.75 * np.pi,
    ]

    pose_i_0_theta = (0, 0, 0)
    for offset, label_delta in zip(offsets, label_delta_for_0_theta):
        pos_f = (pose_i_0_theta[0] + offset[0], pose_i_0_theta[1] + offset[1])
        assert calculate_delta_i_f(pose_i_0_theta, pos_f) == pytest.approx(label_delta)

    pose_i_pi_theta = (0, 0, np.pi)
    for offset, label_delta in zip(offsets, label_delta_for_pi_theta):
        pos_f = (pose_i_pi_theta[0] + offset[0], pose_i_pi_theta[1] + offset[1])
        assert calculate_delta_i_f(pose_i_pi_theta, pos_f) == pytest.approx(label_delta)

    # Test overlapping poses.
    poses = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    thetas = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, -0.5 * np.pi]

    for theta in thetas:
        for pos in poses:
            pose_i = (*pos, theta)
            pos_f = pos
            assert calculate_delta_i_f(pose_i, pos_f) == pytest.approx(0.0)


def test_calculate_arc_path_radius():
    _NON_ZERO_OFFSET = 1e-3
    # Test along the circle with radius r.
    pose_i = (0, 0, 0)
    radius_candidates = [0.5, 1.0, 2.0]

    # r > 0
    for r in radius_candidates:
        for theta in np.linspace(
            -0.5 * np.pi + _NON_ZERO_OFFSET, 1.5 * np.pi - _NON_ZERO_OFFSET, 20
        ):
            pos_f = (r * np.cos(theta), r + r * np.sin(theta))
            assert calculate_arc_path_radius(pose_i, pos_f) == pytest.approx(r)

    # r < 0
    for r in radius_candidates:
        for theta in np.linspace(
            0.5 * np.pi + _NON_ZERO_OFFSET, 2.5 * np.pi - _NON_ZERO_OFFSET, 20
        ):
            pos_f = (r * np.cos(theta), -r + r * np.sin(theta))
            assert calculate_arc_path_radius(pose_i, pos_f) == pytest.approx(-r)

    # Test along the straight line.
    pose_i = (0, 0, 0)
    pos_f_candidates = [(-1.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    for pos_f in pos_f_candidates:
        assert calculate_arc_path_radius(pose_i, pos_f) == np.inf

    # Test at same pos
    pose_i_candidates = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, np.pi),
        (1.0, 1.0, np.pi),
        (-1.0, -1.0, -0.5 * np.pi),
    ]
    for pose_i in pose_i_candidates:
        pos_f = pose_i[:2]
        assert calculate_arc_path_radius(pose_i, pos_f) == 0.0


def test_check_if_dist_is_below_threshold():
    assert check_if_dist_is_below_threshold((0.0, 0.0, 0.0), (0.0, 0.0), 1e-6) == True
    assert check_if_dist_is_below_threshold((0.0, 0.0, 0.0), (0.0, 0.01), 1e-6) == False
    assert check_if_dist_is_below_threshold((0.0, 0.0, 0.0), (0.0, 1e-6), 1e-6) == False


def test_check_if_pos_in_same_side_of_heading():
    pose_i = (0, 0, 0)
    assert check_if_pos_in_same_side_of_heading(pose_i, (1.0, 0.0)) == True
    assert check_if_pos_in_same_side_of_heading(pose_i, (1.0, 1.0)) == True

    assert check_if_pos_in_same_side_of_heading(pose_i, (0.0, 1.0)) == False
    assert check_if_pos_in_same_side_of_heading(pose_i, (0.0, -1.0)) == False

    assert check_if_pos_in_same_side_of_heading(pose_i, (-1.0, 0)) == False
    assert check_if_pos_in_same_side_of_heading(pose_i, (-1.0, 1.0)) == False

    pose_i = (0, 0, np.pi)
    assert check_if_pos_in_same_side_of_heading(pose_i, (1.0, 0.0)) == False
    assert check_if_pos_in_same_side_of_heading(pose_i, (1.0, 1.0)) == False

    assert check_if_pos_in_same_side_of_heading(pose_i, (0.0, 1.0)) == False
    assert check_if_pos_in_same_side_of_heading(pose_i, (0.0, -1.0)) == False

    assert check_if_pos_in_same_side_of_heading(pose_i, (-1.0, 0)) == True
    assert check_if_pos_in_same_side_of_heading(pose_i, (-1.0, 1.0)) == True
