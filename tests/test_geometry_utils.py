import numpy as np
import pytest

from path_planners.kinematics_utils.geometry_utils import calculate_delta_i_f


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
        assert pytest.approx(calculate_delta_i_f(pose_i_0_theta, pos_f)) == label_delta

    pose_i_pi_theta = (0, 0, np.pi)
    for offset, label_delta in zip(offsets, label_delta_for_pi_theta):
        pos_f = (pose_i_pi_theta[0] + offset[0], pose_i_pi_theta[1] + offset[1])
        assert pytest.approx(calculate_delta_i_f(pose_i_pi_theta, pos_f)) == label_delta

    # Test overlapping poses.
    poses = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    thetas = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, -0.5 * np.pi]

    for theta in thetas:
        for pos in poses:
            pose_i = (*pos, theta)
            pos_f = pos
            assert pytest.approx(calculate_delta_i_f(pose_i, pos_f)) == 0.0
