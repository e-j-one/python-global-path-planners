import numpy as np

from path_planners.kinematics_utils.math_utils import normalize_angle


def test_normalize_angle():
    # Test angles within the range (-pi, pi]
    assert normalize_angle(0) == 0
    assert normalize_angle(np.pi / 2) == np.pi / 2
    assert normalize_angle(-np.pi / 2) == -np.pi / 2
    assert normalize_angle(np.pi) == np.pi

    # Test angles outside the range (-pi, pi]
    assert normalize_angle(3 * np.pi) == np.pi
    assert normalize_angle(-3 * np.pi) == np.pi
    assert normalize_angle(4 * np.pi) == 0
    assert normalize_angle(-4 * np.pi) == 0

    # Test edge cases
    assert normalize_angle(-np.pi) == np.pi
    assert normalize_angle(2 * np.pi) == 0
    assert normalize_angle(-2 * np.pi) == 0
