import numpy as np

from path_planners.utils.math_utils import (
    calculate_dist,
    normalize_angle,
    normalize_angle_positive,
)


def test_calculate_dist():
    assert calculate_dist((0.0, 0.0), (0.0, 0.0)) == 0.0
    assert calculate_dist((0.0, 0.0), (1.0, 0.0)) == 1.0
    assert calculate_dist((0.0, 0.0), (0.0, 1.0)) == 1.0
    assert calculate_dist((0.0, 0.0), (1.0, 1.0)) == np.sqrt(2.0)


def test_normalize_angle():
    # Test angles within the range (-pi, pi]
    assert normalize_angle(0.0) == 0.0
    assert normalize_angle(0.5 * np.pi) == 0.5 * np.pi
    assert normalize_angle(-0.5 * np.pi) == -0.5 * np.pi
    assert normalize_angle(np.pi) == np.pi

    # Test angles outside the range (-pi, pi]
    assert normalize_angle(3.0 * np.pi) == np.pi
    assert normalize_angle(-3.0 * np.pi) == np.pi
    assert normalize_angle(4.0 * np.pi) == 0.0
    assert normalize_angle(-4.0 * np.pi) == 0.0

    # Test edge cases
    assert normalize_angle(-np.pi) == np.pi
    assert normalize_angle(2.0 * np.pi) == 0.0
    assert normalize_angle(-2.0 * np.pi) == 0.0


def test_normalize_angle_positive():
    # Test angles within the range [0, 2 * pi)
    for angle in np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False):
        assert normalize_angle_positive(angle) == angle

    # Test angles outside the range [0, 2 * pi)
    for angle in np.linspace(-2.0 * np.pi, 0.0, 100, endpoint=False):
        assert normalize_angle_positive(angle) == angle + 2.0 * np.pi

    for angle in np.linspace(2.0 * np.pi, 4.0 * np.pi, 100, endpoint=False):
        assert normalize_angle_positive(angle) == angle - 2.0 * np.pi
    # Test edge cases
    assert normalize_angle(2.0 * np.pi) == 0.0
    assert normalize_angle(-2.0 * np.pi) == 0.0
