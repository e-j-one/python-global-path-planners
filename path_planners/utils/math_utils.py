import numpy as np


def normalize_angle(angle: float) -> float:
    """
    Normalize the angle to (-pi, pi]
    """
    if angle > -np.pi and angle <= np.pi:
        return angle

    normalize_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    if normalize_angle == -np.pi:
        return np.pi
    return normalize_angle


def normalize_angle_positive(angle: float) -> float:
    """
    Normalize the angle to [0, 2pi)
    """
    if angle >= 0 and angle < 2 * np.pi:
        return angle
    # print("angle", angle)
    normalize_angle_positive = angle % (2 * np.pi)
    # print("normalize_angle_positive", normalize_angle_positive)
    return normalize_angle_positive


def check_if_angle_diff_is_within_threshold(
    angle_i: float, angle_f: float, threshold: float
) -> bool:
    """
    Check if the angle_i is near the angle_f within the threshold (abs(angle_i - angle_f) < threshold)
    """
    return abs(normalize_angle(angle_i - angle_f)) < threshold
