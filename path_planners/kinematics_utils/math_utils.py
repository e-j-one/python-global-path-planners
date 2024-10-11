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
