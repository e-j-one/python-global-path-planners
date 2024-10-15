from typing import List, Tuple

import numpy as np


def print_path(path: List[Tuple[float, float, float]]) -> None:
    for i, pose in enumerate(path):
        # print path till second decimal point
        print(f"\t{i}: {pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}")
