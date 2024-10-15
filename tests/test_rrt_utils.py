import numpy as np
import pytest

from path_planners.rrt_utils.rrt_utils import drive_pos


def test_drive_pos():
    node_pos = (0.0, 0.0)
    assert drive_pos(node_pos, (0.5, 0.0), 2.0) == (0.5, 0.0)
    assert drive_pos(node_pos, (1.0, 0.0), 2.0) == (1.0, 0.0)
    assert drive_pos(node_pos, (2.0, 0.0), 2.0) == (2.0, 0.0)
    assert drive_pos(node_pos, (4.0, 0.0), 2.0) == (2.0, 0.0)

    assert drive_pos(node_pos, (-0.5, 0.0), 2.0) == (-0.5, 0.0)
    assert drive_pos(node_pos, (-1.0, 0.0), 2.0) == (-1.0, 0.0)
    assert drive_pos(node_pos, (-2.0, 0.0), 2.0) == (-2.0, 0.0)
    assert drive_pos(node_pos, (-4.0, 0.0), 2.0) == (-2.0, 0.0)

    assert drive_pos(node_pos, (0.5, 1.0), np.sqrt(5)) == (0.5, 1.0)
    assert drive_pos(node_pos, (1.0, 2.0), np.sqrt(5)) == (1.0, 2.0)
    assert drive_pos(node_pos, (2.0, 4.0), np.sqrt(5)) == (1.0, 2.0)
    assert drive_pos(node_pos, (4.0, 8.0), np.sqrt(5)) == (1.0, 2.0)

    assert drive_pos(node_pos, (0.0, 0.5), 2.0) == (0.0, 0.5)
    assert drive_pos(node_pos, (0.0, 1.0), 2.0) == (0.0, 1.0)
    assert drive_pos(node_pos, (0.0, 2.0), 2.0) == (0.0, 2.0)
    assert drive_pos(node_pos, (0.0, 4.0), 2.0) == (0.0, 2.0)

    node_pos = (2.0, 1.0)
    assert drive_pos(node_pos, (2.5, 1.0), 2.0) == (2.5, 1.0)
    assert drive_pos(node_pos, (3.0, 1.0), 2.0) == (3.0, 1.0)
    assert drive_pos(node_pos, (4.0, 1.0), 2.0) == (4.0, 1.0)
    assert drive_pos(node_pos, (6.0, 1.0), 2.0) == (4.0, 1.0)
