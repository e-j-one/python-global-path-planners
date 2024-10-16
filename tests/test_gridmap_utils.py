import os
import warnings

import numpy as np
import pytest

from path_planners.utils.gridmap_utils import (
    load_yaml_config,
    load_occupancy_map,
    load_occupancy_map_by_config_path,
    pos_to_grid_cell_idx,
    check_if_pos_inside_map,
    check_if_pos_is_free,
    check_path_segments_are_not_larger_than_threshold,
    check_collision_for_path,
    get_padded_occupancy_map,
)


def test_load_yaml_config():
    # Arrange
    label_map_config = {
        "image": "test_occupancy_map_0.png",
        "resolution": 1.0,
        "origin": [-0.5, -0.5, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    yaml_file_path = os.path.join(
        os.path.dirname(__file__), "../occupancy_maps/test_occupancy_map_0.yaml"
    )

    # Act
    answer_config = load_yaml_config(yaml_file_path)

    # Assert
    assert answer_config == label_map_config


def test_load_occupancy_map():
    # Arrange
    map_config = {
        "image": "test_occupancy_map_0.png",
        "resolution": 1.0,
        "origin": [-0.5, -0.5, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    label_occupancy_map = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
        ]
    )

    # Act
    answer_occupancy_map, answer_resolution, answer_origin = load_occupancy_map(
        map_config
    )

    # Assert
    assert np.array_equal(answer_occupancy_map, label_occupancy_map)
    assert answer_resolution == 1.0
    assert answer_origin == [-0.5, -0.5, 0.0]

    label_occupancy_map = np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    # Act
    answer_occupancy_map, answer_resolution, answer_origin = load_occupancy_map(
        map_config, flip_y_axis=True
    )

    # Assert
    assert np.array_equal(answer_occupancy_map, label_occupancy_map)
    assert answer_resolution == 1.0
    assert answer_origin == [-0.5, -0.5, 0.0]


def test_load_occupancy_map_by_config_path():
    # Arrange
    yaml_file_path = os.path.join(
        os.path.dirname(__file__), "../occupancy_maps/test_occupancy_map_0.yaml"
    )

    label_occupancy_map = np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    # Act
    answer_occupancy_map, answer_resolution, answer_origin = (
        load_occupancy_map_by_config_path(yaml_file_path, flip_y_axis=True)
    )

    # Assert
    assert np.array_equal(answer_occupancy_map, label_occupancy_map)
    assert answer_resolution == 1.0
    assert answer_origin == [-0.5, -0.5, 0.0]


def test_pos_to_grid_cell_idx():
    # Arrange
    resolution = 1.0
    origin = [-0.5, -0.5, 0.0]

    # Act and Assert
    assert pos_to_grid_cell_idx((0.0, 0.0), resolution, origin) == (0, 0)
    assert pos_to_grid_cell_idx((1.0, 0.0), resolution, origin) == (0, 1)
    assert pos_to_grid_cell_idx((2.0, 0.0), resolution, origin) == (0, 2)
    assert pos_to_grid_cell_idx((0.0, 1.0), resolution, origin) == (1, 0)
    assert pos_to_grid_cell_idx((0.0, 2.0), resolution, origin) == (2, 0)
    assert pos_to_grid_cell_idx((0.0, 3.0), resolution, origin) == (3, 0)

    assert pos_to_grid_cell_idx((-0.5, -0.5), resolution, origin) == (0, 0)
    assert pos_to_grid_cell_idx((2.5 - 1e-6, 3.5 - 1e-6), resolution, origin) == (3, 2)

    # Arrange
    resolution = 1.0
    origin = [9.5, -0.5, 0.0]

    # Act and Assert
    assert pos_to_grid_cell_idx((10.0, 0.0), resolution, origin) == (0, 0)
    assert pos_to_grid_cell_idx((11.0, 0.0), resolution, origin) == (0, 1)
    assert pos_to_grid_cell_idx((12.0, 0.0), resolution, origin) == (0, 2)
    assert pos_to_grid_cell_idx((10.0, 1.0), resolution, origin) == (1, 0)
    assert pos_to_grid_cell_idx((10.0, 2.0), resolution, origin) == (2, 0)
    assert pos_to_grid_cell_idx((10.0, 3.0), resolution, origin) == (3, 0)

    assert pos_to_grid_cell_idx((9.5, -0.5), resolution, origin) == (0, 0)
    assert pos_to_grid_cell_idx((12.5 - 1e-6, 3.5 - 1e-6), resolution, origin) == (3, 2)


def test_check_if_pos_inside_map():
    # Arrange
    x_min = -0.5
    y_min = -0.5
    x_max = 2.5
    y_max = 3.5

    inbound_poses = [
        (-0.5, -0.5),
        (0.0, 0.0),
        (2.0, 3.0),
    ]
    outbound_poses = [
        (-0.5, 3.5),
        (2.5, 3.5),
    ]

    # Act and Assert
    for target_pos in inbound_poses:
        assert check_if_pos_inside_map(x_min, y_min, x_max, y_max, target_pos) == True

    for target_pos in outbound_poses:
        assert check_if_pos_inside_map(x_min, y_min, x_max, y_max, target_pos) == False


def test_check_if_pos_is_free():
    # Arrange
    occupancy_map = np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    resolution = 1.0
    origin = [-0.5, -0.5, 0.0]

    free_poses = [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 2.0),
        (0.0, 3.0),
        (1.0, 3.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (2.0, 3.0),
        (-0.5, -0.5),
        (0.499, 0.499),
        (2.499, 3.499),
    ]
    occupied_poses = [
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (0.5, 0.0),
    ]

    # Act and Assert
    for target_pos in free_poses:
        assert (
            check_if_pos_is_free(occupancy_map, resolution, origin, target_pos) == True
        )

    for target_pos in occupied_poses:
        assert (
            check_if_pos_is_free(occupancy_map, resolution, origin, target_pos) == False
        )


def testcheck_path_segments_are_not_larger_than_threshold():
    # Arrange
    threshold = 1.0
    valid_paths = [
        [[0.0, 0.0, 0.0]],
        [[2.0, 2.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, -np.pi]],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ]
    invalid_paths = [
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.1, 0.0]],
        [[0.0, 0.0, 0.0], [0.8, 0.8, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ]

    # Act and Assert
    for target_path in valid_paths:
        assert (
            check_path_segments_are_not_larger_than_threshold(target_path, threshold)
            == False
        )

    for target_path in invalid_paths:
        assert (
            check_path_segments_are_not_larger_than_threshold(target_path, threshold)
            == True
        )


def test_check_collision_for_path():
    # Arrange
    occupancy_map = np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    resolution = 1.0
    origin = [-0.5, -0.5, 0.0]

    collision_free_paths = [
        [[0.0, 0.0, 0.0]],
        [
            [-0.5, -0.5, 0.25 * np.pi],
            [0.0, 0.0, 0.25 * np.pi],
            [0.49, 0.49, 0.25 * np.pi],
        ],
        [[0.0, 0.0, 0.5 * np.pi], [0.0, 0.5, 0.5 * np.pi], [0.0, 1.0, 0.5 * np.pi]],
        [
            [0.0, 0.0, 0.5 * np.pi],
            [0.0, 1.0, 0.5 * np.pi],
            [0.0, 2.0, 0.5 * np.pi],
            [0.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 3.0, -0.5 * np.pi],
            [2.0, 2.0, -0.5 * np.pi],
            [2.0, 1.0, -0.5 * np.pi],
            [2.0, 0.0, -0.5 * np.pi],
        ],
    ]
    collision_paths = [
        [[-1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[2.5, 0.0, 0.0]],
        [[3.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0]],
        [[0.0, 4.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    sparse_collision_free_paths = [
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]

    sparse_collision_paths = [
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 3.0, 0.0]],
        [[1.0, 3.0, 0.0], [1.0, 0.0, 0.0]],
    ]

    # Act and Assert
    for target_path in collision_free_paths:
        assert (
            check_collision_for_path(occupancy_map, resolution, origin, target_path)
            == False
        )

    for target_path in collision_paths:
        assert (
            check_collision_for_path(occupancy_map, resolution, origin, target_path)
            == True
        )

    with pytest.warns(UserWarning):
        for target_path in sparse_collision_free_paths:
            assert (
                check_collision_for_path(occupancy_map, resolution, origin, target_path)
                == False
            )
        for target_path in sparse_collision_paths:
            assert (
                check_collision_for_path(occupancy_map, resolution, origin, target_path)
                == True
            )


def test_get_padded_occupancy_map():
    # Arrange
    resolution = 0.1
    occupancy_map = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 100, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    label_padded_occupancy_map_0d1 = np.array(
        [
            [0.0, 100, 0.0, 0.0, 0.0],
            [100, 100, 100, 0.0, -1.0],
            [0.0, 100, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    label_padded_occupancy_map_0d2 = np.array(
        [
            [100, 100, 100, 0.0, 0.0],
            [100, 100, 100, 100, -1.0],
            [100, 100, 100, 0.0, 0.0],
            [0.0, 100, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    label_padded_occupancy_map_0d3 = np.array(
        [
            [100, 100, 100, 100, 0.0],
            [100, 100, 100, 100, 100],
            [100, 100, 100, 100, 0.0],
            [100, 100, 100, 100, 0.0],
            [0.0, 100, 0.0, 0.0, 0.0],
        ]
    )

    # Act
    answer_padded_occupancy_map_0d1 = get_padded_occupancy_map(
        occupancy_map, 0.1, resolution
    )
    answer_padded_occupancy_map_0d2 = get_padded_occupancy_map(
        occupancy_map, 0.2, resolution
    )
    answer_padded_occupancy_map_0d3 = get_padded_occupancy_map(
        occupancy_map, 0.3, resolution
    )

    # Assert
    assert np.array_equal(
        answer_padded_occupancy_map_0d1, label_padded_occupancy_map_0d1
    )
    assert np.array_equal(
        answer_padded_occupancy_map_0d2, label_padded_occupancy_map_0d2
    )
    assert np.array_equal(
        answer_padded_occupancy_map_0d3, label_padded_occupancy_map_0d3
    )
