import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from path_planners.utils.gridmap_utils import load_occupancy_map_by_config_path
from path_planners.utils.plot_utils import plot_occupancy_grid
from path_planners.a_star_bending import AStarBendingPlanner


# fix seed
np.random.seed(1)

if __name__ == "__main__":
    map_name = "World0"  # "World0", "test_occupancy_map_0"
    yaml_file_path = os.path.join(
        os.path.dirname(__file__), "../occupancy_maps/" + map_name + ".yaml"
    )
    occupancy_map, occupancy_map_resolution, occupancy_map_origin = (
        load_occupancy_map_by_config_path(yaml_file_path, flip_y_axis=True)
    )

    print("occupancy_map.shape", occupancy_map.shape)
    # plot_occupancy_grid(occupancy_map, occupancy_map_resolution, occupancy_map_origin)

    config = {"occupancy_map_obstacle_padding_dist": 0.0, "print_log": True}

    # start_pose = (1.0, 1.0, 0.0)
    # goal_pose = (4.0, 0.0, 0.0)

    # start_pose = (0.0, 0.0, 0.0)
    # goal_pose = (2.0, 1.0, 0.0)

    start_pose = (-40.0, 5.0, 0.0)
    goal_pose = (15.0, 0.0, 0.0)

    path_planner = AStarBendingPlanner(**config)
    path_planner.set_occupancy_map(
        occupancy_map,
        occupancy_map_resolution,
        occupancy_map_origin,
    )

    success, path, num_iters = path_planner.plan_global_path(
        start_pose,
        goal_pose,
        render=True,
        save_plot_to_file=True,
    )

    print("occupancy_map[0]", occupancy_map[0])
    print("occupancy_map", occupancy_map.shape)
    print("success", success)
    print("path", path)
