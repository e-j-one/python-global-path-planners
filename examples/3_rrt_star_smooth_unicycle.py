import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from path_planners.utils.gridmap_utils import load_occupancy_map_by_config_path
from path_planners.utils.plot_utils import plot_occupancy_grid
from path_planners.rrt_star_smooth_unicycle import RrtStarSmoothUnicyclePlanner


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
    # plot_occupancy_grid(occupancy_map, resolution, origin)

    rrt_unicycle_config = {
        "terminate_on_goal_reached": False,
        "goal_reach_dist_threshold": 0.5,
        "goal_reach_angle_threshold": 0.1 * np.pi,
        "occupancy_map_obstacle_padding_dist": 0.5,
        "goal_sample_rate": 0.1,
        "max_iter": int(8000),
        "max_drive_dist": 0.5,
        "linear_velocity": 1.0,
        "max_angular_velocity": 4.0,
        "near_node_dist_threshold": 1.0,
    }

    # start_pose = (0.0, 0.0, 0.0)
    # goal_pose = (1.0, 10.0, 0.0)

    start_pose = (-40.0, 5.0, 0.0)
    goal_pose = (15.0, 0.0, 0.0)

    rrt_unicycle_path_planner = RrtStarSmoothUnicyclePlanner(
        **rrt_unicycle_config, render_tree_during_planning=True
    )

    rrt_unicycle_path_planner.set_occupancy_map(
        occupancy_map,
        occupancy_map_resolution,
        occupancy_map_origin,
    )
    rrt_unicycle_path_planner.plan_global_path(start_pose, goal_pose, render=True)
