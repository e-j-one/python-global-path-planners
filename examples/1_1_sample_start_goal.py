import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from path_planners.utils.gridmap_utils import load_occupancy_map_by_config_path
from path_planners.utils.plot_utils import plot_occupancy_grid, plot_global_path
from path_planners.rrt_star_smooth_unicycle import RrtStarSmoothUnicyclePlanner


# fix seed
np.random.seed()

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
        "interpolate_path": True,
        "d_s": 0.5,
        "goal_reach_dist_threshold": 0.5,
        "goal_reach_angle_threshold": 0.1 * np.pi,
        "occupancy_map_obstacle_padding_dist": 0.5,
        "goal_sample_rate": 0.1,
        "max_iter": int(14000),
        "max_drive_dist": 0.5,
        "linear_velocity": 1.0,
        "max_angular_velocity": 2.0,
        "near_node_dist_threshold": 1.0,
    }

    rrt_unicycle_path_planner = RrtStarSmoothUnicyclePlanner(
        **rrt_unicycle_config, render_tree_during_planning=False
    )

    rrt_unicycle_path_planner.set_occupancy_map(
        occupancy_map,
        occupancy_map_resolution,
        occupancy_map_origin,
    )

    start_pose, goal_pose = rrt_unicycle_path_planner.sample_start_goal_poses(
        min_dist=5.0
    )

    print("start_pose", start_pose)
    print("goal_pose", goal_pose)

    plot_global_path(
        occupancy_map,
        occupancy_map_resolution,
        occupancy_map_origin,
        start_pose,
        goal_pose,
        [],
    )
