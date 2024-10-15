import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from path_planners.utils.gridmap_utils import load_occupancy_map_by_config_path
from path_planners.utils.plot_utils import plot_occupancy_grid
from path_planners.rrt_unicycle import RrtUnicyclePlanner


# fix seed
np.random.seed(0)

if __name__ == "__main__":
    yaml_file_path = os.path.join(
        os.path.dirname(__file__), "../occupancy_maps/World0.yaml"
    )
    occupancy_map, occupancy_map_resolution, occupancy_map_origin = (
        load_occupancy_map_by_config_path(yaml_file_path, flip_y_axis=True)
    )

    print("occupancy_map.shape", occupancy_map.shape)
    # plot_occupancy_grid(occupancy_map, resolution, origin)

    rrt_unicycle_config = {
        "goal_reach_dist_threshold": 0.1,
        "goal_reach_angle_threshold": 0.1 * np.pi,
        "goal_sample_rate": 0.1,
        "max_iter": 20000,
        "max_drive_dist": 0.5,
        "linear_velocity": 1.0,
        "max_angular_velocity": 4.0,
    }

    start_pose = (0.0, 0.0, 0.0)
    goal_pose = (1.0, 10.0, 0.0)

    rrt_unicycle_path_planner = RrtUnicyclePlanner(**rrt_unicycle_config)

    rrt_unicycle_path_planner.set_occupancy_map(
        occupancy_map,
        occupancy_map_resolution,
        occupancy_map_origin,
    )
    rrt_unicycle_path_planner.plan_global_path(start_pose, goal_pose, render=True)
