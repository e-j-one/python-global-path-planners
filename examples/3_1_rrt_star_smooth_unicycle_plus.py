import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from path_planners.utils.gridmap_utils import load_occupancy_map_by_config_path
from path_planners.utils.plot_utils import plot_occupancy_grid
from path_planners.rrt_star_smooth_unicycle_plus import RrtStarSmoothUnicyclePlusPlanner


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
        "interpolate_path": True,
        "d_s": 0.5,
        "goal_reach_dist_threshold": 0.5,
        "goal_reach_angle_threshold": 0.1 * np.pi,
        "occupancy_map_obstacle_padding_dist": 0.5,
        "print_log": True,
        "goal_sample_rate": 0.1,
        "max_iter": int(20000),
        "max_drive_dist": 0.5,
        "linear_velocity": 1.0,
        "max_angular_velocity": 2.0,
        "near_node_dist_threshold": 1.0,
    }

    # start_pose = (0.0, 0.0, 0.0)
    # goal_pose = (1.0, 10.0, 0.0)

    start_pose = (-40.0, 5.0, 0.0)
    goal_pose = (15.0, 0.0, 0.0)

    # start_pose = (-12.0, 15.0, 0.0)
    # goal_pose = (15.0, 0.0, 0.0)

    rrt_unicycle_path_planner = RrtStarSmoothUnicyclePlusPlanner(
        **rrt_unicycle_config, render_tree_during_planning=False
    )

    rrt_unicycle_path_planner.set_occupancy_map(
        occupancy_map,
        occupancy_map_resolution,
        occupancy_map_origin,
    )

    success, path, num_nodes_sampled = rrt_unicycle_path_planner.plan_global_path(
        start_pose, goal_pose, render=True, save_plot_to_file=True
    )

    print("Number of nodes sampled:", num_nodes_sampled)
    if not success:
        print("Path not found!")
        exit()

    print("Path found!")
    print("Path length:", len(path))

    # save path to file
    curr_date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_file_name = (
        f"results/rrt_star_smooth_unicycle_plus_{map_name}_{curr_date_time_str}.csv"
    )
    path = np.array(path)

    # if directory does not exist, create it
    if not os.path.exists("results"):
        os.makedirs("results")

    np.savetxt(path_file_name, path, delimiter=",")
    print(f"Path saved to {path_file_name}")

    # save start, goal and config to file
    config_file_name = f"results/rrt_star_smooth_unicycle_plus_{map_name}_{curr_date_time_str}_config.txt"
    with open(config_file_name, "w") as f:
        f.write(f"Start pose: {start_pose}\n")
        f.write(f"Goal pose: {goal_pose}\n")
        f.write(f"Config: {rrt_unicycle_config}\n")
