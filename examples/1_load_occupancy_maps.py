import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from path_planners.utils.gridmap_utils import load_yaml_config, load_occupancy_map
from path_planners.utils.plot_utils import plot_occupancy_grid

if __name__ == "__main__":
    yaml_file_path = os.path.join(
        os.path.dirname(__file__), "../occupancy_maps/World0.yaml"
    )
    map_config = load_yaml_config(yaml_file_path)

    # Load occupancy map using the configuration, with option to flip Y-axis
    occupancy_map, resolution, origin = load_occupancy_map(map_config, flip_y_axis=True)

    print("occupancy_map.shape", occupancy_map.shape)
    plot_occupancy_grid(occupancy_map, resolution, origin)
