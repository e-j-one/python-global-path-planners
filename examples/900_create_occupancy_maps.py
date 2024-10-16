from collections import OrderedDict

from PIL import Image
import numpy as np
import yaml


# Write the dictionary to a YAML file
def write_yaml_compact_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


if __name__ == "__main__":
    map_config = {
        "image": "test_occupancy_map_0.png",
        "resolution": 1.0,
        "origin": [-0.5, -0.5, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    costmap = np.array(
        [
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    costmap = np.flipud(costmap)

    # convert costmap to grey scale. Occupied cells (100.0) are black (0), free cells (0.0) are white (255)
    costmap_grey = np.full_like(costmap, 255, dtype=np.uint8)
    costmap_grey[costmap == 100.0] = 0

    # costmap_normalized = (costmap * 255).astype(np.uint8)
    img = Image.fromarray(costmap_grey)
    img.save("occupancy_maps/test_occupancy_map_0.png")

    yaml.add_representer(list, write_yaml_compact_list)
    with open("occupancy_maps/test_occupancy_map_0.yaml", "w") as file:
        yaml.dump(map_config, file)
