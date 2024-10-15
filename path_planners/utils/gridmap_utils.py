import cv2
import yaml
import os
import numpy as np


# Load YAML configuration for ROS map
def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Load and process occupancy map using ROS conventions
def load_occupancy_map(config, flip_y_axis=False):
    # image_path = os.path.join(
    #     os.path.dirname(__file__), "../occupancy_maps", config["image"]
    # )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    image_path = os.path.join(project_root, "occupancy_maps", config["image"])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Normalize the image to 0-1 scale
    normalized_image = image / 255.0

    # Create an empty occupancy map with ROS conventions
    occupancy_map = np.full_like(normalized_image, -1)  # Initialize with -1 (unknown)

    # Assign values according to ROS occupancy conventions
    occupancy_map[normalized_image >= config["occupied_thresh"]] = 100  # Occupied
    occupancy_map[normalized_image <= config["free_thresh"]] = 0  # Free space

    # If negate flag is set, invert the values
    if config["negate"]:
        occupancy_map = 100 - occupancy_map
        occupancy_map[occupancy_map == -101] = -1  # Handle unknown values

    # Flip the map along the Y-axis if needed
    if flip_y_axis:
        occupancy_map = np.flipud(
            occupancy_map
        )  # Flip along the Y-axis (vertical flip)

    # Handle map resolution and origin
    resolution = config["resolution"]
    origin = config["origin"]
    print(f"Map loaded with resolution: {resolution} m/px and origin at: {origin}")
    if flip_y_axis:
        print("Map has been flipped along the Y-axis")

    return occupancy_map, resolution, origin


def load_occupancy_map_by_config_path(config_file_path: str, flip_y_axis: bool = False):
    map_config = load_yaml_config(config_file_path)
    return load_occupancy_map(map_config, flip_y_axis)
