import cv2
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt

# Load YAML configuration for ROS map
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load and process occupancy map using ROS conventions
def load_occupancy_map(config, flip_y_axis=False):
    image_path = os.path.join(os.path.dirname(__file__), "../occupancy_maps", config['image'])
    
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
    occupancy_map[normalized_image >= config['occupied_thresh']] = 100  # Occupied
    occupancy_map[normalized_image <= config['free_thresh']] = 0        # Free space

    # If negate flag is set, invert the values
    if config['negate']:
        occupancy_map = 100 - occupancy_map
        occupancy_map[occupancy_map == -101] = -1  # Handle unknown values

    # Flip the map along the Y-axis if needed
    if flip_y_axis:
        occupancy_map = np.flipud(occupancy_map)  # Flip along the Y-axis (vertical flip)

    # Handle map resolution and origin
    resolution = config['resolution']
    origin = config['origin']
    print(f"Map loaded with resolution: {resolution} m/px and origin at: {origin}")
    if flip_y_axis:
        print("Map has been flipped along the Y-axis")

    return occupancy_map, resolution, origin

def visualize_occupancy_grid(occupancy_grid, resolution, origin):
    """
    Visualizes the occupancy grid using matplotlib.
    
    Parameters:
    - occupancy_grid: 2D numpy array representing the occupancy grid
    - resolution: The resolution of the map (meters per pixel)
    - origin: Origin of the map in world coordinates (x, y, theta)
    """
    # Create a figure and axis
    plt.figure(figsize=(8, 8))
    
    # Define a color map for occupancy
    # - Free space (0) -> white
    # - Occupied space (100) -> black
    # - Unknown space (-1) -> gray
    cmap = plt.cm.gray
    cmap.set_under(color='white')  # Color for free space
    cmap.set_over(color='black')   # Color for occupied space
    cmap.set_bad(color='lightgray') # Color for unknown space
    
    # Mask unknown values (-1) so they appear as "bad" (gray) values in the plot
    masked_grid = np.ma.masked_where(occupancy_grid == -1, occupancy_grid)
    
    # Visualize the grid with the specified colormap
    plt.imshow(masked_grid, cmap=cmap, origin='lower', interpolation='none', extent=[
        origin[0], origin[0] + occupancy_grid.shape[1] * resolution,
        origin[1], origin[1] + occupancy_grid.shape[0] * resolution
    ])
    
    # Add labels and title
    plt.title('Occupancy Grid Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    
    # Display color bar for reference
    cbar = plt.colorbar()
    cbar.set_label('Occupancy Value')
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(['Free', 'Unknown', 'Occupied'])
    
    # Show the plot
    plt.grid(False)
    plt.show()

# Main function to load and display the map using OpenCV
if __name__ == "__main__":
    # Path to the YAML file
    yaml_file = os.path.join(os.path.dirname(__file__), "../occupancy_maps/World0.yaml")

    # Load configuration from YAML file
    config = load_yaml_config(yaml_file)

    # Load occupancy map using the configuration, with option to flip Y-axis
    occupancy_map, resolution, origin = load_occupancy_map(config, flip_y_axis=True)

    # Display the occupancy map using OpenCV (optional)
    scaled_map = cv2.resize(occupancy_map, (occupancy_map.shape[1]*2, occupancy_map.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    visualize_occupancy_grid(scaled_map, resolution, origin)
