# Set up session:
# IO:
from search_and_sample_return.utils.conf import Conf
import pickle
# General image processing:
import numpy as np
import cv2
# Task specific module:
from search_and_sample_return.binarizers import Binarizer
from search_and_sample_return.transformers import PerspectiveTransformer
from search_and_sample_return.mappers import RoverCoordMapper, RoverPolarMapper, WorldCoordMapper
from search_and_sample_return.utils.databucket import DataBucket
from search_and_sample_return.painters import BirdEyeViewPainter, WorldMapPainter

# Load config:
conf = Conf("conf/conf.json")

# Load binarizers:
with open(conf.binarizer_ground_pickle, "rb") as binarizer_ground_pkl:
    binarizer_ground = pickle.load(binarizer_ground_pkl)
with open(conf.binarizer_rock_pickle, "rb") as binarizer_rock_pkl:
    binarizer_rock = pickle.load(binarizer_rock_pkl)

# Load transformer:
with open(conf.perspective_transformer_pickle, "rb") as transformer_pkl:
    transformer = pickle.load(transformer_pkl)

# Load mappers:
with open(conf.rover_coord_mapper_pickle, "rb") as rover_coord_mapper_pkl:
    rover_coord_mapper = pickle.load(rover_coord_mapper_pkl)
with open(conf.rover_polar_mapper_pickle, "rb") as rover_polar_mapper_pkl:
    rover_polar_mapper = pickle.load(rover_polar_mapper_pkl)
with open(conf.world_coord_mapper_pickle, "rb") as world_coord_mapper_pkl:
    world_coord_mapper = pickle.load(world_coord_mapper_pkl)

# Initialize rover state accessor:
rover_states = DataBucket(
    conf.frame_size,
    conf.frame_view,
    conf.test_dataset_states,
    conf.world_map
)

# Initialize painters:
bird_eye_view_painter = BirdEyeViewPainter(
    conf.frame_size
)
world_map_painter = WorldMapPainter()

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    # Format input:
    frame = cv2.cvtColor(Rover.img, cv2.COLOR_RGB2BGR)

    # Rover state:
    (x_trans, y_trans) = Rover.pos
    yaw = Rover.yaw

    # Segmented ground:
    ground = transformer.transform(
        binarizer_ground.transform(frame)
    )
    # Segmented obstacle:
    obstacle = (ground == 0).astype(
        np.int
    )
    obstacle[rover_states.rover_view == 0] = 0
    # Segmented rock:
    rock = transformer.transform(
        binarizer_rock.transform(frame)
    )

    # Initialize coordinate transform:
    yaw = np.pi / 180.0 * yaw
    scales = (conf.scale, conf.scale)
    translations= (x_trans, y_trans)

    # Extract coordinates:
    coords = {
        "ground": {},
        "obstacle": {},
        "rock": {}
    }
    for obj_name, obj_in_pixel in zip(
        ("ground", "obstacle", "rock"),
        (ground, obstacle, rock),
    ):
        coords[obj_name]["rover"] = rover_coord_mapper.transform(obj_in_pixel)
        coords[obj_name]["polar"] = rover_polar_mapper.transform(
            coords[obj_name]["rover"]
        )
        coords[obj_name]["world"] = world_coord_mapper.transform(
            coords[obj_name]["rover"],
            yaw,
            scales,
            translations
        )

    # Bird eye view:
    Rover.vision_image[:,:,0] = 255 * obstacle
    if rock.any():
        Rover.vision_image[:,:,1] = 255 * rock
    else:
        Rover.vision_image[:,:,1] = 0
    Rover.vision_image[:,:,2] = 255 * ground

    # World map inpainting:
    (obstacle_x_world, obstacle_y_world) = coords["obstacle"]["world"]
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 10
    if rock.any():
        (rock_x_world, rock_y_world) = coords["rock"]["world"]
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
    (ground_x_world, ground_y_world) = coords["ground"]["world"]
    Rover.worldmap[ground_y_world, ground_x_world, 2] += 10

    # Update navigation angle:
    Rover.nav_angles = coords["ground"]["polar"][1]

    return Rover
