# Set up session:
import numpy as np
import pandas as pd
import cv2

class DataBucket:
    """ Rover state access manager
    """
    def __init__(
        self,
        rover_view_size,
        rover_view_polygon,
        rover_states_filename,
        world_map_filename
    ):
        # Initialize mask for rover view:
        W, H = rover_view_size
        self.rover_view = cv2.fillPoly(
            # Black canvas:
            np.zeros((H, W), dtype=np.uint8),
            # Quadrilateral region mask:
            np.array(
                [
                    rover_view_polygon
                ],
                dtype=np.int
            ),
            255
        )

        # Load rover state log:
        self.rover_states = pd.read_csv(
            rover_states_filename,
            delimiter=';',
            decimal='.'
        )

        # Front images:
        self.images = self.rover_states["Path"].tolist()

        # World map:
        self.world_map = cv2.imread(world_map_filename)
        self.world_map[:, :, 0] = 0
        self.world_map[:, :, 1] = 0.4 * self.world_map[:, :, 1]
        self.world_map[:, :, 2] = 0
        self.world_map = self.world_map.astype(np.uint8)

    def __getitem__(self, index):
        """ Get (X_pos, Y_pos, yaw) corresponding to given index
        """
        if index >= len(self.images):
            return (0, 0, 0)

        return tuple(
            self.rover_states.ix[
                index,
                ['X_Position', 'Y_Position', 'Yaw']
            ]
        )
