# Set up session:
import numpy as np
import cv2

# Map from pixel coordinates to rover coordinates:
class RoverCoordMapper:
    """ Map ROI from pixel coordinates to rover coordinates
    """
    def __init__(
        self,
        frame_size
    ):
        self.W, self.H = frame_size

    def transform(self, X):
        # Unpack pixel coordinates:
        y_pixel, x_pixel = X.nonzero()
        
        # Map to rover coordinate:
        x_rover = (self.H - y_pixel).astype(np.float)
        y_rover = (self.W / 2 - x_pixel).astype(np.float)

        return (x_rover, y_rover)

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

# Map from rover standard coordinates to rover polar coordinates:
class RoverPolarMapper:
    """ Map from rover standard coordinates to rover polar coordinates
    """
    def __init__(
        self
    ):
        pass

    def transform(self, coords):
        # Unpack standard coordinates:
        (x_standard, y_standard) = coords

        rho = np.sqrt(x_standard**2 + y_standard**2)
        theta = np.arctan2(y_standard, x_standard)

        return (rho, theta)

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

# Map from rover coordinates to world coordinates:
class WorldCoordMapper:
    """ Map ROI from rover coordinates to world coordinates
    """

    def __init__(
        self,
        world_size
    ):
        self.W, self.H = world_size

    def transform(self, coords, rotation, scales, translations):
        # Unpack rover coordinates:
        (x_rover, y_rover) = coords
        # Rotate:
        x_world = np.cos(rotation)*x_rover - np.sin(rotation)*y_rover
        y_world = np.sin(rotation)*x_rover + np.cos(rotation)*y_rover

        # Unpack scales and translations:
        (x_scale, y_scale) = scales
        (x_trans, y_trans) = translations
        # Scale and translate:
        x_world = (x_world / x_scale) + x_trans
        y_world = (y_world / y_scale) + y_trans

        # Clip:
        x_world = np.int_(np.clip(x_world, 0, self.W))
        y_world = np.int_(np.clip(y_world, 0, self.H))

        return (x_world, y_world)

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
