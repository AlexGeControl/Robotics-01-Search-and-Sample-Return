# Set up session:
import numpy as np
import cv2

# Sklearn transformer interface:
from sklearn.base import BaseEstimator, TransformerMixin

class PerspectiveTransformer(TransformerMixin):
    """ Distortion rectification transformer for single view camera
    """

    def __init__(
        self,
        src_points,
        dst_points,
        frame_size
    ):
        # Get transform & inverse transform matrix:
        src_points = np.float32(
            src_points
        )
        dst_points = np.float32(
            dst_points
        )
        # Forward:
        self.M_scene_to_laneline = cv2.getPerspectiveTransform(
            src_points,
            dst_points
        )
        # Inverse:
        self.M_laneline_to_scene = cv2.getPerspectiveTransform(
            dst_points,
            src_points
        )
        # Frame size:
        self.frame_size = frame_size

    def transform(self, X):
        """ Map from scene plane to lane-line plane
        """
        return cv2.warpPerspective(
            X,
            self.M_scene_to_laneline,
            self.frame_size,
            cv2.INTER_NEAREST
        )

    def inverse_transform(self, y):
        """ Map from lane-line plane to scene plane
        """
        return cv2.warpPerspective(
            y,
            self.M_laneline_to_scene,
            self.frame_size,
            cv2.INTER_LINEAR
        )

    def fit(self, X, y=None):
        """ Estimate camera matrix
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
