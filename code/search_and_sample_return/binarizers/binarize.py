# Set up session:
import argparse

import numpy as np
import cv2

from .utils import get_channel_mask

# Sklearn transformer interface:
from sklearn.base import TransformerMixin

class Binarizer(TransformerMixin):
    """ Transform input image to thresholded binary image based on color space filtering
    """
    def __init__(
        self,
        thresholds,
        morphology_kernel_size
    ):
        # Thresholds for image binarization:
        self.thresholds = thresholds
        self.morphology_kernel = np.ones(
            (morphology_kernel_size,morphology_kernel_size),
            np.uint8
        )

    def transform(self, X):
        """ Binarize input image
        """
        # Convert to HSV:
        YUV = cv2.cvtColor(
            X, cv2.COLOR_BGR2YUV
        )

        # Get mask for each channel component:
        masks = [get_channel_mask(channel_component, threshold) for (channel_component, threshold) in zip(cv2.split(YUV), self.thresholds)]

        # Generate final mask:
        mask = masks[0] & masks[1] & masks[2]

        # morphological filtering:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.morphology_kernel,
            iterations=1
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.morphology_kernel,
            iterations=2
        )

        return mask

    def fit(self, X, y=None):
        """ Do nothing
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image filename."
    )
    args = vars(parser.parse_args())

    binarizer = Binarizer(
        (
            (160, 255),
            (128, 142),
            (112, 128)
        ),
        7
    )

    binary = binarizer.transform(
        cv2.imread(args["input"])
    )

    cv2.imshow("Binarized", 255 * binary)
    cv2.waitKey(0)
