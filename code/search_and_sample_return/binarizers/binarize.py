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
        color_space,
        thresholds,
        morphology_kernel_size,
        morphology_iters = 0
    ):
        if color_space == "RGB":
            self.conversion = cv2.COLOR_BGR2RGB
        else:
            self.conversion = cv2.COLOR_BGR2YUV

        # Thresholds for image binarization:
        self.thresholds = thresholds

        self.morphology_kernel = np.ones(
            (morphology_kernel_size,morphology_kernel_size),
            np.uint8
        )
        self.morphology_iters = morphology_iters

    def transform(self, X):
        """ Binarize input image
        """
        # Convert to HSV:
        converted = cv2.cvtColor(
            X, self.conversion
        )

        # Get mask for each channel component:
        masks = [get_channel_mask(channel_component, threshold) for (channel_component, threshold) in zip(cv2.split(converted), self.thresholds)]

        # Generate final mask:
        mask = masks[0] & masks[1] & masks[2]

        # morphological filtering:
        if self.morphology_iters > 0:
            for _ in range(self.morphology_iters):
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    self.morphology_kernel
                )
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    self.morphology_kernel
                )
        elif self.morphology_iters < 0:
            for _ in range(-self.morphology_iters):
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    self.morphology_kernel
                )
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    self.morphology_kernel
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
        "RGB",
        (
            (160, 255),
            (160, 142),
            (160, 128)
        ),
        3
    )

    binary = binarizer.transform(
        cv2.imread(args["input"])
    )

    cv2.imshow("Binarized", 255 * binary)
    cv2.waitKey(0)
