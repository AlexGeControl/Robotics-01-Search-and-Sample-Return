# Set up session:
import numpy as np
import cv2

def get_channel_mask(
    image,
    threshold
):
    """ Generate mask based on channel component values

    Args:
        image (numpy 2-d array): input image with selected channel component
        channel_idx (int): channel index
        thresholds (2-element tuple): min & max values for thresholding
    """
    # Image dimensions:
    H, W = image.shape

    # Generate mask:
    mask = np.zeros((H, W), dtype=np.uint8)

    channel_min, channel_max = threshold

    mask[
        (channel_min <= image) & (image <= channel_max)
    ] = 1

    return mask
