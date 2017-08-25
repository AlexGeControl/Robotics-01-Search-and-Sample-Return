# Set up session:
import argparse

from os.path import basename
import json

import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

def get_region_mask(image, top_left, bottom_right):
    """ Generate quadrilateral region mask for lane detection
    """
    H, W, _ = image.shape

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(
        mask,
        top_left,
        bottom_right,
        1,
        -1
    )

    return mask

def get_annotations(annotation_filename):
    # Load annotations:
    with open(annotation_filename) as annotation_file:
        annotations = json.load(annotation_file)

    pixel_values = []
    for annotation in annotations:
        # Image
        image = cv2.imread(annotation["filename"])
        YUV = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2YUV
        )

        # Mask:
        (top, left, height, width) = (
            annotation["annotations"][0]["y"],
            annotation["annotations"][0]["x"],
            annotation["annotations"][0]["height"],
            annotation["annotations"][0]["width"]
        )
        top_left = (int(left), int(top))
        bottom_right = (int(left + width), int(top + height))
        mask = get_region_mask(image, top_left, bottom_right)

        pixel_values.append(
            YUV[mask > 0]
        )

    return np.vstack(tuple(pixel_values))


if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--annotation",
        type=str,
        required=True,
        help="Ground region annotation filename."
    )
    args = vars(parser.parse_args())

    pixel_values = get_annotations(args["annotation"])
    '''
    pixel_values = pixel_values[
        np.random.choice(len(pixel_values), 1000)
    ]
    '''
    
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(
        pixel_values[:, 0],
        pixel_values[:, 1],
        pixel_values[:, 2],
        c = ('r', 'g', 'b')
    )

    ax.set_xlabel('Y')
    ax.set_ylabel('U')
    ax.set_zlabel('V')

    plt.show()
