"""
Module with various processing functions
"""

import skimage.feature
import numpy as np


def get_feature_vector(image):
    """
    Given an image, return its feature vector
    :param image:
    :return: 1D numpy array
    """

    # Compute HOG features over all channels
    channel_features = [
        skimage.feature.hog(image[:, :, channel], block_norm='L2', cells_per_block=(4, 4)) for channel in range(3)]
    feature_vector = np.concatenate(channel_features)

    return feature_vector
