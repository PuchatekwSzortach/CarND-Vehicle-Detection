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


def get_scanning_windows_coordinates(image_shape, window_size, step, start=None, end=None):
    """
    Get a list of coordinates of scanning windows
    :param image_shape: shape of image to scan
    :param window_size: size of each window
    :param step: step between windows
    :param start: coordinates at which scanning should start, defaults to upper left image corner
    :param end: coordinates at which scanning should end, defaults to lower right image corner
    :return: list of bounding boxes for windows
    """

    if start is None:
        start = (0, 0)

    if end is None:
        end = (image_shape[1], image_shape[0])

    subwindows = []

    y = start[1]

    while y + window_size <= end[1]:

        x = start[0]

        while x + window_size <= end[0]:

            subwindow = ((x, y), ((x + window_size), (y + window_size)))
            subwindows.append(subwindow)

            x += step

        y += step

    return subwindows


def get_detections(image, classifier, scaler, crop_size):
    """
    Get objects detections in image using provided classifier.
    :param image: numpy array
    :param classifier: classifier
    :param scaler: scaler
    :param crop_size: size of subwindow classifier works with
    :return: list of bounding boxes
    """

    box = ((100, 100), (200, 200))
    return [box]