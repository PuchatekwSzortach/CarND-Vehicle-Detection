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


def get_detections(image, classifier, scaler, window_size):
    """
    Get objects detections in image using provided classifier.
    :param image: numpy array
    :param classifier: classifier
    :param scaler: scaler
    :param window_size: size of subwindow classifier works with
    :return: list of bounding boxes
    """

    windows_coordinates = get_scanning_windows_coordinates(
        image.shape, window_size=window_size, step=window_size // 1,
        start=(0, image.shape[0]//2))

    subwindows = [get_subwindow(image, coordinates) for coordinates in windows_coordinates]

    # Slow version, computes HOGS on each subwindow separately
    features = np.array([get_feature_vector(image) for image in subwindows])

    scaled_features = scaler.transform(features)

    predictions = classifier.predict(scaled_features)

    detections = []

    for index, prediction in enumerate(predictions):

        if prediction == 1:

            detections.append(windows_coordinates[index])

    return detections


def get_subwindow(image, coordinates):
    """
    Given image and region of interest coordinates, return subwindow
    :param image:
    :param coordinates:
    :return: subwindow
    """

    return image[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]