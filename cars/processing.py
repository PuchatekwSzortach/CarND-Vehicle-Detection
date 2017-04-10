"""
Module with various processing functions
"""

import time

import skimage.feature
import numpy as np
import cv2
import vlogging

import cars.config


def get_feature_vector(image, parameters):
    """
    Given an image, return its feature vector
    :return: 1D numpy array
    """

    channel_features = [skimage.feature.hog(
        image[:, :, channel], block_norm='L2',
        pixels_per_cell=parameters["pixels_per_cell"], cells_per_block=parameters["cells_per_block"],
        feature_vector=False) for channel in range(3)]

    feature_vector = np.concatenate([features.flatten() for features in channel_features])

    return feature_vector


def get_scanning_windows_coordinates(image_shape, window_size, window_step, start=None, end=None):
    """
    Get a list of coordinates of scanning windows
    :param image_shape: shape of image to scan
    :param window_size: size of each window
    :param window_step: step between windows
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

            x += window_step

        y += window_step

    return subwindows


def get_detections(image, classifier, scaler, parameters, logger):

    start = time.time()

    smallest_scale = 0.25
    largest_scale = 1

    scale = 0.4

    detections = []

    # while scale <= largest_scale:

    print("Processing scale {}".format(scale))

    target_size = (np.array(image.shape[:2]) * scale).astype(np.int)

    scaled_image = cv2.resize(image, (target_size[1], target_size[0]))

    # single_scale_detections = get_single_scale_detections(
    #     scaled_image, classifier, scaler, window_size)

    single_scale_detections = get_fast_single_scale_detections(
        scaled_image, classifier, scaler, parameters=parameters, logger=logger)

    rescaled_detections = []

    for detection in single_scale_detections:

        rescaled_detection = get_scaled_detection(detection, 1 / scale)
        rescaled_detections.append(rescaled_detection)

    print("Added {}".format(len(single_scale_detections)))

    detections.extend(rescaled_detections)

    scale *= 1.1

    print("Detection took: {}".format(time.time() - start))

    return detections


def get_scaled_detection(detection, scale):

    return (
        (int(detection[0][0] * scale), int(detection[0][1] * scale)),
        (int(detection[1][0] * scale), int(detection[1][1] * scale))
        )


def get_single_scale_detections(image, classifier, scaler, window_size):
    """
    Get objects detections in image using provided classifier.
    :param image: numpy array
    :param classifier: classifier
    :param scaler: scaler
    :param window_size: size of subwindow classifier works with
    :return: list of bounding boxes
    """

    windows_coordinates = get_scanning_windows_coordinates(
        image.shape, window_size=window_size, window_step=window_size // 1,
        start=(0, image.shape[0]//2))

    print("Checking {} subwindows".format(len(windows_coordinates)))

    subwindows = [get_subwindow(image, coordinates) for coordinates in windows_coordinates]

    # Slow version, computes HOGS on each subwindow separately
    features = np.array([get_feature_vector(
        image, pixels_per_cell=cars.config.pixels_per_cell, cells_per_block=cars.config.cells_per_block)
                         for image in subwindows])

    scaled_features = scaler.transform(features)

    predictions = classifier.predict(scaled_features)

    detections = []

    for index, prediction in enumerate(predictions):

        if prediction == 1:

            detections.append(windows_coordinates[index])

    return detections


def get_fast_single_scale_detections(image, classifier, scaler, parameters, logger):
    """
    Get objects detections in image using provided classifier.
    :param image: numpy array
    :param classifier: classifier
    :param scaler: scaler
    :param parameters: various detection parameters
    :param logger: logger
    :return: list of bounding boxes
    """

    channels_hog_images = [skimage.feature.hog(
        image[:, :, channel], block_norm='L2',
        pixels_per_cell=parameters["pixels_per_cell"], cells_per_block=parameters["cells_per_block"],
        feature_vector=False) for channel in range(3)]

    windows_coordinates = get_scanning_windows_coordinates(
        image.shape, window_size=parameters["window_size"], window_step=parameters["pixels_per_cell"][0])

    window_size = windows_coordinates[0][1][0] - windows_coordinates[0][0][0]

    hog_block_size = parameters["cells_per_block"][0] * parameters["pixels_per_cell"][0]

    step = ((window_size - hog_block_size) // parameters["pixels_per_cell"][0]) + 1

    detections = []

    features_vectors = []

    for coordinates in windows_coordinates:

        x_start = coordinates[0][0] // parameters["pixels_per_cell"][0]
        y_start = coordinates[0][1] // parameters["pixels_per_cell"][0]

        x_end = x_start + step
        y_end = y_start + step

        channel_features = [channels_hog_images[channel][y_start:y_end, x_start:x_end].flatten()
                            for channel in range(3)]

        feature_vector = np.concatenate(channel_features)
        features_vectors.append(feature_vector)

    features_matrix = np.array(features_vectors)
    scaled_features_matrix = scaler.transform(features_matrix)
    predictions = classifier.predict(scaled_features_matrix)

    for prediction, coordinates in zip(predictions, windows_coordinates):

        if prediction == 1:

            detections.append(coordinates)

        # scaled_feature_vector = scaler.transform(feature_vector.reshape(1, -1))
        # prediction = classifier.predict(scaled_feature_vector)
        #
        # if prediction == 1:
        #
        #     detections.append(coordinates)
    #
    # print("Checking {} subwindows".format(len(windows_coordinates)))
    #
    # subwindows = [get_subwindow(image, coordinates) for coordinates in windows_coordinates]
    #
    # # Slow version, computes HOGS on each subwindow separately
    # features = np.array([get_feature_vector(image) for image in subwindows])
    #
    # scaled_features = scaler.transform(features)
    #
    # predictions = classifier.predict(scaled_features)

    # for index, prediction in enumerate(predictions):
    #
    #     if prediction == 1:
    #
    #         detections.append(windows_coordinates[index])

    return detections


def get_subwindow(image, coordinates):
    """
    Given image and region of interest coordinates, return subwindow
    :param image:
    :param coordinates:
    :return: subwindow
    """

    return image[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]