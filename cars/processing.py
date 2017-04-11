"""
Module with various processing functions
"""

import time

import skimage.feature
import numpy as np
import cv2
import vlogging
import scipy.ndimage.measurements
import shapely.geometry

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


def get_detections(image, classifier, scaler, parameters):

    start = time.time()

    scales = [0.2, 0.25, 0.35, 0.5]
    relative_starts = [(0.2, 0.5), (0.2, 0.6), (0.4, 0.6), (0.5, 0.5)]
    relative_ends = [(1, 1), (1, 0.85), (0.8, 0.8), (0.8, 0.7)]
    window_steps = [8, 8, 8, 8]

    detections = []

    # Search image at different scales
    for scale, relative_start, relative_end, window_step in zip(scales, relative_starts, relative_ends, window_steps):

        target_size = (np.array(image.shape[:2]) * scale).astype(np.int)
        scaled_image = cv2.resize(image, (target_size[1], target_size[0]))

        single_scale_detections = get_single_scale_detections(
            scaled_image, classifier, scaler, parameters=parameters,
            relative_start_position=relative_start, relative_end_position=relative_end, window_step=window_step)

        rescaled_detections = []

        for detection in single_scale_detections:

            rescaled_detection = get_scaled_detection(detection, 1 / scale)
            rescaled_detections.append(rescaled_detection)

        detections.extend(rescaled_detections)

    # return detections

    # Draw all detections on a heatmap
    heatmap = np.zeros(image.shape[:2])

    for detection in detections:

        heatmap[detection[0][1]:detection[1][1], detection[0][0]:detection[1][0]] += 1

    # Filter out false positives
    heatmap[heatmap < parameters["heatmap_threshold"]] = 0

    # print(np.max(heatmap))

    # Get connected components to merge multiple positive detections
    labels_image, labels_count = scipy.ndimage.measurements.label(heatmap)

    # Convert results into bounding boxes
    cars_detections = get_bounding_boxes_from_labels(labels_image, labels_count)

    # print("Detection took {:.3f} seconds".format(time.time() - start))

    # Do a last check on detections
    sensible_detections = [detection for detection in cars_detections if is_detection_sensible(detection)]
    return sensible_detections


def get_bounding_boxes_from_labels(labels_image, labels_count):

    detections = []

    for label_index in range(1, labels_count + 1):

        # Slightly modified code from Udacity class
        nonzero_indices = (labels_image == label_index).nonzero()

        # Identify x and y values of those pixels
        nonzero_ys = np.array(nonzero_indices[0])
        nonzero_xs = np.array(nonzero_indices[1])
        # Define a bounding box based on min/max x and y

        detection = ((np.min(nonzero_xs), np.min(nonzero_ys)), (np.max(nonzero_xs), np.max(nonzero_ys)))
        detections.append(detection)

    return detections


def get_scaled_detection(detection, scale):

    return (
        (int(detection[0][0] * scale), int(detection[0][1] * scale)),
        (int(detection[1][0] * scale), int(detection[1][1] * scale))
        )


def get_single_scale_detections(
        image, classifier, scaler, parameters, relative_start_position, relative_end_position, window_step):
    """
    Get objects detections in image using provided classifier.
    :param image: numpy array
    :param classifier: classifier
    :param scaler: scaler
    :param parameters: various detection parameters
    :param relative_start_position: x, y position from which scanning should start
    :param relative_end_position: x, y position at which scanning should end
    :param window_step: scanning window step
    :return: list of bounding boxes
    """

    channels_hog_images = [skimage.feature.hog(
        image[:, :, channel], block_norm='L2',
        pixels_per_cell=parameters["pixels_per_cell"], cells_per_block=parameters["cells_per_block"],
        feature_vector=False) for channel in range(3)]

    start_position = (int(relative_start_position[0] * image.shape[1]), int(relative_start_position[1] * image.shape[0]))
    end_position = (int(relative_end_position[0] * image.shape[1]), int(relative_end_position[1] * image.shape[0]))

    windows_coordinates = get_scanning_windows_coordinates(
        image.shape, window_size=parameters["window_size"], window_step=window_step,
        start=start_position, end=end_position)

    window_size = windows_coordinates[0][1][0] - windows_coordinates[0][0][0]

    hog_block_size = parameters["cells_per_block"][0] * parameters["pixels_per_cell"][0]

    hog_window_span = ((window_size - hog_block_size) // parameters["pixels_per_cell"][0]) + 1

    detections = []

    features_vectors = []

    for coordinates in windows_coordinates:

        x_start = coordinates[0][0] // parameters["pixels_per_cell"][0]
        y_start = coordinates[0][1] // parameters["pixels_per_cell"][0]

        x_end = x_start + hog_window_span
        y_end = y_start + hog_window_span

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

    return detections


def get_subwindow(image, coordinates):
    """
    Given image and region of interest coordinates, return subwindow
    :param image:
    :param coordinates:
    :return: subwindow
    """

    return image[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]


def is_detection_sensible(detection):
    """
    A simple predicate that checkes if a detection seems sensible.
    It does that by looking at its size and aspect ratio
    :param detection:
    :return: bool
    """

    x_size = detection[1][0] - detection[0][0]
    y_size = detection[1][1] - detection[0][1]

    if x_size < 32 or y_size < 32:

        return False

    aspect_ratio = x_size / y_size

    if aspect_ratio < 0.25 or aspect_ratio > 4:

        return False

    # If all tests passed, detection looks alright
    return True


def get_intersection_over_union(first_detection, second_detection):

    first_box = shapely.geometry.box(
        first_detection[0][0], first_detection[0][1], first_detection[1][0], first_detection[1][1])

    second_box = shapely.geometry.box(
        second_detection[0][0], second_detection[0][1], second_detection[1][0], second_detection[1][1])

    intersection_polygon = first_box.intersection(second_box)
    union_polygon = first_box.union(second_box)

    return intersection_polygon.area / union_polygon.area


class SimpleVideoProcessor:
    """
    A simple video processor that detects cars in each frame separately and doesn't combine results between frames
    """

    def __init__(self, classifier, scaler, parameters):

        self.classifier = classifier
        self.scaler = scaler
        self.parameters = parameters

    def process(self, frame):

        luv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2LUV)

        detections = get_detections(luv_image, self.classifier, self.scaler, self.parameters)

        for detection in detections:
            cv2.rectangle(frame, detection[0], detection[1], thickness=6, color=(0, 255, 0))

        return frame


class AdvancedVideoProcessor:
    """
    A video processor that looks at detections from consecutive frames to compute final detection bounding boxes
    """

    def __init__(self, classifier, scaler, parameters):

        self.classifier = classifier
        self.scaler = scaler
        self.parameters = parameters

        self.tracked_detections_and_counts = []

    def process(self, frame):

        luv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2LUV)
        detections = get_detections(luv_image, self.classifier, self.scaler, self.parameters)

        new_detections_and_counts = []

        tracked_detections_matches = [False] * len(self.tracked_detections_and_counts)

        # Go over all detections from current frame
        for detection in detections:

            is_detection_matched = False

            # Try to match them to any of existing detections
            for index in range(len(self.tracked_detections_and_counts)):

                # If IOU is high update tracked detection
                if get_intersection_over_union(self.tracked_detections_and_counts[index][0], detection) > 0.5:

                    is_detection_matched = True
                    tracked_detections_matches[index] = True

                    tracked_detection = self.tracked_detections_and_counts[index][0]

                    # Update tracked detection position
                    left_top = ((tracked_detection[0][0] + detection[0][0]) // 2,
                                (tracked_detection[0][1] + detection[0][1]) // 2)

                    right_bottom = ((tracked_detection[1][0] + detection[1][0]) // 2,
                                    (tracked_detection[1][1] + detection[1][1]) // 2)

                    # Update
                    self.tracked_detections_and_counts[index][0] = (left_top, right_bottom)
                    self.tracked_detections_and_counts[index][1] += 1

            # If detection wasn't matched to any of existing detections, add it as a new detection
            if is_detection_matched is False:

                new_detections_and_counts.append([detection, 1])

        # Any existing detection that wasn't matched should have its count decreased
        for index in range(len(self.tracked_detections_and_counts)):

            if tracked_detections_matches[index] is False:

                self.tracked_detections_and_counts[index][1] -= 1

        self._remove_stale_detections()
        self.tracked_detections_and_counts.extend(new_detections_and_counts)

        self._draw_tracked_detections(frame, self.tracked_detections_and_counts)

        return frame

    def _remove_stale_detections(self):

        tracked_detections_copy = self.tracked_detections_and_counts.copy()

        # Any tracked detections that aren't seen for a while, should be removed
        for detection_count in tracked_detections_copy:

            if detection_count[1] <= -3:

                self.tracked_detections_and_counts.remove(detection_count)

    def _draw_tracked_detections(self, frame, tracked_detections_and_counts):

        for detection, count in tracked_detections_and_counts:

            if count >= 3:

                cv2.rectangle(frame, detection[0], detection[1], thickness=6, color=(0, 255, 0))
