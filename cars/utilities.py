"""
Utilities module
"""

import os
import glob
import logging

import matplotlib.image
import numpy as np


def get_logger(path):
    """
    Returns a logger that writes to an html page
    :param path: path to html page
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("simulation")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_image(path):
    """
    Given image path returns numpy array representing image.
    Array is of type uint8, in range 0-255 and in RGB order
    :param path: path to image
    :return: numpy array
    """

    _, extension = os.path.splitext(path)

    if extension == ".png":

        image = matplotlib.image.imread(path)
        image = (255 * image).astype(np.uint8)

        return image

    elif extension == ".jpg" or extension == ".jpeg":

        return matplotlib.image.imread(path)

    else:

        raise NotImplementedError()


def get_basic_dataset(vehicles_path, non_vehicles_path):
    """
    Read vehicles and non vehicles images, return images and labels
    :param vehicles_path:
    :param non_vehicles_path:
    :return: tuple (images, labels)
    """

    vehicles_files = glob.glob(os.path.join(vehicles_path, "**/*.png"), recursive=True)
    non_vehicles_files = glob.glob(os.path.join(non_vehicles_path, "**/*.png"), recursive=True)

    vehicles_images = [get_image(path) for path in vehicles_files]
    non_vehicles_images = [get_image(path) for path in non_vehicles_files]

    images = vehicles_images + non_vehicles_images
    labels = ([1] * len(vehicles_images)) + ([0] * len(non_vehicles_images))

    return images, labels