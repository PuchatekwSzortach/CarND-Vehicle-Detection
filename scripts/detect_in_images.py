"""
Script for detecting vehicles in test images
"""

import os
import glob
import pickle
import random

import vlogging
import cv2
import numpy as np

import cars.config
import cars.utilities
import cars.processing


def main():

    logger = cars.utilities.get_logger(cars.config.log_path)

    with open(cars.config.classifier_path, mode="rb") as file:

        data = pickle.load(file)

        scaler = data['scaler']
        classifier = data['classifier']

    paths = glob.glob(os.path.join(cars.config.test_images_directory, "*.jpg"))

    for path in paths:

        image = cars.utilities.get_image(path)
        luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

        detections = cars.processing.get_detections(
            luv_image, classifier, scaler, parameters=cars.config.parameters, logger=logger)

        for detection in detections:
            cv2.rectangle(image, detection[0], detection[1], thickness=6, color=(0, 255, 0))

        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.info(vlogging.VisualRecord(os.path.basename(path), bgr_image, fmt='jpg'))


if __name__ == "__main__":

    main()