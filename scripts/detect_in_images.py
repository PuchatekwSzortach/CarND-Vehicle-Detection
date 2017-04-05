"""
Script for detecting vehicles in test images
"""

import os
import glob

import vlogging
import cv2

import cars.config
import cars.utilities


def main():

    logger = cars.utilities.get_logger(cars.config.log_path)

    paths = glob.glob(os.path.join(cars.config.test_images_directory, "*.jpg"))

    for path in paths:

        image = cars.utilities.get_image(path)

        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.info(vlogging.VisualRecord("Image", bgr_image, fmt='jpg'))


if __name__ == "__main__":

    main()