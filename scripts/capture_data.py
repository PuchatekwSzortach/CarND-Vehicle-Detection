"""
Simple script to extract a few additional frames from project video
"""

import moviepy.editor
import cv2
import vlogging

import cars.utilities
import cars.config


def main():

    logger = cars.utilities.get_logger(cars.config.log_path)

    path = "./project_video.mp4"
    clip = moviepy.editor.VideoFileClip(path)

    for time in [8, 9, 29, 31]:

        frame = clip.get_frame(t=time)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        logger.info(vlogging.VisualRecord(str(time), [frame], fmt="jpg"))


if __name__ == "__main__":

    main()