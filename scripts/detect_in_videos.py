import pickle
import os

import moviepy.editor

import cars.config
import cars.processing


def make_vehicles_detections_video():

    paths = ["./project_video.mp4"]

    with open(cars.config.classifier_path, mode="rb") as file:

        data = pickle.load(file)

        scaler = data['scaler']
        classifier = data['classifier']

    # video_processor = cars.processing.SimpleVideoProcessor(classifier, scaler, cars.config.parameters)
    video_processor = cars.processing.AdvancedVideoProcessor(classifier, scaler, cars.config.parameters)

    for path in paths:

        clip = moviepy.editor.VideoFileClip(path)

        # print("\n\nWe are only using a subclip now!\n\n".upper())
        # clip = clip.subclip(25, 35)

        processed_clip = clip.fl_image(video_processor.process)

        output_path = os.path.join(cars.config.video_output_directory, os.path.basename(path))
        processed_clip.write_videofile(output_path, fps=12, audio=False)


def main():

    make_vehicles_detections_video()


if __name__ == "__main__":

    main()