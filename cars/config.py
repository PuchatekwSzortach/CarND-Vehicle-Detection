"""
Simple config
"""

basic_dataset_vehicles_path = "../../data/vehicle_detection/vehicles"
basic_dataset_non_vehicles_path = "../../data/vehicle_detection/non-vehicles"

classifier_path = "../../data/vehicle_detection/classifier.p"
log_path = "/tmp/vehicles_detection.html"

test_images_directory = "./test_images"

parameters = {
        "window_size": 64,
        "window_step": 16,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (4, 4),
        "heatmap_threshold": 3
    }

video_output_directory = "../../data/vehicle_detection/"
