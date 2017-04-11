"""
Module for training a car vs not a car classifier
"""

import time
import pickle

import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import cv2
import vlogging

import cars.config
import cars.utilities
import cars.processing


def main():

    logger = cars.utilities.get_logger("/tmp/vehicle_detection.html")

    # Get data
    images, labels = cars.utilities.get_basic_dataset(
        cars.config.basic_dataset_vehicles_path,
        cars.config.basic_dataset_non_vehicles_path)

    # Create feature vectors from data
    luv_images = [cv2.cvtColor(image, cv2.COLOR_RGB2LUV) for image in images]

    features = np.array([cars.processing.get_feature_vector(image, cars.config.parameters)
                         for image in luv_images])

    # Shuffle data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, labels, test_size=0.3)

    # Normalize data
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)

    # Select data size
    size = len(X_train)
    X_train, X_test, y_train, y_test = X_train[:size], X_test[:size], y_train[:size], y_test[:size]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    start = time.time()
    classifier = sklearn.svm.SVC(kernel='linear')
    classifier.fit(X_train_scaled, y_train)

    print("Classification score is: {}".format(classifier.score(X_test_scaled, y_test)))
    print("Training classifier took {:.2f} seconds".format(time.time() - start))

    # # Save scaler and classifier
    data = {'scaler': scaler, 'classifier': classifier}

    with open(cars.config.classifier_path, mode="wb") as file:
        pickle.dump(data, file)


if __name__ == "__main__":

    main()
