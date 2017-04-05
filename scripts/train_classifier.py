"""
Module for training a car vs not a car classifier
"""

import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm

import cars.config
import cars.utilities


def main():

    # Get data
    images, labels = cars.utilities.get_basic_dataset(
        cars.config.basic_dataset_vehicles_path,
        cars.config.basic_dataset_non_vehicles_path)

    # Create feature vectors from data
    images = np.array([image.ravel() for image in images]).astype(np.float32)

    # Shuffle data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        images, labels, test_size=0.3)

    # Normalize data
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    classifier = sklearn.svm.SVC(kernel='linear')
    classifier.fit(X_train_scaled, y_train)

    print("Score is: {}".format(classifier.score(X_test_scaled, y_test)))

    # Save scaler and classifier


if __name__ == "__main__":

    main()