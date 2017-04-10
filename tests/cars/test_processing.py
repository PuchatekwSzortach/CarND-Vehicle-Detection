"""
Tests for processing module
"""

import numpy as np

import cars.processing


def test_get_scanning_windows_coordinates_horizontal_scan():

    image_shape = (3, 10)

    expected = [
        ((0, 0), (3, 3)),
        ((3, 0), (6, 3)),
        ((6, 0), (9, 3)),
    ]

    actual = cars.processing.get_scanning_windows_coordinates(image_shape, window_size=3, window_step=3)

    assert expected == actual


def test_get_scanning_windows_coordinates_vertical_scan():

    image_shape = (10, 4)

    expected = [
        ((0, 0), (3, 3)),
        ((0, 4), (3, 7)),
    ]

    actual = cars.processing.get_scanning_windows_coordinates(image_shape, window_size=3, window_step=4)

    assert expected == actual


def test_get_scanning_windows_coordinates_start_defined():

    image_shape = (4, 12)

    expected = [
        ((5, 0), (8, 3)),
        ((8, 0), (11, 3)),
    ]

    actual = cars.processing.get_scanning_windows_coordinates(
        image_shape, window_size=3, window_step=3, start=(5, 0))

    assert expected == actual


def test_get_scanning_windows_coordinates_end_defined():

    image_shape = (4, 12)

    expected = [
        ((0, 0), (4, 4)),
        ((3, 0), (7, 4)),
    ]

    actual = cars.processing.get_scanning_windows_coordinates(
        image_shape, window_size=4, window_step=3, end=(8, 4))

    assert expected == actual