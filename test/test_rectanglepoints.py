import numpy as np
import pytest

from .context import (Point, RectanglePoints, SAMADatasetImporterException)

RECTANGLE = RectanglePoints(
        np.array([[200, 600], [200, 200], [800, 200], [800, 600]]))


def test_big_rectangle_width_and_height():
    width, height = RECTANGLE.calculate_width_and_height()
    expected_width, expected_height = 400, 600

    assert expected_width == width
    assert  expected_height == height


def test_get_rectangle_top_left_point():
    top_left_point = RECTANGLE.get_top_left_point()
    print(top_left_point)
    expected_top_left_point = Point(200,600)

    assert top_left_point.__eq__(expected_top_left_point) 

def test_get_bottom_right_point():
    bottom_right_point = RECTANGLE.get_bottom_right_point()
    print(bottom_right_point)
    expected_bottom_right_point = Point(800,200)

    assert bottom_right_point.__eq__(expected_bottom_right_point)


def test_get_top_left_point():
    point = RECTANGLE.get_top_left_point()
    expected_point = Point(200,600)

    assert point.__eq__(expected_point)

def test_get_bottom_right_point():
    point = RECTANGLE.get_bottom_right_point()
    expected_point = Point(800,200)

    assert point.__eq__(expected_point)

def test_triangle_rectangle():
    with pytest.raises(SAMADatasetImporterException) as rectangle_points_exception:
         RectanglePoints([[2, 3], [1, 1], [3, 1]])
    assert 'details: ERROR, the rectangle points are not valid' == str(
        rectangle_points_exception.value)

