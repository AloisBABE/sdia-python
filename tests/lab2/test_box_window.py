import numpy as np
import pytest

from lab2.box_window import BallWindow, BoxWindow, UnitBoxWindow


def test_raise_type_error_when_something_is_called():
    with pytest.raises(TypeError):
        # call_something_that_raises_TypeError()
        raise TypeError()


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[2.5, 2.5]]), "BoxWindow: [2.5, 2.5]"),
        (np.array([[0, 5], [0, 5]]), "BoxWindow: [0, 5] x [0, 5]"),
        (
            np.array([[0, 5], [-1.45, 3.14], [-10, 10]]),
            "BoxWindow: [0, 5] x [-1.45, 3.14] x [-10, 10]",
        ),
    ],
)
def test_box_string_representation(bounds, expected):
    assert str(BoxWindow(bounds)) == expected


@pytest.fixture
def box_2d_05():
    return BoxWindow(np.array([[0, 5], [0, 5]]))


@pytest.mark.parametrize(
    "points, expected",
    [
        (np.array([0, 0]), np.array([True])),
        (np.array([2.5, 2.5]), np.array([True])),
        (np.array([[-1, 5], [0, 5]]), np.array([False, True])),
        (np.array([[10, 3], [1, 2], [2.5, 4.6]]), np.array([False, True, True])),
    ],
)
def test_indicator_function_box_2d(box_2d_05, points, expected):
    is_in = box_2d_05.indicator_function(points)
    assert np.array_equal(is_in, expected)


# ================================
# ==== WRITE YOUR TESTS BELOW ====
# ================================


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[0, 4], [0, 6]]), 6),
        (np.array([[0, 3], [0, 6]]), 6),
        (np.array([[-1, 4], [3.5, 3.6], [5, 9]]), 5),
    ],
)
def test_len(bounds, expected):
    box = BoxWindow(bounds)
    length = len(box)
    assert expected == length


@pytest.mark.parametrize(
    "bounds, n",
    [
        (np.array([[0, 4], [0, 6]]), 100),
        (np.array([[0, 3], [0, 6]]), 1000),
        (np.array([[-1, 4], [3.5, 3.6], [5, 9]]), 10000),
    ],
)
def test_rand(bounds, n):
    box = BoxWindow(bounds)
    points = box.rand(n)
    assert np.array_equal(box.indicator_function(points), True * np.ones(n))


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (np.array([[0, 4], [0, 6]]), np.array([2, 3])),
        (np.array([[0, 3], [0, 6]]), np.array([1.5, 3])),
        (np.array([[-1, 4], [3.5, 3.6], [5, 9]]), np.array([1.5, 3.55, 7])),
    ],
)
def test_center_box_(bounds, expected):
    box = BoxWindow(bounds)
    center = box.center()
    assert np.array_equal(center, expected)


@pytest.mark.parametrize(
    "center, expected",
    [
        (np.array([0.5, 0.5, 0.5]), np.array([[0, 1], [0, 1], [0, 1]])),
        (np.array([6.5]), np.array([[6, 7]])),
    ],
)
def test_bounds_unitBox_(center, expected):
    box = UnitBoxWindow(center)
    bounds = box.bounds
    assert np.array_equal(bounds, expected)


@pytest.mark.parametrize(
    "center", [(np.array([0.5, 0.5, 0.5])), (np.array([6.5, 7, 18, 6.23])),],
)
def test_volume_unitBox_(center):
    box = UnitBoxWindow(center)
    volume = box.volume()
    assert volume == 1


@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([1, 2, 3]), 4, "BallWindow: center:[1 2 3] radius:4"),
        (np.array([1]), 4.2, "BallWindow: center:[1] radius:4.2"),
    ],
)
def test_ball_string_representation(center, radius, expected):
    ball = BallWindow(center, radius)
    string = str(ball)
    assert string == expected


@pytest.mark.parametrize(
    "center, radius, expected",
    [(np.array([1, 2, 3]), 4, 8), (np.array([1, 5.12, 4.65, np.pi]), 4.21, 8.42),],
)
def test_ball_string_representation(center, radius, expected):
    ball = BallWindow(center, radius)
    diameter = len(ball)
    assert diameter == expected


@pytest.mark.parametrize(
    "center, radius, expected",
    [
        (np.array([3, 4]), 5, np.pi * 5 ** 2),
        (np.array([6.5, 7, 18]), 4, 4 * np.pi * 4 ** 3 / 3),
    ],
)
def test_volume_BallWindow_(center, radius, expected):
    box = BallWindow(center, radius)
    volume = box.volume()
    assert np.abs(volume - expected) < 10 ** (-8)


@pytest.mark.parametrize(
    "center, radius, point, expected",
    [
        (np.array([0, 0]), 1, np.array([0, 0]), True),
        (np.array([5, 4, 1]), 3, np.array([8, 4, 1]), True),
        (np.array([7, 0, 4]), 2, np.array([9, 0.01, 4.01]), False),
        (
            np.array([0, 0]),
            2,
            np.array([[0, 1], [1, 1], [5, -3]]),
            np.array([True, True, False]),
        ),
    ],
)
def test_indicator_function_ballWindow(center, radius, point, expected):
    ball = BallWindow(center, radius)
    is_in = ball.indicator_function(point)
    assert np.array_equal(is_in, expected)
