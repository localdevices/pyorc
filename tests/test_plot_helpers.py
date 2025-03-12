import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
from shapely.geometry import LineString, Polygon

from pyorc.plot_helpers import plot_3d_line, plot_3d_polygon, plot_line


@pytest.fixture()
def sample_3d_polygon():
    # Create a simple 3D polygon
    return Polygon([(0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 1, 0), (0, 0, 0)])


@pytest.fixture()
def sample_polygon():
    # Create a simple 3D polygon
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])


@pytest.fixture()
def sample_line():
    return LineString([(0, 0), (1, 1)])


@pytest.fixture()
def sample_3d_line():
    return LineString([(0, 0, 0), (1, 1, 1)])


def test_plot_3d_polygon_creates_polycollection(sample_3d_polygon):
    ax = plt.figure().add_subplot(projection="3d")
    result = plot_3d_polygon(sample_3d_polygon, ax=ax)
    assert isinstance(result, Poly3DCollection), "Returned object is not a Poly3DCollection"


def test_plot_3d_polygon_no_ax(sample_3d_polygon):
    result = plot_3d_polygon(sample_3d_polygon)
    assert isinstance(result, Poly3DCollection), "Returned object is not a Poly3DCollection when ax is None"


def test_plot_3d_polygon_custom_kwargs(sample_3d_polygon):
    ax = plt.figure().add_subplot(projection="3d")
    result = plot_3d_polygon(sample_3d_polygon, ax=ax, alpha=0.5, edgecolor="r")
    assert np.allclose(result.get_edgecolor()[0], np.array([1.0, 0.0, 0.0, 0.5])), "Edge color not set correctly"


def test_plot_3d_line(sample_3d_line):
    ax = plt.figure().add_subplot(projection="3d")
    result = plot_3d_line(sample_3d_line, ax=ax)
    assert isinstance(result, list), "returned object is not a list"
    assert isinstance(result[0], Line3D), "object should contain Line3D element"


def test_plot_3d_line_no_ax(sample_3d_line):
    result = plot_3d_line(sample_3d_line)
    assert isinstance(result, list), "returned object is not a list"
    assert isinstance(result[0], Line3D), "object should contain Line3D element"


def test_plot_line(sample_line):
    ax = plt.figure().add_subplot()
    result = plot_line(sample_line, ax=ax)
    assert isinstance(result, list), "returned object is not a list"
    assert isinstance(result[0], Line2D), "object should contain Line3D element"


def test_plot_line_no_ax(sample_line):
    result = plot_line(sample_line)
    assert isinstance(result, list), "returned object is not a list"
    assert isinstance(result[0], Line2D), "object should contain Line3D element"
