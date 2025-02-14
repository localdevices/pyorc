import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyorc.plot_helpers import plot_3d_polygon
from shapely.geometry import Polygon


@pytest.fixture
def sample_polygon():
    # Create a simple 3D polygon
    return Polygon([(0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 1, 0), (0, 0, 0)])


def test_plot_3d_polygon_creates_polycollection(sample_polygon):
    ax = plt.figure().add_subplot(projection="3d")
    result = plot_3d_polygon(sample_polygon, ax=ax)
    assert isinstance(result, Poly3DCollection), "Returned object is not a Poly3DCollection"


def test_plot_3d_polygon_no_ax(sample_polygon):
    result = plot_3d_polygon(sample_polygon)
    assert isinstance(result, Poly3DCollection), "Returned object is not a Poly3DCollection when ax is None"


def test_plot_3d_polygon_custom_kwargs(sample_polygon):
    ax = plt.figure().add_subplot(projection="3d")
    result = plot_3d_polygon(sample_polygon, ax=ax, alpha=0.5, edgecolor='r')
    assert np.allclose(result.get_edgecolor()[0], np.array([1.0, 0.0, 0.0, 0.5])), "Edge color not set correctly"
