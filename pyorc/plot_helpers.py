"""pyorc plot helpers functions."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_polygon(polygon, ax=None, **kwargs):
    """Plot a shapely.geometry.Polygon on matplotlib 3d ax."""
    x, y, z = zip(*polygon.exterior.coords)
    verts = [list(zip(x, y, z))]

    if ax is None:
        ax = plt.axes(projection="3d")
    poly = Poly3DCollection(verts, **kwargs)
    # p = ax.plot_trisurf(x, y, z, **kwargs)
    p = ax.add_collection3d(poly)
    return p


def plot_polygon(polygon, ax=None, **kwargs):
    """Plot a shapely.geometry.Polygon on matplotlib ax."""
    # x, y = zip(*polygon.exterior.coords)
    if ax is None:
        ax = plt.axes()
    patch = plt.Polygon(polygon.exterior.coords, **kwargs)
    p = ax.add_patch(patch)
    return p


def plot_3d_line(line, ax=None, **kwargs):
    """Plot a shapely.geometry.LineString on matplotlib ax."""
    if ax is None:
        ax = plt.axes(projection="3d")
    x, y, z = zip(*line.coords)
    p = ax.plot(x, y, z, **kwargs)
    return p


def plot_line(line, ax=None, **kwargs):
    """Plot a shapely.geometry.LineString on matplotlib ax."""
    if ax is None:
        ax = plt.axes()
    x, y = zip(*line.coords)
    p = ax.plot(x, y, **kwargs)
    return p
