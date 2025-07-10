"""pyorc plot helpers functions."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely import geometry


def _plot_3d_pol(polygon, ax=None, **kwargs):
    """Plot single polygon on matplotlib 3d ax."""
    x, y, z = zip(*polygon.exterior.coords)
    verts = [list(zip(x, y, z))]

    if ax is None:
        ax = plt.axes(projection="3d")
    poly = Poly3DCollection(verts, **kwargs)
    # p = ax.plot_trisurf(x, y, z, **kwargs)
    p = ax.add_collection3d(poly)
    return p


def plot_3d_polygon(polygon, ax=None, **kwargs):
    """Plot a shapely.geometry.Polygon or MultiPolygon on matplotlib 3d ax."""
    if isinstance(polygon, geometry.MultiPolygon):
        for pol in polygon.geoms:
            p = _plot_3d_pol(pol, ax=ax, **kwargs)
    else:
        p = _plot_3d_pol(polygon, ax=ax, **kwargs)
    return p


def plot_polygon(polygon, ax=None, **kwargs):
    """Plot a shapely.geometry.Polygon or MultiPolygon on matplotlib ax."""
    if ax is None:
        ax = plt.axes()
    if isinstance(polygon, geometry.MultiPolygon):
        for pol in polygon.geoms:
            patch = plt.Polygon(pol.exterior.coords, **kwargs)
            p = ax.add_patch(patch)
    else:
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
