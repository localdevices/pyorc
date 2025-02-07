"""Tests for water level functionalities."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyproj import CRS
from shapely import geometry, wkt

from pyorc import CameraConfig, CrossSection, plot_helpers


@pytest.fixture()
def zs():
    return [
        152.754,
        152.436,
        152.124,
        151.65,
        151.171,
        150.959,
        150.689,
        150.215,
        150.227,
        150.204,
        150.148,
        150.181,
        150.114,
        150.14,
        150.096,
        150.207,
        150.474,
        150.684,
        150.931,
        151.136,
        151.558,
        151.943,
        152.711,
        153.016,
    ]


@pytest.fixture()
def xs():
    return [
        5.913483043333334,
        5.91350165,
        5.913509225,
        5.913517873333333,
        5.913526728333333,
        5.913537678333333,
        5.913544631666667,
        5.913551016666665,
        5.91356275,
        5.913577963333334,
        5.913591855,
        5.913605991666667,
        5.91362158,
        5.91362959,
        5.913639568333333,
        5.913647405,
        5.913650936666666,
        5.91365698,
        5.913666071666667,
        5.913672016666667,
        5.913678495,
        5.91368494,
        5.913693873333334,
        5.913725518333333,
    ]


@pytest.fixture()
def ys():
    return [
        50.807081403333335,
        50.80708851833334,
        50.80709163333333,
        50.807093645,
        50.807096580000014,
        50.807099555,
        50.807102958333346,
        50.80710621,
        50.80710916,
        50.807112763333336,
        50.80711691833334,
        50.807121985,
        50.80712629833334,
        50.807129086666656,
        50.807132803333324,
        50.80713549666667,
        50.807136676666666,
        50.807138608333325,
        50.80714141666667,
        50.80714368666667,
        50.80714608333333,
        50.80714834333333,
        50.80715788,
        50.807162983333335,
    ]


@pytest.fixture()
def crs():
    """Sample CRS for Netherlands."""
    return CRS.from_user_input(28992)


@pytest.fixture()
def gdf(xs, ys, zs):
    """Sample cross section GeoDataFrame for Geul river (real data)."""
    geometry = gpd.points_from_xy(xs, ys, zs)
    crs = CRS.from_user_input(4326)  # latlon
    return gpd.GeoDataFrame({"id": np.arange(len(xs))}, geometry=geometry, crs=crs)


@pytest.fixture()
def xyz(gdf, crs):
    """Sample cross section xyz list for Geul river (real data). Must be returned in CameraConfig set up."""
    gdf.to_crs(crs, inplace=True)
    g = gdf.geometry
    x, y, z = g.x, g.y, g.z
    return list(map(list, zip(x, y, z, strict=False)))


@pytest.fixture()
def camera_config():
    camera_config = {
        "height": 1080,
        "width": 1920,
        "crs": CRS.from_user_input(28992),
        "resolution": 0.01,
        "gcps": {
            "src": [[158, 314], [418, 245], [655, 162], [948, 98], [1587, 321], [1465, 747]],
            "dst": [
                [192102.50255553858, 313157.5882846481, 150.831],
                [192101.3882378415, 313160.1101843005, 150.717],
                [192099.77023223988, 313163.2868999007, 150.807],
                [192096.8922817797, 313169.2557434712, 150.621],
                [192105.2958125107, 313172.0257530752, 150.616],
                [192110.35620407888, 313162.5371485311, 150.758],
            ],
            "h_ref": 92.45,
            "z_0": 150.49,
        },
        "window_size": 64,
        "is_nadir": False,
        "camera_matrix": [[1750.3084716796875, 0.0, 960.0], [0.0, 1750.3084716796875, 540.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [[-0.48456448702008914], [0.44089348828121366], [0.0], [0.0], [0.0]],
        "bbox": wkt.loads(
            "POLYGON ((192102.55970673775 313154.1397356759, 192098.0727491934 313163.2664060433, 192108.81475944887"
            " 313168.5475153654, 192113.3017169932 313159.420844998, 192102.55970673775 313154.1397356759))"
        ),
    }
    return CameraConfig(**camera_config)


@pytest.fixture()
def cs(xyz, camera_config):
    return CrossSection(camera_config=camera_config, cross_section=xyz)


def test_init_water_level(xyz, camera_config):
    cs = CrossSection(camera_config=camera_config, cross_section=xyz)
    assert isinstance(cs, CrossSection)


def test_init_water_level_from_gdf(gdf, camera_config):
    cs = CrossSection(camera_config=camera_config, cross_section=gdf)
    assert isinstance(cs, CrossSection)

    # get coordinates


def test_get_csl_point(cs):
    h1 = 92.5
    h2 = 93.0
    # both should get two points back
    cross1 = cs.get_csl_point(h=h1)
    cross2 = cs.get_csl_point(h=h2)
    ax = plt.axes(projection="3d")

    for cross in cross1:
        ax.plot(*cross.coords[0], "bo", label="cross 1")
    for cross in cross2:
        ax.plot(*cross.coords[0], "ro", label="cross 2")
    cs.plot_cs(ax=ax, marker=".", color="c")
    ax.legend()
    plt.show()


def test_get_csl_point_camera(cs):
    h1 = 92.5
    h2 = 93.0
    # both should get two points back
    cross1 = cs.get_csl_point(h=h1, camera=True)
    cross2 = cs.get_csl_point(h=h2, camera=True)
    ax = plt.axes()

    for cross in cross1:
        ax.plot(*cross.coords[0], "bo")
    for cross in cross2:
        ax.plot(*cross.coords[0], "ro")
    cs.plot_cs(ax=ax, camera=True)
    ax.axis("equal")
    ax.set_xlim([0, cs.camera_config.width])
    ax.set_ylim([0, cs.camera_config.height])
    plt.show()


def test_get_csl_point_s(cs):
    s1 = 5.0
    s2 = 8.0
    # both should get two points back
    cross1 = cs.get_csl_point(s=s1)
    cross2 = cs.get_csl_point(s=s2)
    ax = plt.axes(projection="3d")

    for cross in cross1:
        ax.plot(*cross.coords[0], "bo", label="cross 1")
    for cross in cross2:
        ax.plot(*cross.coords[0], "ro", label="cross 2")
    cs.plot_cs(ax=ax, marker=".", color="c")
    ax.legend()
    plt.show()



def test_get_csl_line(cs):
    h1 = 92.5
    h2 = 93.0

    cross1 = cs.get_csl_line(h=h1, offset=0.0, length=4)
    cross2 = cs.get_csl_line(h=h2, offset=0.0, length=4)
    assert len(cross1) == 2
    assert len(cross2) == 2

    ax = plt.axes(projection="3d")
    for cross in cross1:
        ax.plot(*cross.xy, cs.camera_config.h_to_z(h1), "b-o", label="cross 1")
    for cross in cross2:
        ax.plot(*cross.xy, cs.camera_config.h_to_z(h2), "r-o", label="cross 2")
    cs.plot_cs(ax=ax, marker=".", color="c")
    ax.axis("equal")
    ax.legend()
    plt.show()

def test_get_csl_line_s(cs):
    s1 = 5.0
    s2 = 8.0

    cross1 = cs.get_csl_line(s=s1, offset=0.0, length=4)
    cross2 = cs.get_csl_line(s=s2, offset=0.0, length=4)
    assert len(cross1) == 1
    assert len(cross2) == 1

    ax = plt.axes(projection="3d")
    for cross in cross1:
        ax.plot(*cross.xy, cross.coords[0][-1], "b-o", label="cross 1")
    for cross in cross2:
        ax.plot(*cross.xy, cross.coords[0][-1], "r-o", label="cross 2")
    cs.plot_cs(ax=ax, marker=".", color="c")
    ax.axis("equal")
    ax.legend()
    plt.show()


def test_get_csl_camera(cs):
    h1 = 92.5

    cross1 = cs.get_csl_line(h=h1, offset=2.0, camera=True)
    cross2 = cs.get_csl_line(h=h1, offset=0.0, camera=True)
    assert len(cross1) == 2
    assert len(cross2) == 2
    ax = plt.axes()
    for cross in cross1:
        ax.plot(*cross.xy, "b-o", label="cross 1")
    for cross in cross2:
        ax.plot(*cross.xy, "r-o", label="cross 2")
    cs.plot_cs(ax=ax, camera=True, marker=".", color="c")
    ax.axis("equal")
    ax.set_xlim([0, cs.camera_config.width])
    ax.set_ylim([0, cs.camera_config.height])
    ax.legend()
    plt.show()


def test_get_csl_line_above_first_bank(cs):
    h = 94.9  # this level is above one of the banks and should return only one line
    cross = cs.get_csl_line(h=h)
    # check length
    assert len(cross) == 1


def test_get_csl_pol(cs):
    h1 = 92.5
    pols1 = cs.get_csl_pol(h=h1, offset=0.0, padding=(-2, 0), length=4.0)
    pols2 = cs.get_csl_pol(h=h1, offset=0.0, padding=(0, 2), length=4.0)

    ax = plt.axes(projection="3d")
    cs.plot_cs(ax=ax, marker=".", color="c", label="cross section")
    p1_1, p_1_2 = [plot_helpers.plot_3d_polygon(pol, ax=ax, alpha=0.3, label="h=92.5", color="r") for pol in pols1]
    p2_1, p_2_2 = [plot_helpers.plot_3d_polygon(pol, ax=ax, alpha=0.3, label="h=93.0", color="g") for pol in pols2]
    ax.axis("equal")
    ax.legend()
    plt.show()

def test_get_csl_pol(cs):
    s1 = 5.0
    pols1 = cs.get_csl_pol(s=s1, offset=0.0, padding=(-2, 0), length=4.0)
    pols2 = cs.get_csl_pol(s=s1, offset=0.0, padding=(0, 2), length=4.0)

    ax = plt.axes(projection="3d")
    cs.plot_cs(ax=ax, marker=".", color="c", label="cross section")
    p1 = [plot_helpers.plot_3d_polygon(pol, ax=ax, alpha=0.3, label="h=92.5", color="r") for pol in pols1]
    p2 = [plot_helpers.plot_3d_polygon(pol, ax=ax, alpha=0.3, label="h=93.0", color="g") for pol in pols2]
    ax.axis("equal")
    ax.legend()
    plt.show()


def test_get_csl_pol_camera(cs):
    h1 = 92.5
    pols1 = cs.get_csl_pol(h=h1, offset=0.0, padding=(-2, 0), length=4.0, camera=True)
    pols2 = cs.get_csl_pol(h=h1, offset=0.0, padding=(0, 2), length=4.0, camera=True)

    ax = plt.axes()
    cs.plot_cs(ax=ax, marker=".", color="c", label="cross section", camera=True)
    p1_1, p_1_2 = [plot_helpers.plot_polygon(pol, ax=ax, alpha=0.3, label="h=92.5", color="r") for pol in pols1]
    p2_1, p_2_2 = [plot_helpers.plot_polygon(pol, ax=ax, alpha=0.3, label="h=93.0", color="g") for pol in pols2]
    ax.axis("equal")
    ax.set_xlabel("Camera pixel column [-]")
    ax.set_ylabel("Camera pixel row [-]")
    ax.set_xlim([0, cs.camera_config.width])
    ax.set_ylim([0, cs.camera_config.height])
    ax.legend()
    plt.show()


def test_get_planar_surface(cs):
    h1 = 92.5
    h2 = 93.0
    h3 = 94.9
    pol1 = cs.get_planar_surface(h=h1, length=10.0)
    pol2 = cs.get_planar_surface(h=h2, length=5, offset=2.5)
    with pytest.raises(ValueError, match="must be 2 for"):
        cs.get_planar_surface(h=h3)
    ax = plt.axes(projection="3d")
    cs.plot_cs(ax=ax, marker=".", color="c", label="cross section")

    _ = plot_helpers.plot_3d_polygon(pol1, ax=ax, alpha=0.3, label="h=92.5", color="r")
    _ = plot_helpers.plot_3d_polygon(pol2, ax=ax, alpha=0.3, label="h=93.0", color="g")
    ax.axis("equal")
    ax.legend()
    plt.show()


def test_get_planar_surface_camera(cs):
    h1 = 92.5
    h2 = 93.0
    pol1 = cs.get_planar_surface(h=h1, length=10.0, camera=True)
    pol2 = cs.get_planar_surface(h=h2, length=5, offset=2.5, camera=True)
    ax = plt.axes()
    _ = plot_helpers.plot_polygon(pol1, ax=ax, alpha=0.3, label="h=92.5", color="r")
    _ = plot_helpers.plot_polygon(pol2, ax=ax, alpha=0.3, label="h=93.0", color="g")
    ax.axis("equal")
    cs.plot_cs(ax=ax, camera=True, marker=".", color="c", label="cross section")
    ax.set_xlim([0, cs.camera_config.width])
    ax.set_ylim([0, cs.camera_config.height])
    ax.legend()
    plt.show()


def test_get_wetted_surface(cs):
    h1 = 92.5
    h2 = 93.0
    h3 = 94.9
    pol1 = cs.get_wetted_surface(h=h1)
    pol2 = cs.get_wetted_surface(h=h2)
    assert isinstance(pol1, geometry.Polygon)
    assert pol1.has_z
    assert pol2.has_z

    with pytest.raises(ValueError, match="Water level is not crossed"):
        cs.get_wetted_surface(h=h3)

    ax = plt.axes(projection="3d")
    cs.plot_cs(ax=ax, marker=".", color="c", label="cross section")
    _ = plot_helpers.plot_3d_polygon(pol1, ax=ax, alpha=0.3, label="h=92.5", color="r")
    _ = plot_helpers.plot_3d_polygon(pol2, ax=ax, alpha=0.3, label="h=93.0", color="g")
    ax.axis("equal")
    ax.legend()
    plt.show()
