"""Tests for water level functionalities."""

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely import wkt

from pyorc import CameraConfig, WaterLevel


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


def test_init_water_level(xyz, camera_config):
    wl = WaterLevel(camera_config=camera_config, cross_section=xyz)
    assert isinstance(wl, WaterLevel)

    # get coordinates
