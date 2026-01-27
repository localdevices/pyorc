"""Tests for water level functionalities."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyproj import CRS
from shapely import geometry, wkt

from pyorc import CameraConfig, CrossSection, Video, plot_helpers, sample_data


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
    """Sample cross-section xyz list for Geul river (real data). Must be returned in CameraConfig set up."""
    gdf.to_crs(crs, inplace=True)
    g = gdf.geometry
    x, y, z = g.x, g.y, g.z
    return list(map(list, zip(x, y, z)))


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
def vid_file():
    return sample_data.get_hommerich_dataset()  # os.path.join(EXAMPLE_DATA_DIR, "hommerich", "20240718_162737.mp4")


@pytest.fixture()
def img(vid_file):
    vid = Video(vid_file, start_frame=0, end_frame=1)
    return vid.get_frame(0)


@pytest.fixture()
def img_rgb(vid_file):
    vid = Video(vid_file, start_frame=0, end_frame=1)
    return vid.get_frame(0, method="rgb")


@pytest.fixture()
def cs(xyz, camera_config):
    return CrossSection(camera_config=camera_config, cross_section=xyz)


def test_init_water_level(xyz, camera_config):
    cs = CrossSection(camera_config=camera_config, cross_section=xyz)
    assert isinstance(cs, CrossSection)


def test_init_water_level_from_gdf(gdf, camera_config):
    cs = CrossSection(camera_config=camera_config, cross_section=gdf)
    assert isinstance(cs, CrossSection)


def test_cs_str(cs):
    assert isinstance(cs.__str__(), str)


def test_cs_repr(cs):
    assert isinstance(cs.__repr__(), str)


def test_get_bbox_dry_wet(cs):
    # check also what happens with a double geom in wet part
    bbox_wet = cs.get_bbox_dry_or_wet(h=92.09)  # just below local peak of 92.1 in bathymetry
    bbox_dry = cs.get_bbox_dry_or_wet(h=92.09, dry=True)
    assert len(bbox_wet.geoms) == 2
    assert len(bbox_dry.geoms) == 3
    bbox_dry = cs.get_bbox_dry_or_wet(h=93.0, dry=True)
    bbox_wet = cs.get_bbox_dry_or_wet(h=93.0)
    assert isinstance(bbox_wet, geometry.MultiPolygon)
    assert isinstance(bbox_dry, geometry.MultiPolygon)
    assert len(bbox_wet.geoms) == 1
    assert len(bbox_dry.geoms) == 2
    assert bbox_wet.has_z
    assert bbox_dry.has_z
    # now retrieve with camera = True
    bbox_dry = cs.get_bbox_dry_or_wet(h=93.0, camera=True, dry=True)
    bbox_wet = cs.get_bbox_dry_or_wet(h=93.0, camera=True)
    assert isinstance(bbox_wet, geometry.MultiPolygon)
    assert isinstance(bbox_dry, geometry.MultiPolygon)
    assert bbox_wet.has_z == False
    assert bbox_dry.has_z == False


def test_get_cs_waterlevel(cs):
    line = cs.get_cs_waterlevel(h=93.0)
    assert isinstance(line, geometry.LineString)
    assert line.has_z
    # also try with extend
    line_extend = cs.get_cs_waterlevel(h=93.0, extend_by=0.2)
    assert np.isclose(line_extend.length - line.length, 0.2 * 2)


def test_get_cs_waterlevel_sz(cs):
    line = cs.get_cs_waterlevel(h=93.0, sz=True)
    assert isinstance(line, geometry.LineString)
    assert line.has_z == False
    line_extend = cs.get_cs_waterlevel(h=93.0, sz=True, extend_by=0.2)
    assert np.isclose(line_extend.length - line.length, 0.2 * 2)


def test_get_csl_point(cs):
    h1 = 92.5
    h2 = 93.0
    # both should get two points back as we are seeking at a vertical level that crosses twice
    p1 = cs.get_csl_point(h=h1)
    p2 = cs.get_csl_point(h=h2)
    assert len(p1) == 2
    assert len(p2) == 2
    # points must be 3d
    assert p1[0].has_z
    assert p2[0].has_z


def test_get_csl_point_camera(cs):
    h1 = 92.5
    h2 = 93.0
    # both should get two points back as we are seeking at a vertical level that crosses twice
    p1 = cs.get_csl_point(h=h1, camera=True)
    p2 = cs.get_csl_point(h=h2, camera=True)
    assert len(p1) == 2
    assert len(p2) == 2


def test_get_csl_point_l(cs):
    l1 = 5.0
    l2 = 8.0
    # both should get one point back as we are seeking a certain point left to right
    p1 = cs.get_csl_point(l=l1)
    p2 = cs.get_csl_point(l=l2)
    assert len(p1) == 1
    assert len(p2) == 1


def test_get_csl_point_no_h_l(cs):
    with pytest.raises(ValueError, match="One of h or l"):
        cs.get_csl_point()


def test_get_csl_point_both_h_l(cs):
    with pytest.raises(ValueError, match="Only one of h or l"):
        cs.get_csl_point(h=93.0, l=5.0)


def test_get_csl_line(cs):
    h1 = 92.5
    h2 = 93.0
    cross1 = cs.get_csl_line(h=h1, offset=0.0, length=4)
    cross2 = cs.get_csl_line(h=h2, offset=0.0, length=4)
    assert len(cross1) == 2
    assert len(cross2) == 2


def test_get_csl_line_s(cs):
    l1 = 5.0
    l2 = 8.0
    cross1 = cs.get_csl_line(l=l1, offset=0.0, length=4)
    cross2 = cs.get_csl_line(l=l2, offset=0.0, length=4)
    assert len(cross1) == 1
    assert len(cross2) == 1


def test_get_csl_camera(cs):
    h1 = 92.5
    cross1 = cs.get_csl_line(h=h1, offset=2.0, camera=True)
    cross2 = cs.get_csl_line(h=h1, offset=0.0, camera=True)
    assert len(cross1) == 2
    assert len(cross2) == 2


def test_get_csl_line_above_first_bank(cs):
    h = 94.9  # this level is above one of the banks and should return only one line
    cross = cs.get_csl_line(h=h)
    # check length
    assert len(cross) == 1


def test_get_csl_pol(cs):
    h1 = 93.25
    pol1 = cs.get_csl_pol(h=h1, offset=0.0, padding=(-2, 0), length=4.0)
    pol2 = cs.get_csl_pol(h=h1, offset=0.0, padding=(0, 2), length=4.0)
    assert isinstance(pol1, list)
    assert isinstance(pol1, list)
    assert isinstance(pol1[0], geometry.Polygon)
    assert isinstance(pol2[0], geometry.Polygon)
    assert pol1[0].has_z
    assert pol2[0].has_z


def test_get_csl_pol_camera(cs):
    h1 = 93.25
    pol1 = cs.get_csl_pol(h=h1, offset=0.0, padding=(-2, 0), length=4.0, camera=True)
    pol2 = cs.get_csl_pol(h=h1, offset=0.0, padding=(0, 2), length=4.0, camera=True)
    assert isinstance(pol1, list)
    assert isinstance(pol1, list)
    assert isinstance(pol1[0], geometry.Polygon)
    assert isinstance(pol2[0], geometry.Polygon)


def test_get_planar_surface(cs):
    h1 = 92.5
    h2 = 93.0
    h3 = 94.9
    _ = cs.get_planar_surface(h=h1, length=10.0)
    __ = cs.get_planar_surface(h=h2, length=5, offset=2.5)
    with pytest.raises(ValueError, match="must have at least two"):
        cs.get_planar_surface(h=h3)


def test_get_planar_surface_camera(cs):
    h1 = 92.5
    h2 = 93.0
    _ = cs.get_planar_surface(h=h1, length=10.0, camera=True)
    __ = cs.get_planar_surface(h=h2, length=5, offset=2.5, camera=True)
    ax = plt.axes()
    _ = plot_helpers.plot_polygon(_, ax=ax, alpha=0.3, label="h=92.5", color="r")
    _ = plot_helpers.plot_polygon(__, ax=ax, alpha=0.3, label="h=93.0", color="g")
    ax.axis("equal")
    assert ax.has_data()


def test_get_wetted_surface(cs):
    h1 = 92.5
    h2 = 93.0
    h3 = 94.9
    pol1 = cs.get_wetted_surface(h=h1)
    pol2 = cs.get_wetted_surface(h=h2)
    assert isinstance(pol1, geometry.MultiPolygon)
    assert pol1.has_z
    assert pol2.has_z

    # h3 is above cross section, should still resolve one polygon
    pol3 = cs.get_wetted_surface(h=h3)
    assert isinstance(pol3, geometry.MultiPolygon)
    assert len(pol3.geoms) == 1


def test_get_wetted_surface_sz_perimeter(cs):
    """Test get_wetted_surface with `perimeter=True`."""
    line1 = cs.get_wetted_surface_sz(h=93.0, perimeter=True)
    line2 = cs.get_wetted_surface_sz(h=94.9, perimeter=True)

    assert isinstance(line1, geometry.MultiLineString)
    assert line1.has_z == False
    assert isinstance(line2, geometry.MultiLineString)
    assert line2.has_z == False
    assert line2.length > line1.length


def test_detect_wl(cs, img):
    h = cs.detect_water_level(img, bank="far", length=1.0)
    print(f"Water level: {h}")
    assert h is not None
    assert isinstance(h, float)


def test_detect_wl_min_h(cs, img, min_h=93.3):
    h = cs.detect_water_level(img, bank="far", length=1.0, min_h=min_h)
    print(f"Water level near bank: {h}")
    assert h is not None, "h must have valid value"
    assert h >= min_h, f"h must be above or equal to min_h {h} < {min_h}"
    assert isinstance(h, float)


def test_detect_wl_near_bank(cs, img):
    h = cs.detect_water_level(img, bank="near", length=2.0)
    print(f"Water level near bank: {h}")
    assert h is not None
    assert isinstance(h, float)


def test_detect_wl_no_bank_specified(cs, img, bank="both"):
    h = cs.detect_water_level(img, length=2.0, bank=bank)
    print(f"Water level with both banks specified: {h}")


def test_water_level_score_range(cs, img):
    h, s2n = cs.detect_water_level_s2n(img, bank="far", length=1.0)
    assert h is not None
    assert h > 93.0
    assert s2n > 8
    assert isinstance(h, float)


def test_plot_cs_camera(cs):
    ax = cs.plot(h=93, camera=True)
    cs.plot_cs(ax=ax, camera=True, marker=".", color="c", label="cross section")
    ax.set_xlim([0, cs.camera_config.width])
    ax.set_ylim([0, cs.camera_config.height])
    assert ax.has_data()


def test_plot_cs(cs):
    ax = plt.axes(projection="3d")
    cs.plot_cs(ax=ax, camera=False, marker=".", color="c", label="cross section")
    cs.plot(h=93, ax=ax, camera=False)
    assert ax.has_data()


def test_plot_no_h_no_ax(cs):
    ax = cs.plot()
    assert ax.has_data()


def test_plot_water_level_camera(cs):
    p = cs.plot_cs(camera=True, marker=".", color="c", label="cross section")
    ax = p[0].axes
    cs.plot_water_level(h=93.0, length=2.0, ax=ax, camera=True, color="r", label="water level")
    assert ax.has_data()


def test_plot_water_level(cs):
    ax = plt.axes(projection="3d")
    cs.plot_water_level(h=93.0, ax=ax, camera=False, color="c", label="water level")
    # plt.show()
    assert ax.has_data()


def test_plot_bbox(cs):
    ax = plt.axes(projection="3d")
    cs.plot(ax=ax)
    cs.plot_bbox_dry_wet(h=92.09, ax=ax)
    # ax.legend()
    # plt.show()
    assert ax.has_data()


def test_plot_bbox_camera(cs):
    ax = plt.axes()
    cs.plot(ax=ax, camera=True, swap_y_coords=True)
    cs.plot_bbox_dry_wet(h=92.09, camera=True, ax=ax, swap_y_coords=True)
    # ax.legend()
    # plt.show()
    assert ax.has_data()


def test_rotate_translate(cs):
    cs2 = cs.rotate_translate(angle=0, xoff=10)
    assert np.allclose(cs.y, cs2.y)  # y-coordinates should not have changed
    assert not np.allclose(cs.x, cs2.x)
    cs3 = cs.rotate_translate(angle=20 / 180 * np.pi, xoff=10)
    # check location of centroid. Should be same as cs2
    assert np.isclose(cs2.cs_linestring.centroid.x, cs3.cs_linestring.centroid.x)
    assert np.isclose(cs2.cs_linestring.centroid.y, cs3.cs_linestring.centroid.y)
    cs4 = cs.rotate_translate(angle=0, zoff=10)
    assert not np.allclose(cs.z, cs4.z)
    assert np.allclose(cs.x, cs4.x)
    assert np.allclose(cs.y, cs4.y)
    assert np.allclose(cs.z, cs4.z - 10)
