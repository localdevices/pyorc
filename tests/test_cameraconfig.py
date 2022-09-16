import numpy as np
import os
import pyorc

from pyorc import helpers
from shapely.geometry import Polygon
from rasterio import Affine
import shapely

def test_str(cam_config):
    assert(isinstance(cam_config.__str__(), str))


def test_repr(cam_config):
    assert(isinstance(cam_config.__repr__(), str))

def test_bbox(cam_config):
    bbox = shapely.wkt.loads(cam_config.bbox)
    assert(isinstance(bbox, Polygon))


def test_shape(cam_config):
    assert(cam_config.shape == (840, 1100))


def test_transform(cam_config):
    assert(cam_config.transform == Affine(
        0.0008304547988191776,
        0.009965457582425315,
        642730.15387931,
        0.009965457582425315,
        -0.0008304547988191773,
        8304292.596551724
    ))


def test_get_depth(cam_config, cross_section, h_a):
    z = cross_section["z"]
    depth = cam_config.get_depth(z, h_a=h_a)
    assert(np.allclose(depth.values, np.array([
        0., 0.133, 0.167, 0.2, 0.167, 0.133, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.133, 0.167, 0.2, 0.25, 0.3, 0.267, 0.233,
        0.2, 0.2, 0.2, 0.05, 0., 0., 0., 0., 0.
    ])))


def test_z_to_h(cam_config, cross_section):
    z = cross_section["z"]
    h = cam_config.z_to_h(z)
    assert(np.allclose(h.values, np.array([
        0.1, -0.133, -0.167, -0.2, -0.167, -0.133, -0.1, -0.1,
        -0.1, -0.1, -0.1, -0.133, -0.167, -0.2, -0.25, -0.3,
        -0.267, -0.233, -0.2, -0.2, -0.2, -0.05, 0.1, 0.175,
        0.25, 0.325, 0.4
    ])))


def test_get_M(cam_config, h_a):
    M = cam_config.get_M(h_a=h_a)
    M_expected = np.array([[-4.60858728e-01, -6.35455550e-01, 1.28925099e+03],
                           [ 6.54313311e-01,  5.39379066e-02, -8.38203094e+00],
                           [-2.72899866e-04,  1.10694660e-03,  1.00000000e+00]])
    assert(np.allclose(M, M_expected))


def test_set_corners(cam_config, corners):
    # check if this works
    cam_config.set_corners(corners)
    assert(cam_config.corners==corners)


def test_set_lens_pars(cam_config, lens_pars):
    # check if this works
    cam_config.set_lens_pars(**lens_pars)
    assert(cam_config.lens_pars==lens_pars)


def test_set_gcps(cam_config, gcps):
    cam_config.set_gcps(**gcps)
    assert(cam_config.gcps==gcps)


def test_lens_position(cam_config, lens_position):
    cam_config.set_lens_position(*lens_position)
    # transform to epsg:4326 and see if the automated transform works
    assert(cam_config.lens_position==lens_position)
    x, y, z = lens_position
    x, y = helpers.xy_transform(x, y, cam_config.crs, 4326)
    cam_config.set_lens_position(x, y, z, crs=4326)
    assert(np.allclose(cam_config.lens_position, lens_position))


def test_to_dict(cam_config, cam_config_dict):
    d = cam_config.to_dict()
    assert(d==cam_config_dict)


def test_to_file(tmpdir, cam_config, cam_config_str):
    fn = os.path.join(tmpdir, "cam_config.json")
    cam_config.to_file(fn)
    with open(fn, "r") as f:
        data = f.read()
        assert(data == cam_config_str)


def test_load_camera_config(cam_config_fn, cam_config, lens_position):
    cam_config2 = pyorc.load_camera_config(cam_config_fn)
    # cam_config does not have lens position
    cam_config2.set_lens_position(*lens_position)
    cam_config2.set_lens_position(*lens_position)
    # only h_ref was different in the .json file, adapt this and then compare
    cam_config2.gcps["h_ref"] = 0.
    assert(cam_config2.gcps == cam_config.gcps)
    assert(cam_config2.lens_position == cam_config.lens_position)
    assert(cam_config2.crs == cam_config.crs)
    assert(cam_config2.corners == cam_config.corners)
    assert(cam_config2.window_size == cam_config.window_size)
    assert(cam_config2.resolution == cam_config.resolution)


def test_plot(cam_config):
    from cartopy.mpl.geoaxes import GeoAxesSubplot
    ax = cam_config.plot()
    assert(isinstance(ax, GeoAxesSubplot))