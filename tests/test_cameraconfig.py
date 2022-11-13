import numpy as np
import os
import pyorc
import pytest

from cartopy.mpl.geoaxes import GeoAxesSubplot
from pyorc import helpers, cv

from shapely.geometry import Polygon
from rasterio import Affine
import shapely

def test_str(cam_config):
    assert(isinstance(cam_config.__str__(), str))


def test_repr(cam_config):
    assert(isinstance(cam_config.__repr__(), str))

def test_bbox(cam_config):
    bbox = cam_config.bbox
    assert(isinstance(bbox, Polygon))


def test_gcp_mean(cam_config):
    assert(np.allclose(cam_config.gcp_mean, np.array([642734.7117 , 8304295.74875])))


def test_get_bbox(cam_config, vid):
    bbox = cam_config.get_bbox(camera=True)
    assert(isinstance(bbox, Polygon))


def test_shape(cam_config):
    assert(cam_config.shape == (821, 965))


def test_transform(cam_config):
    assert(np.allclose(cam_config.transform, Affine(
        0.0010061044563599466, 0.009949258958479906, 642730.3058131004,
        0.009949258958479906, -0.0010061044563599508, 8304292.867164782
    )))


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

@pytest.mark.parametrize(
    "to_bbox_grid, M_expected",
    [
        (
            True, np.array(
                [
                    [-4.39245097e-01, -6.69287523e-01, 1.26128097e+03],
                    [6.69392774e-01, 4.50461138e-02, -4.12599188e+01],
                    [-2.65193791e-04, 1.09679892e-03, 1.00000000e+00]
                ]
            )
        ),
        (
            False, np.array(
                [
                    [ 7.38644979e-03, -5.05757015e-03, -3.54741235e+00],
                    [-4.27946398e-03, -9.86475823e-03,  9.70873926e+00],
                    [-2.65193855e-04, 1.09679938e-03, 1.00000000e+00]
                ]
            )
        )
    ]
)
def test_get_M(cam_config, h_a, to_bbox_grid, M_expected):
    M = cam_config.get_M(h_a=h_a, to_bbox_grid=to_bbox_grid)
    assert(np.allclose(M, M_expected))


def test_set_bbox_from_corners(cam_config, corners, bbox):
    # check if this works
    cam_config.set_bbox_from_corners(corners)
    assert(cam_config.bbox==bbox)


def test_set_lens_pars(cam_config, lens_pars, camera_matrix, dist_coeffs):
    # check if this works
    cam_config.set_lens_pars(**lens_pars)

    assert(np.allclose(cam_config.camera_matrix, camera_matrix))
    assert(np.allclose(cam_config.dist_coeffs, dist_coeffs))



def test_set_gcps(cam_config, gcps):
    cam_config.set_gcps(**gcps)
    assert(cam_config.gcps==gcps)


def test_lens_position(cam_config, lens_position):
    cam_config.set_lens_position(*lens_position)
    # transform to epsg:4326 and see if the automated transform works
    assert(cam_config.lens_position==lens_position)
    x, y, z = lens_position
    x, y = helpers.xyz_transform([[x, y]], cam_config.crs, 4326)[0]
    cam_config.set_lens_position(x, y, z, crs=4326)
    assert(np.allclose(cam_config.lens_position, lens_position))


def test_to_dict(cam_config, cam_config_dict):
    d = cam_config.to_dict()
    # ensure to only compare list or string-like things.
    del d["bbox"]
    del d["dist_coeffs"]
    del d["camera_matrix"]
    assert(d==cam_config_dict)


def test_to_file(tmpdir, cam_config, cam_config_str):
    fn = os.path.join(tmpdir, "cam_config.json")
    cam_config.to_file(fn)
    # now test if reading the file yields the same cam_config
    cam_config2 = pyorc.load_camera_config(fn)
    assert(cam_config.to_dict() == cam_config2.to_dict())

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
    assert(cam_config2.window_size == cam_config.window_size)
    assert(cam_config2.resolution == cam_config.resolution)


@pytest.mark.parametrize(
    "camera",
    [
        True,
        False
    ]
)
def test_plot(cam_config, vid, camera):
    import matplotlib
    ax = cam_config.plot(camera=camera)
    if camera:
        assert(
            isinstance(ax, matplotlib.axes._subplots.Subplot)
        )
    else:
        assert(
            isinstance(ax, GeoAxesSubplot)
        )


def test_cv_undistort_points(cam_config):
    # let's fake a distortion
    cam_config.dist_coeffs[0][0] = -3e-3
    src = cam_config.gcps["src"]
    mtx = cam_config.camera_matrix
    dist = cam_config.dist_coeffs
    src_undist = cv.undistort_points(src, mtx, dist)
    src_back_dist = cv.undistort_points(src, mtx, dist, reverse=True)
    # check if points are back to originals after back adn forth undistortion and distortion
    assert(np.allclose(src, src_back_dist))


def test_camera_calib(cam_config_calib, calib_video):
    cam_config_calib.set_lens_calibration(calib_video, max_imgs=5, plot=False, progress_bar=False)
