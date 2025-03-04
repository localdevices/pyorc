import os

import numpy as np
import pytest
from rasterio import Affine
from shapely.geometry import Polygon

import pyorc

# from cartopy.mpl.geoaxes import GeoAxesSubplot
from pyorc import cv, helpers


def test_str(cam_config):
    assert isinstance(cam_config.__str__(), str)


def test_repr(cam_config):
    assert isinstance(cam_config.__repr__(), str)


def test_bbox(cam_config):
    bbox = cam_config.bbox
    assert isinstance(bbox, Polygon)


def test_gcp_mean(cam_config):
    assert np.allclose(cam_config.gcps_mean, np.array([642734.7117, 8304295.74875, 1182.2]))


@pytest.mark.parametrize(("mode", "has_z"), [("geographic", False), ("camera", False), ("3d", True)])
def test_get_bbox(cam_config, vid, mode, has_z):
    bbox = cam_config.get_bbox(mode=mode)
    assert isinstance(bbox, Polygon)
    assert bbox.has_z == has_z


def test_shape(cam_config):
    assert cam_config.shape == (475, 371)


def test_transform(cam_config):
    assert np.allclose(
        cam_config.transform,
        Affine(
            -0.001107604584241635,
            0.009938471315296278,
            642732.3625957984,
            0.009938471315296278,
            0.001107604584241631,
            8304293.51724592,
        ),
    )


def test_get_depth(cam_config, cross_section, h_a):
    z = cross_section["z"]
    depth = cam_config.get_depth(z, h_a=h_a)
    assert np.allclose(
        depth.values,
        np.array(
            [
                0.0,
                0.133,
                0.167,
                0.2,
                0.167,
                0.133,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.133,
                0.167,
                0.2,
                0.25,
                0.3,
                0.267,
                0.233,
                0.2,
                0.2,
                0.2,
                0.05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


def test_z_to_h(cam_config, cross_section):
    z = cross_section["z"]
    h = cam_config.z_to_h(z)
    assert np.allclose(
        h.values,
        np.array(
            [
                0.1,
                -0.133,
                -0.167,
                -0.2,
                -0.167,
                -0.133,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.133,
                -0.167,
                -0.2,
                -0.25,
                -0.3,
                -0.267,
                -0.233,
                -0.2,
                -0.2,
                -0.2,
                -0.05,
                0.1,
                0.175,
                0.25,
                0.325,
                0.4,
            ]
        ),
    )


@pytest.mark.parametrize(
    ("to_bbox_grid", "M_expected"),
    [
        (
            True,
            np.array(
                [
                    [-4.62466994e-01, -7.62938375e-01, 8.75609302e02],
                    [6.48451357e-01, -6.15534992e-01, -2.04821521e02],
                    [-1.21275313e-04, 6.33985726e-04, 1.00000000e00],
                ]
            ),
        ),
        (
            False,
            np.array(
                [
                    [6.95684503e-03, -5.27244231e-03, -3.00544137e00],
                    [-3.87798711e-03, -8.26420874e-03, 8.47535569e00],
                    [-1.21275338e-04, 6.33985524e-04, 1.00000000e00],
                ]
            ),
        ),
    ],
)
def test_get_M(cam_config, h_a, to_bbox_grid, M_expected):
    M = cam_config.get_M(h_a=h_a, to_bbox_grid=to_bbox_grid)
    assert np.allclose(M, M_expected)


@pytest.mark.parametrize(
    ("cur_cam_config", "cur_corners", "cur_bbox"),
    [("cam_config_6gcps", "corners_6gcps", "bbox_6gcps"), ("cam_config", "corners", "bbox")],
)
def test_set_bbox_from_corners(cur_cam_config, cur_corners, cur_bbox, request):
    cur_cam_config = request.getfixturevalue(cur_cam_config)
    cur_corners = request.getfixturevalue(cur_corners)
    cur_bbox = request.getfixturevalue(cur_bbox)

    # check if this works
    cur_cam_config.set_bbox_from_corners(cur_corners)
    assert np.allclose(cur_cam_config.bbox.bounds, cur_bbox.bounds)


def test_set_lens_pars(cam_config, lens_pars, camera_matrix, dist_coeffs):
    # check if this works
    cam_config.set_lens_pars(**lens_pars)
    assert np.allclose(cam_config.camera_matrix, camera_matrix)
    assert np.allclose(cam_config.dist_coeffs, dist_coeffs)


def test_set_gcps(cam_config, gcps):
    cam_config.set_gcps(**gcps)
    assert cam_config.gcps == gcps


def test_lens_position(cam_config, lens_position):
    cam_config.set_lens_position(*lens_position)
    # transform to epsg:4326 and see if the automated transform works
    assert cam_config.lens_position == lens_position
    x, y, z = lens_position
    x, y = helpers.xyz_transform([[x, y]], cam_config.crs, 4326)[0]
    cam_config.set_lens_position(x, y, z, crs=4326)
    assert np.allclose(cam_config.lens_position, lens_position)


def test_estimate_lens_position(cam_config):
    lens_pos = cam_config.estimate_lens_position()
    assert lens_pos.dtype == np.float64
    assert np.allclose(lens_pos, [6.42731099e05, 8.30429131e06, 1.18996749e03])


def test_optimize_intrinsic(cam_config):
    camera_matrix, dist_coeffs, err = cv.optimize_intrinsic(
        cam_config.gcps["src"],
        cam_config.gcps_dest,
        cam_config.height,
        cam_config.width,
        lens_position=cam_config.lens_position,
    )
    print(camera_matrix, dist_coeffs, err)


def test_to_file(tmpdir, cam_config, cam_config_str):
    fn = os.path.join(tmpdir, "cam_config.json")
    cam_config.to_file(fn)
    # now test if reading the file yields the same cam_config
    cam_config2 = pyorc.load_camera_config(fn)
    assert cam_config.to_dict() == cam_config2.to_dict()


def test_load_camera_config(cam_config_fn, cam_config, lens_position):
    cam_config2 = pyorc.load_camera_config(cam_config_fn)
    # cam_config does not have lens position
    cam_config2.set_lens_position(*lens_position)
    cam_config2.set_lens_position(*lens_position)
    # only h_ref was different in the .json file, adapt this and then compare
    cam_config2.gcps["h_ref"] = 0.0
    assert cam_config2.gcps == cam_config.gcps
    assert cam_config2.lens_position == cam_config.lens_position
    # assert(cam_config2.crs == cam_config.crs) # these may differ a very small bit, hence left out of testing
    assert cam_config2.window_size == cam_config.window_size
    assert cam_config2.resolution == cam_config.resolution


@pytest.mark.parametrize(
    "mode",
    ["camera", "geographical", "3d"],
)
def test_plot(cam_config, vid, mode):
    import matplotlib as mpl

    ax = cam_config.plot(mode=mode)
    if mode == "camera":
        assert isinstance(ax, mpl.axes.Axes)


def test_cv_undistort_points(cam_config):
    # let's fake a distortion
    cam_config.dist_coeffs[0][0] = -3e-3
    src = cam_config.gcps["src"]
    mtx = cam_config.camera_matrix
    dist = cam_config.dist_coeffs
    _ = cv.undistort_points(src, mtx, dist)
    src_back_dist = cv.undistort_points(src, mtx, dist, reverse=True)
    # check if points are back to originals after back adn forth undistortion and distortion
    assert np.allclose(src, src_back_dist)


@pytest.mark.parametrize("cur_cam_config", ["cam_config_6gcps", "cam_config", "cam_config_rvec_tvec"])
def test_unproject_points(cur_cam_config, request):
    cur_cam_config = request.getfixturevalue(cur_cam_config)
    dst = cur_cam_config.gcps_dest
    # project x, y, z point to camera objective
    src_est = cur_cam_config.project_points(dst)
    # now back project and compare if the results are nearly identical
    zs = [pt[-1] for pt in dst]
    dst_est = cur_cam_config.unproject_points(src_est, zs)
    print(dst_est[0] - dst[0])
    assert np.allclose(dst, dst_est)


def test_camera_calib(cam_config_calib, calib_video):
    cam_config_calib.set_lens_calibration(calib_video, max_imgs=5, plot=False, progress_bar=False)
