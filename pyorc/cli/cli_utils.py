import click
import cv2
import geopandas as gpd
import hashlib
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from shapely.geometry import Point

import pyorc.api
from .. import Video, CameraConfig, cv, load_camera_config, helpers
from .cli_elements import GcpSelect, AoiSelect, StabilizeSelect


def get_corners_interactive(
        fn,
        gcps,
        crs=None,
        crs_gcps=None,
        frame_sample=0.,
        camera_matrix=None,
        dist_coeffs=None,
        rotation=None,
        logger=logging,
):
    vid = Video(fn, start_frame=frame_sample, end_frame=frame_sample + 1, rotation=rotation)
    # get first frame
    frame = vid.get_frame(0, method="rgb")
    src = gcps["src"]
    if crs_gcps is not None:
        dst = helpers.xyz_transform(gcps["dst"], crs_from=crs_gcps, crs_to=4326)
    else:
        dst = gcps["dst"]
    # setup preliminary cam config
    cam_config = CameraConfig(
        height=frame.shape[0],
        width=frame.shape[1],
        gcps=gcps,
        crs=crs,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rotation=rotation,
    )
    selector = AoiSelect(frame, src, dst, cam_config, logger=logger)
    # uncomment below to test the interaction, not suitable for automated unit test
    plt.show(block=True)
    return selector.src

    # setup a cam_config without


def get_gcps_interactive(
        fn,
        dst,
        crs=None,
        crs_gcps=None,
        frame_sample=0,
        lens_position=None,
        rotation=None,
        logger=logging
):
    vid = Video(fn, start_frame=frame_sample, end_frame=frame_sample + 1, rotation=rotation)
    # get first frame
    frame = vid.get_frame(0, method="rgb")
    if crs_gcps is not None:
        dst = helpers.xyz_transform(dst, crs_from=crs_gcps, crs_to=4326)
    selector = GcpSelect(frame, dst, crs=crs, lens_position=lens_position, logger=logger)
    plt.show(block=True)
    return selector.src, selector.camera_matrix, selector.dist_coeffs


def get_stabilize_pol(
        fn,
        frame_sample=0,
        rotation=None,
        logger=logging
):
    vid = Video(fn, start_frame=frame_sample, end_frame=frame_sample + 1, rotation=rotation)
    frame = vid.get_frame(0, method="rgb")
    selector = StabilizeSelect(frame, logger=logger)
    plt.show(block=True)
    return selector.src


def get_file_hash(fn):
    hash256 = hashlib.sha256()
    with open(fn, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            # print(byte_block)
            hash256.update(byte_block)
    return hash256


def get_gcps_optimized_fit(src, dst, height, width, c=2., lens_position=None):
    # optimize cam matrix and dist coeffs with provided control points
    if np.array(dst).shape == (4, 2):
        _dst = np.c_[np.array(dst), np.zeros(4)]
    else:
        _dst = np.array(dst)
    camera_matrix, dist_coeffs, err = cv.optimize_intrinsic(
        src,
        _dst,
        height,
        width,
        c=c,
        lens_position=lens_position,
        # dist_coeffs=cv.DIST_COEFFS
    )
    # once optimized, solve the perspective, and estimate the GCP locations with the perspective rot/trans
    coord_mean = np.array(_dst).mean(axis=0)
    _src = np.float32(src)
    _dst = np.float32(np.array(_dst) - coord_mean)
    success, rvec, tvec = cv2.solvePnP(_dst, _src, camera_matrix, np.array(dist_coeffs))

    # estimate source point location
    src_est, jacobian = cv2.projectPoints(_dst, rvec, tvec, camera_matrix, np.array(dist_coeffs))
    src_est = np.array([list(point[0]) for point in src_est])
    dst_est = cv.unproject_points(_src, _dst[:, -1], rvec, tvec, camera_matrix, dist_coeffs)
    dst_est = np.array(dst_est)[:, 0:len(coord_mean)] + coord_mean
    return src_est, dst_est, camera_matrix, dist_coeffs, err


def parse_json(ctx, param, value):
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, "r") as f:
            kwargs = json.load(f)
    else:
        if value.strip("{").startswith("'"):
            value = value.replace("'", '"')
        try:
            kwargs = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f'Could not decode JSON "{value}"')
    return kwargs


def parse_corners(ctx, param, value):
    if value is None:
        return None
    # check if list contains lists of 2 values
    corners = json.loads(value)
    assert(len(corners) == 4), "--corners must contain a list of lists with exactly 4 points"
    for n, val in enumerate(corners):
        assert(isinstance(val, list)), f"--corners value {n} is not a list {val}"
        assert(len(val) == 2), f"--corners value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return [[int(x), int(y)] for x, y in corners]


def validate_file(ctx, param, value):
    if value is not None:
        if not(os.path.isfile(value)):
            raise click.FileError(f"{value}")
        return value


def validate_dir(ctx, param, value):
    if not(os.path.isdir(value)):
        os.makedirs(value)
    return value


def validate_rotation(ctx, param, value):
    if value is not None:
        if not(value in [90, 180, 270, None]):
            raise click.UsageError(f"Rotation value must be either 90, 180 or 270")
        return value


def parse_camconfig(ctx, param, camconfig_file):
    """
    Read and validate cam config file

    Parameters
    ----------
    camconfig_file : str,
       file containing camera configuration

    Returns
    -------
    cam_config: dict
        configuration as dictionary

    """
    # validate if file can be read as camera config object
    camconfig = load_camera_config(camconfig_file)

    # return dict formatted
    return camconfig.to_dict_str()


def parse_recipe(ctx, param, recipe_file):
    """
    Read and validate entire recipe from top to bottom, add compulsory classes where needed

    Parameters
    ----------
    recipe_file : str,
        file containing .yml formatted recipe

    Returns
    -------
    recipe : dict,
        dictionary with entire recipe for running processing
    """
    with open(recipe_file, "r") as f:
        body = f.read()
    recipe = yaml.load(body, Loader=yaml.FullLoader)
    recipe = validate_recipe(recipe)
    return recipe


def parse_src(ctx, param, value):
    value = parse_json(ctx, param, value)
    if value is not None:
        # check if at least 4 of 2
        assert(len(value)>=4), "--src must contain a list of lists [column, row] with at least 4 points"
        for n, val in enumerate(value):
            assert(isinstance(val, list)), f"--src value {n} is not a list {val}"
            assert(len(val) == 2), f"--src value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return value

def parse_dst(ctx, param, value):
    value = parse_json(ctx, param, value)
    value = validate_dst(value)
    return value


def parse_str_num(ctx, param, value):
    if value is not None:
        try:
            float(value)
        except:
            return value
        if value.isnumeric():
            return int(value)
        else:
            return float(value)


def read_shape(fn=None, geojson=None):
    if fn is None and geojson is None:
        raise click.UsageError(f"Either fn or geojson must be provided")
    if geojson:
        if "crs" in geojson:
            crs = geojson["crs"]["properties"]["name"]
        else:
            crs = None
        gdf = gpd.GeoDataFrame().from_features(geojson, crs=crs)
    else:
        gdf = gpd.read_file(fn)
    # check if all geometries are points
    assert(all([isinstance(geom, Point) for geom in gdf.geometry])), f'shapefile may only contain geometries of type ' \
                                                                     f'"Point"'
    # use the first point to check if points are 2d or 3d
    if gdf.geometry[0].has_z:
        coords = [[p.x, p.y, p.z] for p in gdf.geometry]
    else:
        coords = [[p.x, p.y] for p in gdf.geometry]
    if not(hasattr(gdf, "crs")):
        click.echo(f"shapefile or geojson does not contain CRS, assuming CRS is the same as camera config CRS")
        crs = None
    elif gdf.crs is None:
        click.echo(f"shapefile or geojson does not contain CRS, assuming CRS is the same as camera config CRS")
        crs = None
    else:
        crs = gdf.crs.to_wkt()
    return coords, crs


def validate_dst(value):
    if value is not None:
        if len(value) in [2, 4]:
            # assume [x, y] pairs are provided
            len_points = 2
        elif len(value) < 6:
            raise click.UsageError(f"--dst must contain exactly 2 or 4 with [x, y], or at least 6 with [x, y, z] points, contains {len(value)}.")
        else:
            len_points = 3
        for n, val in enumerate(value):
            assert(isinstance(val, list)), f"--dst value {n} is not a list {val}"
            assert(len(val) == len_points), f"--dst value {n} must contain 3 coordinates (x, y, z) but consists of {len(val)} numbers, value is {val}"
    return value


def validate_recipe(recipe):
    valid_classes = ["video", "frames", "velocimetry", "mask", "transect", "plot"]  # allowed classes
    required_classes = ["video", "frames", "velocimetry"]  # mandatory classes (if not present, these are added)
    check_args = {
        "video": "video",
        "frames": "frames",
        # "velocimetry": "frames"
    } # check if arguments to underlying methods called by recipe section are available and get valid arguments
    skip_checks = ["plot"]  # these sections are not checked for valid inputs
    process_methods = ["write"]  # methods that are specifically needed within process steps and not part of pyorc class methods
    for k in recipe:  # main section
        if k not in valid_classes:
            raise ValueError(f"key '{k}' is not allowed, must be one of {valid_classes}")
        # if k in check_args:
        # loop through all methods and check if their inputs are valid
        for m in recipe[k]:  # method in section
            if recipe[k][m] is None:
                # replace for empty dict
                recipe[k][m] = {}
            if m not in process_methods and k in check_args:
                # get the subclass that is called within the section of interest
                cls = getattr(pyorc.api, check_args[k].capitalize())
                if (not hasattr(cls, m)):
                    raise ValueError(f"Class '{check_args[k].capitalize()}' does not have a method or property '{m}'")
                method = getattr(cls, m)
                # find valid args to method
                if hasattr(method, "__call__"):
                    if k in check_args:
                        # check if kwargs is contained in method
                        if "kwargs" in method.__code__.co_varnames:
                            valid_args = None
                        else:
                            valid_args = method.__code__.co_varnames[:method.__code__.co_argcount]  # get input args to method
                        if valid_args:
                            for arg in recipe[k][m]:
                                if not arg in valid_args:
                                    raise ValueError(f"Method '{check_args[k].capitalize()}.{m}' does not have input argument '{arg}', must be one of {valid_args}")
    # add empty dicts for missing but compulsory classes
    for _c in required_classes:
        if _c not in recipe:
            # add empties for compulsory recipe components
            recipe[_c] = {}
    return recipe
