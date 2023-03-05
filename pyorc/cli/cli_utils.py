import click
import geopandas as gpd
import json
import logging
import matplotlib.pyplot as plt
import os
import pyorc
from shapely.geometry import Point
import yaml

from pyorc import Video, helpers, CameraConfig
from pyorc.cli.cli_elements import GcpSelect, AoiSelect


def get_corners_interactive(fn, gcps, crs=None, frame_sample=0., logger=logging):
    vid = Video(fn, start_frame=frame_sample, end_frame=frame_sample + 1)
    # get first frame
    frame = vid.get_frame(0, method="rgb")
    src = gcps["src"]
    if crs is not None:
        dst = helpers.xyz_transform(gcps["dst"], crs_from=crs, crs_to=4326)
    else:
        dst = gcps["dst"]
    # setup preliminary cam config
    cam_config = CameraConfig(height=frame.shape[0], width=frame.shape[1], gcps=gcps, crs=crs)
    selector = AoiSelect(frame, src, dst, cam_config, logger=logger)
    # uncomment below to test the interaction, not suitable for automated unit test
    plt.show(block=True)
    return selector.src

    # setup a cam_config without

def get_gcps_interactive(fn, dst, crs=None, crs_gcps=None, frame_sample=0., logger=logging):
    vid = Video(fn, start_frame=frame_sample, end_frame=frame_sample + 1)
    # get first frame
    frame = vid.get_frame(0, method="rgb")
    if crs_gcps is not None:
        dst = helpers.xyz_transform(dst, crs_from=crs_gcps, crs_to=4326)
    selector = GcpSelect(frame, dst, crs=crs, logger=logger)
    # uncomment below to test the interaction, not suitable for automated unit test
    plt.show(block=True)
    return selector.src

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
                cls = getattr(pyorc, check_args[k].capitalize())
                if (not hasattr(cls, m)):
                    raise ValueError(f"Class '{check_args[k].capitalize()}' does not have a method or property '{m}'")
                method = getattr(cls, m)
                # find valid args to method
                if hasattr(method, "__call__"):
                    if k in check_args:
                        valid_args = method.__code__.co_varnames[:method.__code__.co_argcount]  # get input args to method
                        for arg in recipe[k][m]:
                            if not(arg in valid_args):
                                raise ValueError(f"Method '{check_args[k].capitalize()}.{m}' does not have input argument '{arg}', must be one of {valid_args}")
    # add empty dicts for missing but compulsory classes
    for _c in required_classes:
        if _c not in recipe:
            # add empties for compulsory recipe components
            recipe[_c] = {}
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
    # if value is not None:
    #     if len(value) == 4:
    #         # assume [x, y] pairs are provided
    #         len_points = 2
    #     elif len(value) < 6:
    #         raise click.UsageError(f"--dst must contain at least 4 with [x, y] or 6 with [x, y, z] points, contains {len(value)}.")
    #     else:
    #         len_points = 3
    #     for n, val in enumerate(value):
    #         assert(isinstance(val, list)), f"--dst value {n} is not a list {val}"
    #         assert(len(val) == len_points), f"--src value {n} must contain row, column coordinate but consists of {len(val)} numbers"
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


def read_shape(fn):
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
        raise click.FileError(f"{fn} does not contain CRS, use a GIS program to add a valid CRS.")
    if gdf.crs is None:
        raise click.FileError(f"{fn} does not contain CRS, use a GIS program to add a valid CRS.")
    return coords, gdf.crs.to_wkt()

def validate_dst(value):
    if value is not None:
        if len(value) == 4:
            # assume [x, y] pairs are provided
            len_points = 2
        elif len(value) < 6:
            raise click.UsageError(f"--dst must contain at least 4 with [x, y] or 6 with [x, y, z] points, contains {len(value)}.")
        else:
            len_points = 3
        for n, val in enumerate(value):
            assert(isinstance(val, list)), f"--dst value {n} is not a list {val}"
            assert(len(val) == len_points), f"--src value {n} must contain row, column coordinate but consists of {len(val)} numbers"
    return value
