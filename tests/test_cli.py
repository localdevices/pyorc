import json
import os.path

import click
import matplotlib.pyplot as plt
import pytest
from matplotlib import backend_bases

import pyorc.service
from pyorc.cli import cli_utils
from pyorc.cli.cli_elements import AoiSelect, GcpSelect, StabilizeSelect
from pyorc.cli.main import cli
from pyorc.helpers import xyz_transform

from .conftest import EXAMPLE_DATA_DIR


@pytest.fixture()
def cross_section_2_geojson_fn():
    return os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "cross_section2.geojson")


@pytest.fixture()
def recipe_no_waterlevel(recipe_yml):
    from pyorc.cli import cli_utils

    r = cli_utils.parse_recipe("a", "b", recipe_yml)
    del r["video"]["h_a"]
    return r


def test_cli_info(cli_obj):
    result = cli_obj.invoke(cli, ["--info"], echo=True)
    assert result.exit_code == 0


def test_cli_license(cli_obj):
    result = cli_obj.invoke(cli, ["--license"], echo=True)
    assert result.exit_code == 0


def test_cli_cam_config(cli_obj):
    result = cli_obj.invoke(cli, ["camera-config", "--help"], echo=True)
    assert result.exit_code == 0


def test_cli_cam_config_video(cli_obj, vid_file, gcps_src, gcps_dst, lens_position, corners, cli_cam_config_output):
    _ = cli_obj.invoke(
        cli,
        [
            "camera-config",
            "-V",
            vid_file,
            # '--src',
            # json.dumps(gcps_src),
            "--dst",
            json.dumps(gcps_dst),
            "--crs",
            "32735",
            "--z_0",
            "1182.2",
            "--h_ref",
            "0.0",
            "--lens_position",
            json.dumps(lens_position),
            "--resolution",
            "0.03",
            "--focal_length",
            "1500",
            "--window_size",
            "25",
            "--corners",
            json.dumps(corners),
            "-vvv",
            cli_cam_config_output,
        ],
        echo=True,
    )
    # assert result.exit_code == 0


def test_cli_velocimetry(cli_obj, vid_file, cam_config_fn, cli_recipe_fn, cli_output_dir):
    # ensure we are in the right folder
    print(f"current file is: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    result = cli_obj.invoke(
        cli,
        ["velocimetry", "-V", vid_file, "-c", cam_config_fn, "-r", cli_recipe_fn, "-v", cli_output_dir, "-u"],
        echo=True,
    )
    import time

    time.sleep(1)
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "recipe_",
    [
        "recipe",
        "recipe_geojson",
    ],
)
def test_service_video(recipe_, vid_file, cam_config, cli_prefix, cli_output_dir, request):
    recipe_ = request.getfixturevalue(recipe_)
    # ensure we are in the right folder
    print(f"current file is: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    pyorc.service.velocity_flow(
        recipe=recipe_, videofile=vid_file, cameraconfig=cam_config.to_dict(), prefix=cli_prefix, output=cli_output_dir
    )


def test_service_video_no_waterlevel(
    recipe_no_waterlevel, vid_file, cam_config, cli_prefix, cli_output_dir, cross_section_2_geojson_fn
):
    print(f"current file is: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    pyorc.service.velocity_flow(
        recipe=recipe_no_waterlevel,
        videofile=vid_file,
        cameraconfig=cam_config.to_dict(),
        prefix=cli_prefix,
        output=cli_output_dir,
        cross=cross_section_2_geojson_fn,
    )


def test_service_video_no_waterlevel_subprocess(
    recipe_no_waterlevel, vid_file, cam_config, cli_prefix, cli_output_dir, cross_section_2_geojson_fn
):
    with open(cross_section_2_geojson_fn, "r") as f:
        cross_section_2 = json.load(f)
    cameraconfig = cam_config.to_dict()
    # alter the recipe to get the cross section out

    print(f"current file is: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    pyorc.service.velocity_flow_subprocess(
        videofile=vid_file,
        recipe=recipe_no_waterlevel,
        cameraconfig=cameraconfig,
        output=cli_output_dir,
        prefix="subprocess_",
        cross=cross_section_2,
    )


def test_gcps_interact(gcps_dst, frame_rgb):
    import matplotlib.pyplot as plt

    # convert dst to
    crs = 32735
    # crs = None
    if crs is not None:
        dst = xyz_transform(gcps_dst, crs_from=crs, crs_to=4326)
    else:
        dst = gcps_dst
    selector = GcpSelect(frame_rgb, dst, crs=crs)
    # uncomment below to test the interaction, not suitable for automated unit test
    # plt.show(block=True)
    # make a click event for testing callbacks
    event = backend_bases.MouseEvent(
        name="click",
        canvas=selector.fig.canvas,
        x=5,
        y=5,
    )
    selector.on_press(event)
    selector.on_move(event)
    selector.on_click(event)
    selector.on_left_click(event)
    selector.on_right_click(event)
    selector.on_release(event)
    plt.close("all")


def test_aoi_interact(frame_rgb, cam_config_without_aoi):
    # convert dst to
    # del cam_config_without_aoi.crs
    src = cam_config_without_aoi.gcps["src"]
    if hasattr(cam_config_without_aoi, "crs"):
        dst = xyz_transform(cam_config_without_aoi.gcps["dst"], crs_from=cam_config_without_aoi.crs, crs_to=4326)
    else:
        dst = cam_config_without_aoi.gcps["dst"]
    selector = AoiSelect(frame_rgb, src, dst, cam_config_without_aoi)
    # uncomment below to test the interaction, not suitable for automated unit test
    # plt.show(block=True)
    # make a click event for testing callbacks
    event = backend_bases.MouseEvent(
        name="click",
        canvas=selector.fig.canvas,
        x=5,
        y=5,
    )
    selector.on_press(event)
    selector.on_move(event)
    selector.on_click(event)
    selector.on_left_click(event)
    selector.on_right_click(event)
    selector.on_release(event)
    plt.close("all")


def test_stabilize_interact(frame_rgb):
    selector = StabilizeSelect(frame_rgb)
    event = backend_bases.MouseEvent(
        name="click",
        canvas=selector.fig.canvas,
        x=5,
        y=5,
    )

    selector.on_press(event)
    selector.on_move(event)
    selector.on_click(event)
    selector.on_left_click(event)
    selector.on_right_click(event)
    selector.on_release(event)
    selector.on_close(event)
    selector.close_window(event)
    # uncomment below to test the interaction, not suitable for automated unit test
    # plt.show(block=True)
    plt.close("all")


def test_read_shape(gcps_fn):
    coords, wkt = cli_utils.read_shape(gcps_fn)
    assert isinstance(wkt, str)
    assert isinstance(coords, list)


def test_parse_cross_section_file_missing():
    with pytest.raises(click.FileError, match="unknown"):
        cli_utils.parse_cross_section_gdf("", "", "missing_file")


def test_parse_cross_section(gcps_fn):
    fn = cli_utils.parse_cross_section_gdf("", "", gcps_fn)
    assert os.path.isfile(fn)
