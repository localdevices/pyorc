import json
from matplotlib import backend_bases
import matplotlib.pyplot as plt
import os.path
import pytest
import warnings
from click.testing import CliRunner

import pyorc.service
from pyorc.cli.main import cli
from pyorc.cli.cli_elements import GcpSelect, AoiSelect, StabilizeSelect
from pyorc.cli import cli_utils
from pyorc.helpers import xyz_transform

def test_cli_cam_config(cli_obj):
    result = cli_obj.invoke(
        cli, [
            'camera-config',
            '--help'
        ],
        echo=True
    )
    assert result.exit_code == 0


def test_cli_cam_config_video(cli_obj, vid_file, gcps_src, gcps_dst, lens_position, corners, cli_cam_config_output):
    result = cli_obj.invoke(
        cli, [
            'camera-config',
            '-V',
            vid_file,
            # '--src',
            # json.dumps(gcps_src),
            '--dst',
            json.dumps(gcps_dst),
            '--crs',
            '32735',
            '--z_0',
            '1182.2',
            '--h_ref',
            '0.0',
            '--lens_position',
            json.dumps(lens_position),
            '--resolution',
            '0.03',
            '--window_size',
            '25',
            '--corners',
            json.dumps(corners),
            '-vvv',
            cli_cam_config_output,

        ],
        echo=True
    )
    # assert result.exit_code == 0


def test_cli_velocimetry(cli_obj, vid_file, cam_config_fn, cli_recipe_fn, cli_output_dir):
    # ensure we are in the right folder
    print(f"current file is: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    result = cli_obj.invoke(
        cli, [
            'velocimetry',
            '-V',
            vid_file,
            '-c',
            cam_config_fn,
            '-r',
            cli_recipe_fn,
            '-v',
            cli_output_dir,
            '-u'
        ],
        echo=True
    )
    import time
    time.sleep(1)
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "recipe_",
    [
        "recipe",
        "recipe_geojson",
    ]
)
def test_service_video(recipe_, vid_file, cam_config, cli_prefix, cli_output_dir, request):
    recipe_ = request.getfixturevalue(recipe_)
    # ensure we are in the right folder
    print(f"current file is: {os.path.dirname(__file__)}")
    os.chdir(os.path.dirname(__file__))
    pyorc.service.velocity_flow(
        recipe=recipe_,
        videofile=vid_file,
        cameraconfig=cam_config.to_dict(),
        prefix=cli_prefix,
        output=cli_output_dir
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
        name="click", canvas=selector.fig.canvas, x=5, y=5,
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
        name="click", canvas=selector.fig.canvas, x=5, y=5,
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
        name="click", canvas=selector.fig.canvas, x=5, y=5,
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
    assert(isinstance(wkt, str))
    assert(isinstance(coords, list))
