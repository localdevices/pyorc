from click.testing import CliRunner
from pyorc.cli.main import cli
from pyorc.cli.cli_elements import GcpSelect, AoiSelect
from pyorc.helpers import xyz_transform
import json

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
            '--src',
            json.dumps(gcps_src),
            '--dst',
            json.dumps(gcps_dst),
            '--crs',
            '32735',
            '--z0',
            '1182.2',
            '--href',
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
    result = cli_obj.invoke(
        cli, [
            'velocimetry',
            '-V',
            vid_file,
            '-c',
            cam_config_fn,
            '-r',
            cli_recipe_fn,
            '-vvv',
            cli_output_dir,
            '-u'
        ],
        echo=True
    )
    assert result.exit_code == 0


def test_service_video(velocity_flow_processor):
    raise NotImplementedError


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
    plt.show(block=True)



def test_gcps_interact_waterdunen(gcps_dst, frame_rgb):
    import pyorc
    fn = "/home/hcwinsemius/Media/projects/P013_RWSCIV/Metingen/Drone/DJI_0018.MOV"
    vid = pyorc.Video(fn)
    frame_rgb = vid.get_frame(0, method="rgb")
    gcps_dst = [
        [
            24193.95617541109,
            380966.1278342909,
            45.455
        ],
        [
            24191.886643009275,
            380971.87953922385,
            45.502
        ],
        [
            24189.00241680468,
            380979.22564477695,
            45.751
        ],
        [
            24187.163463049525,
            380984.4158229723,
            45.733
        ],
        [
            24230.08041334225,
            380977.349173881,
            46.103
        ],
        [
            24228.96874076969,
            380980.82290807355,
            46.057
        ],
        [
            24226.83237745527,
            380986.90019010287,
            46.015
        ],
        [
            24223.755738388892,
            380994.9752361445,
            45.818
        ]
    ]

    import matplotlib.pyplot as plt
    # convert dst to
    crs = 28992
    # crs = None
    if crs is not None:
        dst = xyz_transform(gcps_dst, crs_from=crs, crs_to=4326)
    else:
        dst = gcps_dst
    selector = GcpSelect(frame_rgb, dst, crs=crs)
    # uncomment below to test the interaction, not suitable for automated unit test
    plt.show(block=True)


def test_aoi_interact(frame_rgb, cam_config_without_aoi):
    import matplotlib.pyplot as plt
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
