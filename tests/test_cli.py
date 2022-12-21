from click.testing import CliRunner
from pyorc.cli.main import cli
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


def test_cli_cam_config_video(cli_obj, vid_file, gcps_src, gcps_dst, lens_position, corners):
    result = cli_obj.invoke(
        cli, [
            'camera-config',
            '-v',
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
            '0.01',
            '--window_size',
            '25',
            '--corners',
            json.dumps(corners),
            'OUTPUT.json',

        ],
        echo=True
    )
    assert result.exit_code == 0



