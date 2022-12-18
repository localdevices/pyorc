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


def test_cli_cam_config_video(cli_obj, vid_file, gcps_src, gcps_dst):
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
            '0.0',
            '--href',
            '5.6',
            'OUTPUT.json',

        ],
        echo=True
    )
    assert result.exit_code == 0



