from click.testing import CliRunner
from pyorc.cli.main import cli

def test_hello_world():
    runner = CliRunner()
    result = runner.invoke(
        cli, [
            'camera-config',
            '--gcps',
            "{'a': 5}",
            "test.json",
        ]
    )
    assert result.exit_code == 0

