# Main command line routines
import click
import os
import logging
import warnings
import numpy as np
from . import cli_utils
# import pyorc api below
from .. import __version__
# import cli components below
## MAIN
logger = logging.getLogger(__name__)

def print_info(ctx, param, value):
    if not value:
        return {}
    click.echo(f"PyOpenRiverCam, Copyright Localdevices, Rainbow Sensing")
    ctx.exit()


@click.group()
@click.version_option(__version__, message="PyOpenRiverCam version: %(version)s")
@click.option(
    "--info",
    default=False,
    is_flag=True,
    is_eager=True,
    help="Print information and version of PyOpenRiverCam",
    callback=print_info,
)
@click.option(
    '--debug/--no-debug',
    default=False,
    envvar='REPO_DEBUG'
)
@click.pass_context
def cli(ctx, info, debug):  # , quiet, verbose):
    """Command line interface for hydromt models."""
    if ctx.obj is None:
        ctx.obj = {}

    # ctx.obj["log_level"] = max(10, 30 - 10 * (verbose - quiet))
    # logging.basicConfig(stream=sys.stderr, level=ctx.obj["log_level"])


## CAMERA CONFIG

@cli.command(short_help="Prepare Camera Configuration file")
@click.argument(
    'OUTPUT',
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    required=True
)
@click.option(
    "-v",
    "--video",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    help="video file with required objective and resolution and control points in view",
)
@click.option(
    "--gcps",
    type=str,
    callback=cli_utils.parse_gcps
)
@click.option(
    "--corners",
    type=str,
    callback=cli_utils.parse_corners
)
@click.pass_context
def camera_config(ctx, output, video, gcps, corners):
    print(corners)
    # pass
