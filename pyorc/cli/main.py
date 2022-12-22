# Main command line routines
import click
from typing import List, Optional, Union
from typeguard import typechecked
import os
import yaml

# import CLI components
from pyorc.cli import cli_utils
from pyorc.cli import log
# import pyorc api below
from pyorc import __version__
import pyorc
# import cli components below


## MAIN

def print_info(ctx, param, value):
    if not value:
        return {}
    click.echo(f"PyOpenRiverCam, Copyright Localdevices, Rainbow Sensing")
    ctx.exit()

def print_license(ctx, param, value):
    if not value:
        return {}
    click.echo(f"GNU Affero General Public License v3 (AGPLv3). See https://www.gnu.org/licenses/agpl-3.0.en.html")
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
    "--license",
    default=False,
    is_flag=True,
    is_eager=True,
    help="Print license information for PyOpenRiverCam",
    callback=print_license,
)
@click.option(
    '--debug/--no-debug',
    default=False,
    envvar='REPO_DEBUG'
)
@click.pass_context
def cli(ctx, info, license, debug):  # , quiet, verbose):
    """Command line interface for hydromt models."""
    if ctx.obj is None:
        ctx.obj = {}

    # ctx.obj["log_level"] = max(10, 30 - 10 * (verbose - quiet))
    # logging.basicConfig(stream=sys.stderr, level=ctx.obj["log_level"])


## CAMERA CONFIG
#
@cli.command(short_help="Prepare Camera Configuration file")
@click.argument(
    'OUTPUT',
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    required=True,
)
@click.option(
    "-v",
    "--videofile",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    help="video file with required objective and resolution and control points in view",
    callback=cli_utils.validate_file
)
@click.option(
    "--crs",
    type=str,
    callback=cli_utils.parse_str_num,
    help="Coordinate reference system to be used for camera configuration"
)
@click.option(
    "--src",
    type=str,
    callback=cli_utils.parse_src,
    help='Source control points as list of [column, row] pairs.'
)
@click.option(
    "--dst",
    type=str,
    callback=cli_utils.parse_dst,
    help='Destination control points as list of 4 [x, y] pairs, or at least 6 [x, y, z]. If --crs_gcps is provided, --dst is assumed to be in this CRS."'
)
@click.option(
    "--z0",
    type=float,
    help="Water level [m] +CRS (e.g. geoid or ellipsoid of GPS)"
)
@click.option(
    "--href",
    type=float,
    help="Water level [m] +local datum (e.g. staff or pressure gauge)"
)
@click.option(
    "--crs_gcps",
    type=str,
    callback=cli_utils.parse_str_num,
    help="Coordinate reference system in which destination GCP points (--dst) are measured"
)
@click.option(
    "--resolution",
    type=float,
    help="Target resolution [m] for ortho-projection."
)
@click.option(
    "--window_size",
    type=int,
    help="Target window size [px] for interrogation window for Particle Image Velocimetry"
)
@click.option(
    "--shapefile",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    help="Shapefile containing dst GCP points [x, y] or [x, y, z] in its geometry",
    callback=cli_utils.validate_file
)
@click.option(
    "--lens_position",
    type=str,
    help="Lens position as [x, y, z]. If --crs_gcps is provided, --lens_position is assumed to be in this CRS.",
    callback=cli_utils.parse_json
)
@click.option(
    "--corners",
    type=str,
    callback=cli_utils.parse_corners,
    help="Video ojective corner points as list of 4 [column, row] points"
)
@click.pass_context
@typechecked
def camera_config(
        ctx,
        output: str,
        videofile: str,
        crs: Optional[Union[str, int]],
        src: Optional[List[List[float]]],
        dst: Optional[List[List[float]]],
        z0: Optional[float],
        href: Optional[float],
        crs_gcps: Optional[Union[str, int]],
        resolution: Optional[float],
        window_size: Optional[int],
        lens_position: Optional[List[float]],
        shapefile: Optional[str],
        corners: Optional[List[List]]
):
    logger = log.setuplog("cameraconfig", os.path.abspath("pyorc.log"), append=False)
    logger.info(f"Preparing your cameraconfig file in {output}")
    logger.info(f"Found video file  {videofile}")


    if src is not None:
        logger.info("Source points found and validated")
    if dst is not None:
        logger.info("Destination points found and validated")
    if z0 is None:
        z0: float = click.prompt("--z0 not provided, please enter a number, or Enter for default", default=0.0)
    if href is None:
        href: float = click.prompt("--href not provided, please enter a number, or Enter for default", default=0.0)
    if resolution is None:
        resolution: float = click.prompt("--resolution not provided, please enter a number, or Enter for default", default=0.05)
    if window_size is None:
        window_size: int = click.prompt("--window_size not provided, please enter a number, or Enter for default", default=10)

    if not src:
        logger.warning("No source control points provided. No problem, you can interactively click them in your objective")
        if click.confirm('Do you want to continue and provide source points interactively?', default=True):
            raise click.UsageError("Interactive clicker is not implemented yet.")
    if not corners:
        logger.warning("No corner points for projection provided. No problem, you can interactively click them in your objective")
        if click.confirm('Do you want to continue and provide corners interactively?', default=True):
            raise click.UsageError("Interactive clicker is not implemented yet.")


    if crs is None and crs_gcps is not None:
        raise click.UsageError(f"--crs is None while --crs_gcps is {crs_gcps}, please supply --crs.")
    gcps = {
        "src": src,
        "dst": dst,
        "z_0": z0,
        "h_ref": href,
        "crs": crs_gcps
    }
    pyorc.service.camera_config(
        video_file=videofile,
        cam_config_file=output,
        gcps=gcps,
        crs=crs,
        resolution=resolution,
        window_size=window_size,
        lens_position=lens_position,
        corners=corners
    )
    # raise NotImplementedError

## VELOCIMETRY
@cli.command(short_help="Estimate velocimetry")
@click.argument(
    'OUTPUT',
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
    callback=cli_utils.validate_dir,
    required=True,
)
@click.option(
    "-v",
    "--videofile",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="video file with required objective and resolution and control points in view",
    callback=cli_utils.validate_file,
    required=True
)
@click.option(
    "-r",
    "--recipe",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="Options file (*.yml)",
    callback=cli_utils.parse_recipe,
    required=True
)
@click.option(
    "-c",
    "--cameraconfig",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="Camera config file (*.json)",
    callback=cli_utils.validate_file,
    required=True
)
@click.pass_context
def velocimetry(ctx, output, videofile, recipe, cameraconfig):
    logger = log.setuplog("velocimetry", os.path.abspath("pyorc.log"), append=False)
    logger.info(f"Preparing your velocimetry result in {output}")
    # read yaml
    pass

