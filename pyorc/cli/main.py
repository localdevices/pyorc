"""Main command line routines pyorc."""

import os
import sys
from typing import List, Optional, Union

import click
from typeguard import typechecked

from pyorc import __version__, service
from pyorc.cli import cli_utils, log


def print_info(ctx, param, value):
    """Display pyorc information."""
    if not value:
        return {}
    click.echo(f"PyOpenRiverCam version {__version__}, Copyright Localdevices, Rainbow Sensing\n")
    ctx.exit()


def print_license(ctx, param, value):
    """Display pyorc license."""
    if not value:
        return {}
    click.echo("GNU Affero General Public License v3 (AGPLv3). See https://www.gnu.org/licenses/agpl-3.0.en.html\n")
    ctx.exit()


verbose_opt = click.option("--verbose", "-v", count=True, help="Increase verbosity.")
video_opt = click.option(
    "-V",
    "--videofile",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="video file with required objective and resolution and control points in view",
    callback=cli_utils.validate_file,
    required=True,
)


@click.group(context_settings={"max_content_width": 120})
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
@click.pass_context
def cli(ctx, info, license):
    """Command line interface for pyOpenRiverCam."""
    if ctx.obj is None:
        ctx.obj = {}


# CAMERA CONFIG
@cli.command(short_help="Prepare Camera Configuration file")
@click.argument(
    "OUTPUT",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    required=True,
)
@video_opt
@click.option(
    "--crs",
    type=str,
    callback=cli_utils.parse_str_num,
    help="Coordinate reference system to be used for camera configuration",
)
@click.option(
    "-f", "--frame-sample", type=int, default=0, help="Frame number to use for camera configuration background"
)
@click.option(
    "--src", type=str, callback=cli_utils.parse_src, help="Source control points as list of [column, row] pairs."
)
@click.option(
    "--dst",
    type=str,
    callback=cli_utils.parse_dst,
    help="Destination control points as list of 2 or 4 [x, y] pairs, or at least 6 [x, y, z]. If --crs_gcps is "
    "provided, --dst is assumed to be in this CRS.",
)
@click.option("--z_0", type=float, help="Water level [m] +CRS (e.g. geoid or ellipsoid of GPS)")
@click.option("--h_ref", type=float, help="Water level [m] +local datum (e.g. staff or pressure gauge)")
@click.option(
    "--crs_gcps",
    type=str,
    callback=cli_utils.parse_str_num,
    help="Coordinate reference system in which destination GCP points (--dst) are measured",
)
@click.option("--resolution", type=float, help="Target resolution [m] for ortho-projection.")
@click.option("--focal_length", type=float, help="Focal length [pix] of lens.")
@click.option(
    "--k1",
    type=float,
    help="First lens radial distortion coefficient k1 [-]. See also https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html",
)
@click.option(
    "--k2",
    type=float,
    help="Second lens radial distortion coefficient k2 [-]. See also https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html",
)
@click.option(
    "--window_size", type=int, help="Target window size [px] for interrogation window for Particle Image Velocimetry"
)
@click.option(
    "--shapefile",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True),
    help="Shapefile or other GDAL compatible vector file containing dst GCP points [x, y] or [x, y, z] in its geometry",
    callback=cli_utils.validate_file,
)
@click.option(
    "--lens_position",
    type=str,
    help="Lens position as [x, y, z]. If --crs_gcps is provided, --lens_position is assumed to be in this CRS.",
    callback=cli_utils.parse_json,
)
@click.option(
    "--corners",
    type=str,
    callback=cli_utils.parse_corners,
    help="Video ojective corner points as list of 4 [column, row] points",
)
@click.option(
    "--stabilize",
    "-s",
    is_flag=True,
    default=False,
    help="Stabilize the videos using this camera configuration (you can provide a stable area in an interactive view).",
)
@click.option(
    "--rotation",
    type=int,
    required=False,
    callback=cli_utils.validate_rotation,
    help="Provide a rotation of either 90, 180 or 170 degrees if needed to correctly rotate the video",
)
@verbose_opt
@click.pass_context
@typechecked
def camera_config(
    ctx,
    output: str,
    videofile: str,
    crs: Optional[Union[str, int]],
    frame_sample: Optional[int],
    src: Optional[List[List[float]]],
    dst: Optional[List[List[float]]],
    z_0: Optional[float],
    h_ref: Optional[float],
    crs_gcps: Optional[Union[str, int]],
    focal_length: Optional[float],
    k1: Optional[float],
    k2: Optional[float],
    resolution: Optional[float],
    window_size: Optional[int],
    lens_position: Optional[List[float]],
    shapefile: Optional[str],
    corners: Optional[List[List]],
    stabilize: Optional[bool],
    rotation: Optional[int],
    verbose: int,
):
    """CLI subcommand for camera configuration."""
    log_level = max(10, 20 - 10 * verbose)
    logger = log.setuplog("cameraconfig", os.path.abspath("pyorc.log"), append=False, log_level=log_level)
    logger.info(f"Preparing your cameraconfig file in {output}")
    logger.info(f"Found video file  {videofile}")

    if src is not None:
        logger.info("Source points found and validated")
    if dst is not None:
        logger.info("Destination points found and validated")
    if z_0 is None:
        z_0: float = click.prompt("--z_0 not provided, please enter a number, or Enter for default", default=0.0)
    if h_ref is None:
        h_ref: float = click.prompt("--h_ref not provided, please enter a number, or Enter for default", default=0.0)
    if resolution is None:
        resolution: float = click.prompt(
            "--resolution not provided, please enter a number, or Enter for default", default=0.05
        )
    if window_size is None:
        window_size: int = click.prompt(
            "--window_size not provided, please enter a number, or Enter for default", default=10
        )
    if shapefile is not None:
        if dst is None:
            dst, crs_gcps = cli_utils.read_shape(shapefile)
            # validate if amount of points is logical
            dst = cli_utils.validate_dst(dst)
        else:
            logger.warning(
                f"Shapefile {shapefile} not used, because --dst was provided explicitly and overrules the use of "
                f"--shapefile."
            )
    if dst is not None:
        logger.info("Destination points found and validated")
    else:
        raise click.UsageError(
            "No destination control points for found, either provide a list of points with --dst or provide a "
            "shapefile with --shapefile"
        )
    if not src:
        logger.warning(
            "No source control points provided. No problem, you can interactively click them in your objective"
        )
        if click.confirm("Do you want to continue and provide source points interactively?", default=True):
            src, camera_matrix, dist_coeffs = cli_utils.get_gcps_interactive(
                videofile,
                dst,
                crs=crs,
                crs_gcps=crs_gcps,
                frame_sample=frame_sample,
                focal_length=focal_length,
                k1=k1,
                k2=k2,
                lens_position=lens_position,
                rotation=rotation,
                logger=logger,
            )
            if len(src) != len(dst):
                raise click.UsageError(f"You have not provided enough source points {len(src)}/{len(dst)} available")
    if crs is None and crs_gcps is not None:
        raise click.UsageError(f"--crs is None while --crs_gcps is {crs_gcps}, please supply --crs.")
    gcps = {"src": src, "dst": dst, "z_0": z_0, "h_ref": h_ref, "crs": crs_gcps}
    if not corners:
        logger.warning(
            "No corner points for projection provided. No problem, you can interactively click them in your objective"
        )
        if click.confirm("Do you want to continue and provide corners interactively?", default=True):
            corners = cli_utils.get_corners_interactive(
                videofile,
                gcps,
                crs=crs,
                crs_gcps=crs_gcps,
                frame_sample=frame_sample,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rotation=rotation,
                logger=logger,
            )
    if stabilize:
        stabilize_pol = cli_utils.get_stabilize_pol(
            videofile,
            frame_sample=frame_sample,
            rotation=rotation,
            logger=logger,
        )
    else:
        stabilize_pol = None
    service.camera_config(
        video_file=videofile,
        cam_config_file=output,
        gcps=gcps,
        crs=crs,
        frame_sample=frame_sample,
        resolution=resolution,
        window_size=window_size,
        lens_position=lens_position,
        corners=corners,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        stabilize=stabilize_pol,
        rotation=rotation,
    )
    logger.info(f"Camera configuration created and stored in {output}")


# VELOCIMETRY
@cli.command(short_help="Estimate velocimetry")
@click.argument(
    "OUTPUT",
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
    callback=cli_utils.validate_dir,
    required=True,
)
@video_opt
@click.option(
    "-r",
    "--recipe",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="Options file (*.yml)",
    callback=cli_utils.parse_recipe,
    required=True,
)
@click.option(
    "-c",
    "--cameraconfig",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="Camera config file (*.json)",
    callback=cli_utils.parse_camconfig,
    required=True,
)
@click.option("-p", "--prefix", type=str, default="", help="Prefix for produced output files")
@click.option(
    "-h",
    "--h_a",
    type=float,
    required=False,
    help="Water level in local vertical datum (e.g. staff or pressure gauge) belonging to video.",
)
@click.option(
    "--cross",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="Cross section file (*.geojson). This will be used for discharge estimation if the `transect` "
    " section is provided in your recipe.",
    callback=cli_utils.parse_cross_section_gdf,
    required=False,
)
@click.option(
    "--cross_wl",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, file_okay=True),
    help="Cross section file (*.geojson). If you provide this, you may add water level retrieval settings to the"
    " recipe in the section `water_level`. For more information see"
    " [PyORC docs](https://localdevices.github.io/pyorc/user-guide/cross_section/index.html)",
    callback=cli_utils.parse_cross_section_gdf,
    required=False,
)
@click.option(
    "-u",
    "--update",
    is_flag=True,
    default=False,
    help="Only update requested output files with changed inputs or if not present on file system",
)
@click.option(
    "--lowmem",
    is_flag=True,
    default=False,
    help="Reduce memory consumption. Computation will be slower",
)
@verbose_opt
@click.pass_context
def velocimetry(ctx, output, videofile, recipe, cameraconfig, prefix, h_a, cross, cross_wl, update, lowmem, verbose):
    """CLI subcommand for velocimetry."""
    log_level = max(10, 20 - 10 * verbose)
    logger = log.setuplog("velocimetry", os.path.abspath("pyorc.log"), append=False, log_level=log_level)
    logger.info(f"Preparing your velocimetry result in {output}")
    # load in recipe and camera config
    service.velocity_flow(
        recipe=recipe,
        videofile=videofile,
        cameraconfig=cameraconfig,
        h_a=h_a,
        cross=cross,
        cross_wl=cross_wl,
        prefix=prefix,
        output=output,
        update=update,
        concurrency=not (lowmem),
        logger=logger,
    )
    pass


# ensure help texts span a reasonable space
# velocimetry(max_content_width=120)

if __name__ == "__main__":
    # if getattr(sys, 'frozen', False):
    cli(sys.argv[1:])
