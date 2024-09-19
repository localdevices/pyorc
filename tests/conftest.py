import functools
import json
import numpy as np
import os
import pytest
import pandas as pd
import shutil
from shapely import wkt
import pyorc
import sys
from click.testing import CliRunner


EXAMPLE_DATA_DIR = os.path.join(os.path.split(__file__)[0], "..", "examples")

# fixtures with input and output files and folders
@pytest.fixture
def calib_video():
    return os.path.join(EXAMPLE_DATA_DIR, "camera_calib", "camera_calib_720p.mkv")


@pytest.fixture
def cross_section():
    fn = os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_cross_section.csv")
    return pd.read_csv(fn)


@pytest.fixture
def gcps_fn():
    fn = os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_gcps.geojson")
    return fn


@pytest.fixture
def cam_config_fn():
    return os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere.json")

@pytest.fixture
def cam_config_6gcps_fn():
    return os.path.join(EXAMPLE_DATA_DIR, "geul", "dk_cam_config.json")


@pytest.fixture
def recipe_yml():
    return os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_test.yml")


@pytest.fixture
def cli_output_dir():
    dir = os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "outputs")
    if not os.path.isdir(dir):
        os.makedirs(dir)
    yield dir
    # if os.path.isdir(dir):
    #     shutil.rmtree(dir)

@pytest.fixture
def cli_prefix():
    return "test_"

@pytest.fixture
def cli_recipe_fn():
    return os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_test.yml")


@pytest.fixture
def cli_cam_config_output():
    cam_config_fn = os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_cli.json")
    yield cam_config_fn
    # remove after test
    # os.remove(cam_config_fn)

@pytest.fixture
def cli_click_event(mocker):
    event_props = {
        "xdata": 5,
        "ydata": 5,

    }
    event = backend_bases.MouseEvent(
        name="click", canvas=pyplot.axes(), x=5, y=5,
    )

    # mocker.patch(
    #     "matplotlib.backend_bases.MouseEvent",
    #     return_value=event_props
    # )
    return event

@pytest.fixture
def gcps_src():
    return [
        [1421, 1001],
        [1251, 460],
        [421, 432],
        [470, 607]
    ]


@pytest.fixture
def gcps_dst():
    return [
        [642735.8076, 8304292.1190],  # lowest right coordinate
        [642737.5823, 8304295.593],  # highest right coordinate
        [642732.7864, 8304298.4250],  # highest left coordinate
        [642732.6705, 8304296.8580]  # highest right coordinate
    ]


# sample data, for Ngwerere
@pytest.fixture
def gcps(gcps_src, gcps_dst):
    return dict(
        src=gcps_src,
        dst=gcps_dst,
        z_0=1182.2,
        h_ref=0.
    )


@pytest.fixture
def lens_position():
    return [642732.6705, 8304289.010, 1188.5]


@pytest.fixture
def bbox():
    return wkt.loads("POLYGON ((642730.233168765 8304293.351276383, 642731.5013330225 8304302.039208209, 642739.2789120832 8304300.903926767, 642738.0107478257 8304292.215994941, 642730.233168765 8304293.351276383))")

@pytest.fixture
def bbox_6gcps():
    return wkt.loads("POLYGON ((192103.06271249574 313152.336519752, 192096.59215064772 313165.9688317118, 192104.64144816675 313169.78942190844, 192111.11201001477 313156.1571099486, 192103.06271249574 313152.336519752))")


@pytest.fixture
def corners():
    return [
        [500, 800],
        [400, 600],
        [1200, 550],
        [1350, 650]
        # [292, 817],
        # [50, 166],
        # [1200, 236],
        # [1600, 834]
    ]

@pytest.fixture
def corners_6gcps():
    return [
        [390, 440],
        [1060, 160],
        [1800, 270],
        [1500, 880]
    ]


@pytest.fixture
def lens_pars():
    return {
        "k1": 0,
        "c": 2.0,
        "focal_length": 1550.
    }


@pytest.fixture
def camera_matrix():
    return np.array([[1550., 0., 960.], [0., 1550., 540.], [0., 0., 1.]])


@pytest.fixture
def cam_config(gcps, lens_position, lens_pars, corners):
    return pyorc.CameraConfig(
        height=1080,
        width=1920,
        gcps=gcps,
        lens_position=lens_position,
        lens_pars=lens_pars,
        corners=corners,
        window_size=25,
        resolution=0.01,
        crs=32735
        )

@pytest.fixture
def cam_config_6gcps(cam_config_6gcps_fn):
    # load in memory
    return pyorc.load_camera_config(cam_config_6gcps_fn)

@pytest.fixture
def cam_config_without_aoi(lens_position, gcps):
    return pyorc.CameraConfig(
        height=1080,
        width=1920,
        lens_position=lens_position,
        gcps=gcps,
        window_size=25,
        resolution=0.01,
        crs=32735
        )

@pytest.fixture
def cam_config_calib():
    return pyorc.CameraConfig(
        height=720,
        width=1280,
    )


@pytest.fixture
def dist_coeffs():
    return np.array([[0.], [0.], [0.], [0.]])


@pytest.fixture
def h_a():
    return 0.


@pytest.fixture
def cam_config_dict():
    return {'height': 1080,
            'width': 1920,
            'crs': 'PROJCRS["WGS 84 / UTM zone 35S",BASEGEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]],CONVERSION["UTM zone 35S",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",27,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",10000000,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Engineering survey, topographic mapping."],AREA["Between 24°E and 30°E, southern hemisphere between 80°S and equator, onshore and offshore. Botswana. Burundi. Democratic Republic of the Congo (Zaire). Rwanda. South Africa. Tanzania. Uganda. Zambia. Zimbabwe."],BBOX[-80,24,0,30]],ID["EPSG",32735]]',
            'resolution': 0.01,
            'lens_position': [642732.6705, 8304289.01, 1188.5],
            'gcps': {
                'src': [[1421, 1001], [1251, 460], [421, 432], [470, 607]],
                'dst': [[642735.8076, 8304292.119], [642737.5823, 8304295.593], [642732.7864, 8304298.425], [642732.6705, 8304296.858]],
                'h_ref': 0.0,
                'z_0': 1182.2
            },
            'window_size': 25,
            'is_nadir': False

    }


@pytest.fixture
def cam_config_str():
    # return '{\n    "crs": "PROJCRS[\\"WGS 84 / UTM zone 35S\\",BASEGEOGCRS[\\"WGS 84\\",ENSEMBLE[\\"World Geodetic System 1984 ensemble\\",MEMBER[\\"World Geodetic System 1984 (Transit)\\"],MEMBER[\\"World Geodetic System 1984 (G730)\\"],MEMBER[\\"World Geodetic System 1984 (G873)\\"],MEMBER[\\"World Geodetic System 1984 (G1150)\\"],MEMBER[\\"World Geodetic System 1984 (G1674)\\"],MEMBER[\\"World Geodetic System 1984 (G1762)\\"],MEMBER[\\"World Geodetic System 1984 (G2139)\\"],ELLIPSOID[\\"WGS 84\\",6378137,298.257223563,LENGTHUNIT[\\"metre\\",1]],ENSEMBLEACCURACY[2.0]],PRIMEM[\\"Greenwich\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433]],ID[\\"EPSG\\",4326]],CONVERSION[\\"UTM zone 35S\\",METHOD[\\"Transverse Mercator\\",ID[\\"EPSG\\",9807]],PARAMETER[\\"Latitude of natural origin\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8801]],PARAMETER[\\"Longitude of natural origin\\",27,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8802]],PARAMETER[\\"Scale factor at natural origin\\",0.9996,SCALEUNIT[\\"unity\\",1],ID[\\"EPSG\\",8805]],PARAMETER[\\"False easting\\",500000,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8806]],PARAMETER[\\"False northing\\",10000000,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8807]]],CS[Cartesian,2],AXIS[\\"(E)\\",east,ORDER[1],LENGTHUNIT[\\"metre\\",1]],AXIS[\\"(N)\\",north,ORDER[2],LENGTHUNIT[\\"metre\\",1]],USAGE[SCOPE[\\"Engineering survey, topographic mapping.\\"],AREA[\\"Between 24\\u00b0E and 30\\u00b0E, southern hemisphere between 80\\u00b0S and equator, onshore and offshore. Botswana. Burundi. Democratic Republic of the Congo (Zaire). Rwanda. South Africa. Tanzania. Uganda. Zambia. Zimbabwe.\\"],BBOX[-80,24,0,30]],ID[\\"EPSG\\",32735]]",\n    "resolution": 0.01,\n    "lens_position": [\n        642732.6705,\n        8304289.01,\n        1188.5\n    ],\n    "gcps": {\n        "src": [\n            [\n                1421,\n                1001\n            ],\n            [\n                1251,\n                460\n            ],\n            [\n                421,\n                432\n            ],\n            [\n                470,\n                607\n            ]\n        ],\n        "dst": [\n            [\n                642735.8076,\n                8304292.119\n            ],\n            [\n                642737.5823,\n                8304295.593\n            ],\n            [\n                642732.7864,\n                8304298.425\n            ],\n            [\n                642732.6705,\n                8304296.858\n            ]\n        ],\n        "h_ref": 0.0,\n        "z_0": 1182.2\n    },\n    "lens_pars": {\n        "k1": 0,\n        "c": 2.0,\n        "f": 1.0\n    },\n    "window_size": 25,\n    "corners": [\n        [\n            292,\n            817\n        ],\n        [\n            50,\n            166\n        ],\n        [\n            1200,\n            236\n        ],\n        [\n            1600,\n            834\n        ]\n    ]\n}'
    return '{\n    "height": "1080",\n    "width": "1920",\n    "crs": "PROJCRS[\\"WGS 84 / UTM zone 35S\\",BASEGEOGCRS[\\"WGS 84\\",ENSEMBLE[\\"World Geodetic System 1984 ensemble\\",MEMBER[\\"World Geodetic System 1984 (Transit)\\"],MEMBER[\\"World Geodetic System 1984 (G730)\\"],MEMBER[\\"World Geodetic System 1984 (G873)\\"],MEMBER[\\"World Geodetic System 1984 (G1150)\\"],MEMBER[\\"World Geodetic System 1984 (G1674)\\"],MEMBER[\\"World Geodetic System 1984 (G1762)\\"],MEMBER[\\"World Geodetic System 1984 (G2139)\\"],ELLIPSOID[\\"WGS 84\\",6378137,298.257223563,LENGTHUNIT[\\"metre\\",1]],ENSEMBLEACCURACY[2.0]],PRIMEM[\\"Greenwich\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433]],ID[\\"EPSG\\",4326]],CONVERSION[\\"UTM zone 35S\\",METHOD[\\"Transverse Mercator\\",ID[\\"EPSG\\",9807]],PARAMETER[\\"Latitude of natural origin\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8801]],PARAMETER[\\"Longitude of natural origin\\",27,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8802]],PARAMETER[\\"Scale factor at natural origin\\",0.9996,SCALEUNIT[\\"unity\\",1],ID[\\"EPSG\\",8805]],PARAMETER[\\"False easting\\",500000,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8806]],PARAMETER[\\"False northing\\",10000000,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8807]]],CS[Cartesian,2],AXIS[\\"(E)\\",east,ORDER[1],LENGTHUNIT[\\"metre\\",1]],AXIS[\\"(N)\\",north,ORDER[2],LENGTHUNIT[\\"metre\\",1]],USAGE[SCOPE[\\"Engineering survey, topographic mapping.\\"],AREA[\\"Between 24\\u00b0E and 30\\u00b0E, southern hemisphere between 80\\u00b0S and equator, onshore and offshore. Botswana. Burundi. Democratic Republic of the Congo (Zaire). Rwanda. South Africa. Tanzania. Uganda. Zambia. Zimbabwe.\\"],BBOX[-80,24,0,30]],ID[\\"EPSG\\",32735]]",\n    "resolution": "0.01",\n    "lens_position": "[642732.6705, 8304289.01, 1188.5]",\n    "gcps": "{\'src\': [[1421, 1001], [1251, 460], [421, 432], [470, 607]], \'dst\': [[642735.8076, 8304292.119], [642737.5823, 8304295.593], [642732.7864, 8304298.425], [642732.6705, 8304296.858]], \'h_ref\': 0.0, \'z_0\': 1182.2}",\n    "dist_coeffs": "[[0.]\\n [0.]\\n [0.]\\n [0.]]",\n    "camera_matrix": "[[  1.   0. 960.]\\n [  0.   1. 540.]\\n [  0.   0.   1.]]",\n    "window_size": "25",\n    "bbox": "POLYGON ((642730.15387931 8304292.596551724, 642731.0625 8304303.5, 642739.4004310342 8304302.805172414, 642738.4918103442 8304291.901724137, 642730.15387931 8304292.596551724))"\n}'


@pytest.fixture
def vid_file():
    return os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_20191103.mp4")


@pytest.fixture
def vid_file_6gcps():
    return os.path.join(EXAMPLE_DATA_DIR, "geul", "dk_control.mp4")


@pytest.fixture
def vid(vid_file):
    vid = pyorc.Video(
        vid_file,
        start_frame=0,
        end_frame=2,
    )
    yield vid


@pytest.fixture
def vid_6gcps_cam_config(vid_file_6gcps, cam_config_6gcps):
    vid = pyorc.Video(
        vid_file_6gcps,
        start_frame=0,
        end_frame=2,
        camera_config=cam_config_6gcps,
        h_a=92.36
    )
    yield vid


@pytest.fixture
def vid_cam_config(cam_config):
    vid = pyorc.Video(
        os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_20191103.mp4"),
        start_frame=0,
        end_frame=2,
        camera_config=cam_config,
        h_a=0.
    )
    yield vid

@pytest.fixture
def vid_cam_config_nonlazy(cam_config):
    vid = pyorc.Video(
        os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_20191103.mp4"),
        start_frame=0,
        end_frame=2,
        camera_config=cam_config,
        h_a=0.,
        lazy=False
    )
    yield vid


@pytest.fixture
def vid_cam_config_shift(cam_config):
    vid = pyorc.Video(
        os.path.join(EXAMPLE_DATA_DIR, "ngwerere", "ngwerere_20191103.mp4"),
        start_frame=2,
        end_frame=4,
        camera_config=cam_config,
        h_a=0.
    )
    yield vid



@pytest.fixture
def vid_cam_config_stabilize(cam_config):
    vid = pyorc.Video(
        os.path.join(
            EXAMPLE_DATA_DIR,
            "ngwerere",
            "ngwerere_20191103.mp4"
        ),
        start_frame=0,
        end_frame=20,
        camera_config=cam_config,
        h_a=0.,
        stabilize=[
            [400, 1080],
            [170, 0],
            [1000, 0],
            [1750, 1080],
        ]  # coordinates for which outside area is meant for stabilization
    )
    yield vid


@pytest.fixture
def frame_rgb(vid_cam_config):
    return vid_cam_config.get_frame(0, method="rgb")


@pytest.fixture
def frames_grayscale(vid_cam_config):
    return vid_cam_config.get_frames()


@pytest.fixture
def frames_grayscale_shift(vid_cam_config_shift):
    return vid_cam_config_shift.get_frames()



@pytest.fixture
def frames_rgb_stabilize(vid_cam_config_stabilize):
    return vid_cam_config_stabilize.get_frames(method="rgb")


@pytest.fixture
def frames_rgb(vid_cam_config):
    return vid_cam_config.get_frames(method="rgb")


@pytest.fixture
def frames_proj(frames_grayscale):
    return frames_grayscale.frames.project()


@pytest.fixture
def ani_mp4():
    yield "temp.mp4"
    os.remove("temp.mp4")


@pytest.fixture
def piv(frames_proj):
    # provide a short piv object
    return frames_proj.frames.get_piv()


@pytest.fixture
def piv_transect(piv, cross_section):
    x, y, z = cross_section["x"], cross_section["y"], cross_section["z"]
    # provide a short piv object
    return piv.velocimetry.get_transect(x, y, z)


@pytest.fixture
def cli_obj():
    """Yield a click.testing.CliRunner to invoke the CLI."""
    class_ = CliRunner

    def invoke_wrapper(f):
        """Augment CliRunner.invoke to emit its output to stdout.

        This enables pytest to show the output in its logs on test
        failures.

        """
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            echo = kwargs.pop('echo', False)
            result = f(*args, **kwargs)

            if echo is True:
                sys.stdout.write(result.output)
            return result
        return wrapper
    class_.invoke = invoke_wrapper(class_.invoke)
    cli_runner = class_()
    yield cli_runner


@pytest.fixture
def recipe(recipe_yml):
    from pyorc.cli import cli_utils
    return cli_utils.parse_recipe("a", "b", recipe_yml)


@pytest.fixture
def recipe_geojson(recipe_yml):
    from pyorc.cli import cli_utils
    recipe = cli_utils.parse_recipe("a", "b", recipe_yml)
    for t in recipe["transect"]:
        if t != "write":
            with open(recipe["transect"][t]["shapefile"]) as f:
                data = json.load(f)
                recipe["transect"][t]["geojson"] = data
                del recipe["transect"][t]["shapefile"]
    # remove the file reference to transects and replace by already read json
    return recipe



# @pytest.fixture
# def velocity_flow_processor(recipe, vid_file, cam_config_fn, cli_prefix, cli_output_dir):
#     return pyorc.service.VelocityFlowProcessor(
#         recipe,
#         vid_file,
#         cam_config_fn,
#         cli_prefix,
#         cli_output_dir
#     )
