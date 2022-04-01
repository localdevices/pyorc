import numpy as np
from shapely.geometry import Polygon
import pyorc
import pyproj
camera_type = {
    "name": "Foscam E9900P",  # user-chosen name for camera
    "lens_parameters": {  # the lens parameters need to be known or calibrated
        "k1": -3.0e-6,
        "c": 2,
        "f": 4.0,
    },
}
gcps = {
    "src": [
        [253, 307],
        [931, 202],
        [1308, 337],
        [1251, 859]
    ],
        "dst": [
        [5.913401333, 50.8072278333],
        [5.9133098333, 50.807340333],
        [5.9134281667, 50.8073605],
        [5.9135723333, 50.807271833],
    ],
    "z_0": 138.4,  # reference height of zero water level compared to the crs used
    "h_ref": 0.05,  # actual water level during taking of the gcp points
    "crs": 4326
}

# corner points provided by user in pixel coordinates, starting at upstream-left, downstream-left, downstream-right, upstream-right
corners = [
    [190, 205],
    [953, 143],
    [1735, 480],
    [1290, 914],
]

lens_position = [ 5.9136175, 50.807232333333, 143.1]

lons = list(np.flipud(np.array([
    5.913656,
    5.91365083333333,
    5.913643,
    5.91363616666667,
    5.91362916666667,
    5.91362416666667,
    5.91362016666667,
    5.91361516666667,
    5.91361083333333,
    5.91360916666667,
    5.913608,
    5.91360433333333,
    5.91360033333333,
    5.91359666666667,
    5.913591,
    5.91358466666667,
    5.91357866666667,
    5.91357483333333,
    5.91357133333333,
    5.91356883333333,
    5.9135625,
    5.91355666666667,
    5.9135495,
    5.91354166666667,
    5.91353316666667,
    5.913522,
    5.91351116666667,
    5.913501,
    5.91349366666667,
    5.91348466666667,
    5.91347433333333,
    5.9134625,
    5.91345116666667,
    5.9134415,
    5.91343133333333,
    5.9134175,
    5.91340616666667,
    5.91339466666667,
    5.91338716666667,
    5.91338083333333,
    5.91337616666667,
    5.913372,
    5.913366,
    5.91335816666667,
    5.91335033333333,
    5.9133445,
    5.91334033333333,
    5.91333666666667,
    5.91333283333333,
    5.91332716666667,
    5.91332,
    5.913313,
    5.91330683333333,
    5.91329983333333,
    5.91329283333333,
    5.91328883333333,
    5.91328266666667,
    5.91327583333333,
    5.91326916666667,
])))

lats = list(np.flipud(np.array([
    50.807302,
    50.8073003333333,
    50.8072993333333,
    50.8072978333333,
    50.8072971666667,
    50.8072961666667,
    50.8072956666667,
    50.8072946666667,
    50.8072943333333,
    50.8072936666667,
    50.8072928333333,
    50.8072918333333,
    50.8072908333333,
    50.8072891666667,
    50.8072878333333,
    50.8072865,
    50.807284,
    50.8072825,
    50.8072816666667,
    50.8072808333333,
    50.8072796666667,
    50.8072783333333,
    50.807276,
    50.8072741666667,
    50.807272,
    50.8072695,
    50.8072671666667,
    50.807265,
    50.8072625,
    50.80726,
    50.8072578333333,
    50.8072563333333,
    50.8072558333333,
    50.807254,
    50.8072528333333,
    50.8072508333333,
    50.8072496666667,
    50.8072476666667,
    50.8072465,
    50.8072455,
    50.8072451666667,
    50.8072435,
    50.8072421666667,
    50.8072413333333,
    50.8072405,
    50.8072403333333,
    50.8072396666667,
    50.8072395,
    50.8072391666667,
    50.8072386666667,
    50.8072378333333,
    50.8072363333333,
    50.8072358333333,
    50.807235,
    50.8072348333333,
    50.807235,
    50.8072343333333,
    50.8072333333333,
    50.8072325,
])))
z = list(np.flipud(np.array([
    141,
    140.9,
    140.9,
    140.8,
    140.7,
    140.6,
    140.5,
    140.4,
    140.3,
    140.2,
    139.6,
    139.4,
    139.3,
    139.1,
    139.1,
    139,
    139.3,
    139,
    138.9,
    138.4,
    138.3,
    138.3,
    138.4,
    138.3,
    138.3,
    138.3,
    138.4,
    138.4,
    138.4,
    138.4,
    138.4,
    138.4,
    138.4,
    138.4,
    138.3,
    138.4,
    138.4,
    138.4,
    138.4,
    138.4,
    138.4,
    138.7,
    138.8,
    138.9,
    139,
    139.1,
    139.3,
    139.6,
    139.8,
    140,
    140,
    140.2,
    140.3,
    140.6,
    140.8,
    141,
    141.1,
    141.2,
    141.2,
])))
# x = []
# y = []
# for lon, lat in zip(lons, lats):
#     _x, _y = transform.transform(lon, lat)
#     x.append(_x)
#     y.append(_y)
#
# coords = list(zip(x, y, z))
#
# # make coords entirely jsonifiable by getting rid of tuple construct
# coords = [list(c) for c in coords]
#
# bathymetry = {
#     "crs": 32631,  # int, epsg code in [m], only projected coordinate systems are supported
#     "coords": coords,  # list of (x, y, z) tuples defined in crs [m], coords are not valid in the example
# }
#
# # # transform lens position
# # _x, _y = transform.transform(lensPosition[0], lensPosition[1])
# # lensPosition[0] = _x
# # lensPosition[1] = _y
#
# movie = {
#     "id": 1,
#     "type": "normal",  # str, defines what the movie is used for, either "configuration" or "normal"
#     "camera_config": camera_config,  # dict, camera_config object, relational, because a movie belongs to a given camera_config (which in turn belongs to a site).
#     "file": {  # file contains the actual filename, and the bucket in which it sits.
#         "bucket": "example",
#         "identifier": "example_video.mp4",
#     },
#     "timestamp": "2021-01-01T00:05:30Z",
#     "resolution": "1920x1080",
#     "fps": 25.862,  # float
#     "bathymetry": bathymetry,
#     "h_a": 0.1,  # float, water level with reference to gauge plate zero level
# }
#
