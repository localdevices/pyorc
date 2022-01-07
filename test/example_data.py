import numpy as np
from shapely.geometry import Polygon
import orc as ORC
import pyproj
camera_type = {
    "name": "Foscam E9900P",  # user-chosen name for camera
    "lensParameters": {  # the lens parameters need to be known or calibrated
        "k1": -3.0e-6,
        "c": 2,
        "f": 4.0,
    },
}
gcps = {
    "src": [
        [339, 435],
        [1366, 349],
        [1327, 628],
        # [1935, 1590],
        [225, 702]],
    "dst": [
        [527861.115626636426896, 9251763.739461150020361],
        [527871.161038133315742, 9251764.629861138761044],
        [527870.453648019232787, 9251761.846481902524829],
        [527862.707104541477747, 9251760.572029870003462],
    ],
    "z_0": -17.50,  # reference height of zero water level compared to the crs used
    "h_ref": 0.1,  # actual water level during taking of the gcp points
}

# corner points provided by user in pixel coordinates, starting at upstream-left, downstream-left, downstream-right, upstream-right
corners = {
    "up_left": [0, 250],
    "down_left": [1650, 150],
    "down_right": [1750, 1200],
    "up_right": [153, 1200],
}


site = {
    "name": "Chuo Kikuu - Senga Road",  # str, name of user
    "uuid": "",  # some uuid for relational database purposes
    "position": (
        39.2522,
        -6.7692,
    ),  # approximate geographical location of site in crs (x, y) coordinates in metres.
    "crs": 32737,  # int, coordinate ref system as EPSG code
}

crs_site = pyproj.CRS.from_epsg(site["crs"])
crs_latlon = pyproj.CRS.from_epsg(4326)
transform = pyproj.Transformer.from_crs(crs_latlon, crs_site, always_xy=True)
# make a polygon from corner points, print it to see what it looks like.
src_polygon = Polygon([corners[s] for s in corners])
print(src_polygon)

# this function prepares a bounding box, from 4 user selected corner points, print to see what it looks like
bbox = ORC.cv.get_aoi(gcps["src"], gcps["dst"], corners)
print(bbox)

lensPosition = [ 527869.47, 9251757.48, -14.38]

camera_config = {
    "id": 1,
    "camera_type": camera_type,  # dict, camera object, relational, because a camera configuration belongs to a certain camera.
    "site": site,  # dict, site object, relational because we need to know to whcih site a camera_config belongs. you can have multiple camera configs per site.
    "time_start": "2020-12-16T00:00:00",  # start time of valid range
    "time_end": "2020-12-31T00:00:00",  # end time of valid range, can for instance be used to find the right camera config with a given movie
    "gcps": gcps,  # dict, gcps dictionary, see above
    "corners": corners,  # dict containining corner pixel coordinates, see above
    "resolution": 0.01,  # resolution to be used in reprojection to AOI
    "lensPosition": lensPosition,  # we could also make this a geojson but it is just one point (x, y, z)
    "aoi_bbox": bbox,
    "aoi_window_size": 15,

}

lons = list(np.array([
    39.252163365,
    39.2521632766667,
    39.2521634916667,
    39.2521630066667,
    39.25216389,
    39.252165055,
    39.25216593,
    39.2521661433333,
    39.2521668333333,
    39.252167195,
    39.25216767,
    39.252167945,
    39.2521694266667,
    39.2521713366667,
    39.25217169,
    39.2521741866667,
    39.2521748333333,
    39.2521762533333,
    39.2521762533333,
    39.252176735,
    39.2521767583333,
    39.2521767833333,
    39.2521762166667,
    39.252176145,
    39.2521764333333,
    39.25217638,
    39.2521767816667,
    39.2521772533333,
    39.252177765,
    39.2521777033333,
    39.2521777766667,
]))

lats = [
    -6.76912008666667,
    -6.76912268833333,
    -6.76912599166667,
    -6.769130115,
    -6.76913343333333,
    -6.76913659666667,
    -6.769140815,
    -6.769144665,
    -6.76914837166667,
    -6.769153975,
    -6.76915663833333,
    -6.76915825333333,
    -6.76916068333333,
    -6.76916559333333,
    -6.769167805,
    -6.76917215,
    -6.76917394166667,
    -6.76917648166667,
    -6.76917648166667,
    -6.769178775,
    -6.76918126166667,
    -6.76918686666667,
    -6.76918829333333,
    -6.76919084,
    -6.76919256333333,
    -6.76919439166667,
    -6.76919715333333,
    -6.76920004166667,
    -6.76920142,
    -6.769204245,
    -6.769206575,
]
z = [
    -18.0319991607666,
    -17.9779991607666,
    -17.9489991607666,
    -17.8359991607666,
    -18.1069991607666,
    -18.0289991607666,
    -18.6139991607666,
    -19.2189991607666,
    -19.2069991607666,
    -19.2869991607666,
    -19.2779991607666,
    -19.2849991607666,
    -19.2729991607666,
    -19.2799991607666,
    -19.2429991607666,
    -19.2499991607666,
    -19.2489991607666,
    -19.2789991607666,
    -19.2789991607666,
    -19.2739991607666,
    -19.0549991607666,
    -18.6559991607666,
    -18.5659991607666,
    -18.2629991607666,
    -17.9069991607666,
    -17.8139991607666,
    -17.6219991607666,
    -17.3049991607666,
    -17.1069991607666,
    -17.1069991607666,
    -17.0409991607666,
]
x = []
y = []
for lon, lat in zip(lons, lats):
    _x, _y = transform.transform(lon, lat)
    x.append(_x)
    y.append(_y)

coords = list(zip(x, y, z))

# make coords entirely jsonifiable by getting rid of tuple construct
coords = [list(c) for c in coords]

bathymetry = {
    "crs": 32737,  # int, epsg code in [m], only projected coordinate systems are supported
    "coords": coords,  # list of (x, y, z) tuples defined in crs [m], coords are not valid in the example
}

movie = {
    "id": 1,
    "type": "normal",  # str, defines what the movie is used for, either "configuration" or "normal"
    "camera_config": camera_config,  # dict, camera_config object, relational, because a movie belongs to a given camera_config (which in turn belongs to a site).
    "file": {  # file contains the actual filename, and the bucket in which it sits.
        "bucket": "example",
        "identifier": "example_video.mp4",
    },
    "timestamp": "2021-01-01T00:05:30Z",
    "resolution": "1920x1080",
    "fps": 25.862,  # float
    "bathymetry": bathymetry,
    "h_a": 0.1,  # float, water level with reference to gauge plate zero level
}

