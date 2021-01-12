import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import rasterio.transform
from shapely.geometry import LineString, Point
from shapely.affinity import rotate
from shapely.ops import nearest_points

from OpenRiverCam import io
import math

folder = r"/home/hcwinsemius/projects/OpenRiverCam/piv"
src = os.path.join(folder, "velocity.nc")


def distance_pts(pt1, pt2):
    v = Point(pt2.x - pt1.x, pt2.y - pt1.y)
    dv = math.sqrt(v.x ** 2 + v.y ** 2)
    return dv


def snap(pt, pt1, pt2):
    # type: (Point, Point, Point) -> Point
    v = Point(pt2.x - pt1.x, pt2.y - pt1.y)
    dv = distance_pts(pt1, pt2)
    bh = ((pt.x - pt1.x) * v.x + (pt.y - pt1.y) * pt2.y) / dv
    h = Point(pt1.x + bh * v.x / dv, pt1.y + bh * v.y / dv)
    if 0 <= (pt1.x - h.x) / (pt2.x - h.y) < 1:
        # in the line segment
        return h
    elif distance_pts(h, pt1) < distance_pts(h, pt2):
        # near pt1
        return pt1
    else:
        # near pt2
        return pt2


def vector_to_scalar(
    ds_points,
    line_extend=0.1,
):
    xs = ds_points["x"].values
    ys = ds_points["y"].values
    # find points that are located on the area of interest
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    # check if x should be predictor
    x_pred = np.max(xs) - np.min(xs) > np.max(ys) - np.min(ys)
    if x_pred:
        # choose x as predictors
        predictor = xs
        predictant = ys
    else:
        predictant = xs
        predictor = ys

    # make a LineString with the most extreme points on the line
    slope, offset = np.polyfit(predictor, predictant, 1)
    range = np.max(predictor) - np.min(predictor)

    # extend the line slightly to ensure it encapsulates the entire cross section
    samples = np.array(
        [
            np.min(predictor) - range * line_extend,
            np.max(predictor) + range * line_extend,
        ]
    )
    predicted = samples * slope + offset
    if x_pred:
        line = LineString(zip(samples, predicted))
    else:
        line = LineString(zip(predicted, samples))

    # make a set of shapely Points out of the coordinates lying in the area of interest
    coords = [Point(x, y) for x, y in zip(xs, ys)]

    # snap the coordinates to the cross section straight line
    coords = [nearest_points(line, p)[0] for p in coords]

    # compute the angle of the cross section line assuming the first coordinate is left bank
    left_x, left_y = coords[0].x, coords[0].y
    right_x, right_y = coords[-1].x, coords[-1].y
    diff_x, diff_y = right_x - left_x, right_y - left_y
    angle_cross_section = np.arctan2(diff_x, diff_y)

    # compute angle of flow direction (i.e. the perpendicular of the cross section)
    flow_dir = angle_cross_section - 0.5 * np.pi
    # compute per velocity vector in the dataset, what its angle is
    v_angle = np.arctan2(ds_points["v_x"], ds_points["v_y"])
    # compute the scalar value of velocity
    v_scalar = (ds_points["v_x"] ** 2 + ds_points["v_y"] ** 2) ** 0.5

    # compute difference in angle between velocity and perpendicular of cross section
    angle_diff = v_angle - flow_dir

    # compute effective velocity in the flow direction (i.e. perpendicular to cross section
    v_eff = np.cos(angle_diff) * v_scalar
    v_eff.attrs = {
        "standard_name": "velocity",
        "long_name": "velocity in perpendicular direction of cross section, measured by angle in radians, measured from up-direction",
        "units": "m s-1",
        "angle": flow_dir,
    }
    return v_eff

def depth_average(ds_points, z_0, h_a, v_corr=0.85):
    # compute depth, never smaller than zero
    depth = np.maximum(z_0 + h_a - ds_points["zcoords"], 0)
    q_eff = ds_points["v_eff"] * v_corr * depth
    q_eff.attrs = {
        "standard_name": "velocity_depth",
        "long_name": "velocity averaged over depth",
        "units": "m2 s-1",
    }
    return q_eff



# define cross section
x = np.flipud(
    [
        192087.75935984775424,
        192087.637668401439441,
        192087.515976955008227,
        192087.345608930045273,
        192087.223917483701371,
        192087.077887748018838,
        192086.980534590955358,
        192086.85884314449504,
        192086.761489987518871,
        192086.566783673217287,
        192086.469430516182911,
        192086.323400780500378,
        192086.177371044788742,
        192086.080017887725262,
        192085.885311573481886,
        192085.690605259296717,
        192085.641928680706769,
    ]
)
y = np.flipud(
    [
        313193.458014087867923,
        313194.09080960357096,
        313194.820958279015031,
        313195.478092092089355,
        313196.183902480057441,
        313196.962727739592083,
        313197.741552993596997,
        313198.374348516052123,
        313199.055820613284595,
        313199.785969295422547,
        313200.491779681993648,
        313201.197590066993143,
        313201.95207703957567,
        313202.730902298353612,
        313203.363697814638726,
        313204.166861362638883,
        313204.970024902664591,
    ]
)
z = np.flipud(
    [
        101.88,
        101.67,
        101.3,
        101.0,
        100.49,
        100.34,
        100.0,
        100.2,
        100.6,
        100.9,
        101.05,
        101.0,
        101.1,
        101.3,
        101.5,
        102.4,
        102.5,
    ]
)
coords = list(zip(x, y, z))


c_s = {
    "site": "Hommerich",  # dict, site, relational, because a bathymetry profile belongs to a site.
    "crs": 28992,  # int, epsg code in [m], only projected coordinate systems are supported
    "coords": coords,  # list of (x, y, z) tuples defined in crs [m], coords are not valid in the example
}

# ref
z_0 = 100.0  # m this is the zero level water depth as measured with gps
h_a = 0.9  # actual water level during movie
v_corr = 0.85  # ratio between average velocity over vertical, and surface velocity (used as correction factor)
# open dataset

# This is the main functionality tested: extract cross section from points
ds_points = io.interp_coords(src, *zip(*c_s["coords"]))


# This is secondary functionality tested: integrate to vertically integrated flow


# now compute effective velocity in thge direction of flow
ds_points["v_eff"] = vector_to_scalar(ds_points)
ds_points["q_eff"] = depth_average(ds_points, z_0, h_a, v_corr)

# now compute the depth integrated velocity
# some plotting to check
ds = xr.open_dataset(src)

# ds["v_x"][0].plot()
# ds_points["v_x"].plot()
# plt.plot(x, y, '.')
# plt.pcolormesh(
#     ds["x_grid"], ds["y_grid"], ds["v_x"][0], shading="auto", vmin=0.0, vmax=1.5
# )
# plt.scatter(ds_points.xcoords, ds_points.ycoords, c=ds_points.zcoords, edgecolor="w")
# plt.colorbar()
# v_x = ds_points["v_x"].values
# v_x[v_x<0.01] = np.nan
#
# ds_points["v_x"] = (["time", "points"], v_x)

# coords = [line.intersection(rotate(LineString([(p.x-dx, p.y-dy), (p.x+dx, p.y+dy)]), 90, p)) for p in coords]
ds_points["q_eff"].quantile([0.1, 0.5, 0.9], dim="time").plot.line(x="points")

# plt.plot(*zip(*line.coords))
# plt.plot(xs, ys, '.')
# for c in coords:
#     plt.plot(c.x, c.y, 'r.')
#
# plt.axis("equal")


# ds_points["v_x"].quantile([0.1, 0.5, 0.9], dim="time").plot.line(x="points")
# q_points.quantile([0.1, 0.5, 0.9], dim="time").plot.line(x="points")


# plt.plot(ds_points.xcoords, ds_points.ycoords, '.')
plt.show()
