import os
import numpy as np
import xarray as xr
from orc import io, piv_process
import matplotlib.pyplot as plt

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

folder = r"/home/hcwinsemius/Media/projects/orc/piv"
src = os.path.join(folder, "velocity_filter.nc")
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
ds = xr.open_dataset(src)

xs = np.array(c_s["coords"])[:, 0]
ys = np.array(c_s["coords"])[:, 1]
zs = np.array(c_s["coords"])[:, 2]


# ds_points = io.interp_coords(ds, *zip(*c_s["coords"]))
ds_points = io.interp_coords(ds, xs=xs, ys=ys, zs=zs)

# This is secondary functionality tested: integrate to vertically integrated flow

# now compute effective velocity in thge direction of flow
ds_points["v_eff"] = piv_process.vector_to_scalar(ds_points["v_x"], ds_points["v_y"])

# integrate over depth with vertical correction
ds_points["q"] = piv_process.depth_integrate(
    ds_points["zcoords"], ds_points["v_eff"], z_0, h_a, v_corr
)

# compute Q per time step
# inputs
Q = piv_process.integrate_flow(ds_points["q"], quantile=np.linspace(0.01, 0.99, 99))


print(Q)
# Q.plot()

# q = ds_points["q"]
#
# quantile = [0.1, 0.5, 0.9]
#
# q = q.quantile(quantile, dim="time")
#
# # for x, y in zip(q.xcoords.values, q.ycoords.values)
# # depth_av = [0.]
# # v_av = [0.]
#
# dist = [0.]
#
# for n, (x1, y1, x2, y2) in enumerate(zip(q.xcoords[:-1], q.ycoords[:-1], q.xcoords[1:], q.ycoords[1:])):
#     _dist = distance_pts((x1, y1), (x2, y2))
#     dist.append(dist[n] + _dist)
#
# # assign coordinates for distance
# ds_points = ds_points.assign_coords(dist=("points", dist))
# Q = ds_points["q"].quantile(quantile, dim="time").integrate(dim="dist")

# for n, (q1, q2, depth1, depth2) in enumerate(zip(v_d[:-1], v_d[1:], depth[:-1], depth[1:])):
#     _dist = distance_pts((q1.xcoords, q1.ycoords), (q2.xcoords, q2.ycoords))
#     _depth_av = np.nanmean([depth1, depth2])
#     _v_av = np.nanmean([q1, q2])
#     dist.append(np.mean([_dist]))
#     depth_av.append(_depth_av)
#     v_av.append(_v_av)

# # make arrays
# dist, depth_av, v_av = np.array(dist), np.array(depth_av), np.array(v_av)
#
# q = v_av*depth_av*dist
#
# # FUNCTION ENDS HERE ADD INFO TO ds_points
# ds_points["q"] = ("points", q)


# make temporary dataset


# some plotting to check

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
ds_points["v_eff"].quantile([0.1, 0.5, 0.9], dim="time").plot.line(x="points")

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
