import openpiv.tools
import openpiv.pyprocess
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime, timedelta
from OpenRiverCam import io, piv

folder = r"/home/hcwinsemius/Media/projects/OpenRiverCam"
start_time = "2020-12-17 07:45:00"
t = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

dst_crs = rasterio.crs.CRS.from_epsg(4326)  # lat lon projection
src = os.path.join(folder, "ortho_proj")
dst = os.path.join(folder, "piv")

if not (os.path.isdir(dst)):
    os.makedirs(dst)
dst_fn = os.path.join(dst, "velocity.nc")

fns = glob.glob(os.path.join(src, "*.tif"))
fns.sort()
print(fns)
window_size = 30
search_area_size = 30
overlap = 15
dt = 1.0 / 25
res_x = 0.01
res_y = 0.01
correlation = True
correlation_method = "circular"
normalized_correlation = True

u = []
v = []
sig2noise = []
corr = []
time = []
# for n in range(10):
for n in range(len(fns) - 1):
    print(f"Treating frame {n}")
    frame_a = openpiv.tools.imread(fns[n])
    frame_b = openpiv.tools.imread(fns[n + 1])
    cols, rows, _u, _v, _sig2noise, _corr = piv.piv(
        frame_a,
        frame_b,
        res_x=res_x,
        res_y=res_y,
        # window_size=window_size,
        sig2noise_method="peak2peak",
        search_area_size=search_area_size,
        overlap=overlap,
        dt=dt,
    )

    # time
    ms = timedelta(milliseconds=int(fns[n][-10:-4]))
    time.append(t + ms)

    u.append(_u)
    v.append(_v)
    sig2noise.append(_sig2noise)
    corr.append(_corr)

var_names = ["v_x", "v_y", "s2n", "corr"]
var_attrs = [
    {
        "standard_name": "sea_water_x_velocity",
        "long_name": "Flow element center velocity vector, x-component",
        "units": "m s-1",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
    {
        "standard_name": "sea_water_y_velocity",
        "long_name": "Flow element center velocity vector, y-component",
        "units": "m s-1",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
    {
        "standard_name": "ratio",
        "long_name": "signal to noise ratio",
        "units": "",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
    {
        "standard_name": "correlation_coefficient",
        "long_name": "correlation coefficient between frames",
        "units": "",
        # "grid_mapping": "projected_coordinate_system",
        "coordinates": "lon lat",
    },
]
encoding = {var: {"zlib": True} for var in var_names}

xs, ys, lons, lats = io.convert_cols_rows(fns[0], cols, rows)
spacing_x = np.diff(cols[0])[0]
spacing_y = np.diff(rows[:, 0])[0]
x = np.linspace(
    res_x / 2 * spacing_x, (len(cols[0]) - 0.5) * res_x * spacing_x, len(cols[0])
)
y = np.flipud(
    np.linspace(
        res_y / 2 * spacing_y,
        (len(rows[:, 0]) - 0.5) * res_y * spacing_y,
        len(rows[:, 0]),
    )
)
dataset = io.to_dataset(
    [u, v, sig2noise, corr],
    var_names,
    x,
    y,
    time=time,
    lat=lats,
    lon=lons,
    xs=xs,
    ys=ys,
    attrs=var_attrs,
)

# add lat and lon
dataset.to_netcdf(dst_fn, encoding=encoding)

# frame_cv = cv2.imread(fns[0])
#
# plt.pcolormesh(xi, yi, band)
# plt.imshow(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
# plt.quiver(xq, yq, u[0], v[0], color='r')
# plt.show()
