import os
import xarray as xr
from openpiv import validation
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import numpy as np

def local_median_val(ds):
    v_x, v_y = ds["v_x"].values, ds["v_y"].values
    u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds["v_x"][:] = u
    ds["v_y"][:] = v
    return ds

def spatial_filter(ds, max_nan_frac=0.8, stride=1):
    v_x, v_y = ds["v_x"].values, ds["v_y"].values
    v_x_move = []
    v_y_move = []
    # prepare convolution arrays that cover all possible shifts of the raster to cover the provided stride

    for vert in range(-stride, stride+1):
        for horz in range(-stride, stride+1):
            conv_arr = np.zeros((abs(vert)*2+1, abs(horz)*2+1))
            _y = int(np.floor((abs(vert)*2+1)/2)) + vert
            _x = int(np.floor((abs(horz)*2+1)/2)) + horz
            conv_arr[_y, _x] = 1
            v_x_move.append(convolve2d(v_x, conv_arr, mode="same"))
            v_y_move.append(convolve2d(v_y, conv_arr, mode="same"))
    v_x_move = np.stack(v_x_move)
    v_y_move = np.stack(v_y_move)
    nan_frac = np.float64(np.isnan(v_x_move)).sum(axis=0)/float(len(v_x_move))
    v_x[nan_frac > max_nan_frac] = np.nan
    v_y[nan_frac > max_nan_frac] = np.nan


    # u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds["v_x"][:] = v_x
    ds["v_y"][:] = v_y
    return ds

folder = r"/home/hcwinsemius/Media/projects/OpenRiverCam/piv"
src = os.path.join(folder, "velocity_filter_func.nc")
# out_fn = os.path.join(folder, "velocity_filter_func.nc")

ds = xr.open_dataset(src)
ds_g = ds.groupby("time")


a = ds_g.apply(spatial_filter, **{"max_nan_frac": 0.87})


v_x, v_y = ds["v_x"][0].values, ds["v_y"][0].values
# first on one frame
f = plt.figure(figsize=(15, 8))
ax1 = plt.subplot(111)
ax1.quiver(v_x, v_y, scale=50)
ax1.invert_yaxis()
v_x, v_y = ds["v_x"][0].values, ds["v_y"][0].values

v_x, v_y = a["v_x"][0].values, a["v_y"][0].values

# u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
# polot sometinh
# ax2 = plt.subplot(122)
ax1.quiver(v_x, v_y, scale=50, color='b')
# ax2.invert_yaxis()

# plot another thing
plt.show()

