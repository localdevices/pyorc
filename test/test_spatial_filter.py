import os
import xarray as xr
import openpiv.filters
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import numpy as np

def neighbour_stack(array, stride=1, missing=-9999.):
    """
    Builds a stack of arrays from a 2-D input array, with its neighbours using a provided stride
    :param array: 2-D numpy array with values (may contain NaN)
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return:
    stack of arrays, with all neighbours included

    """
    array = np.copy(array)
    array[np.isnan(array)] = missing
    array_move = []
    for vert in range(-stride, stride+1):
        for horz in range(-stride, stride+1):
            conv_arr = np.zeros((abs(vert)*2+1, abs(horz)*2+1))
            _y = int(np.floor((abs(vert)*2+1)/2)) + vert
            _x = int(np.floor((abs(horz)*2+1)/2)) + horz
            conv_arr[_y, _x] = 1
            array_move.append(convolve2d(array, conv_arr, mode="same"))
    array_move = np.stack(array_move)
    # replace missings by Nan
    array_move[np.isclose(array_move, missing)] = np.nan
    return array_move

# def local_median_val(ds):
#     v_x, v_y = ds["v_x"].values, ds["v_y"].values
#     u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
#     ds["v_x"][:] = u
#     ds["v_y"][:] = v
#     return ds

def spatial_nan_filter(ds, max_nan_frac=0.8, stride=1, missing=-9999.):
    v_x, v_y = ds["v_x"].values, ds["v_y"].values
    v_x_move = neighbour_stack(v_x, stride=stride)
    # replace missings by Nan
    nan_frac = np.float64(np.isnan(v_x_move)).sum(axis=0)/float(len(v_x_move))
    v_x[nan_frac > max_nan_frac] = np.nan
    v_y[nan_frac > max_nan_frac] = np.nan

    # u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds["v_x"][:] = v_x
    ds["v_y"][:] = v_y
    return ds

def spatial_median_filter(ds, max_rel_diff=0.5, stride=1, missing=-9999.):
    v_x, v_y = ds["v_x"].values, ds["v_y"].values
    v = (v_x**2 + v_y**2)**0.5
    v_move = neighbour_stack(v, stride=stride)
    # replace missings by Nan
    v_median = np.nanmedian(v_move, axis=0)
    # now filter points that are very far off from the median
    filter = np.abs(v - v_median)/v_median > max_rel_diff
    v_x[filter] = np.nan
    v_y[filter] = np.nan

    # u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds["v_x"][:] = v_x
    ds["v_y"][:] = v_y
    return ds

def replace_outliers(ds, stride=1, max_iter=1):
    v_x, v_y = ds["v_x"].values, ds["v_y"].values
    for n in range(max_iter):
        v_x_move = neighbour_stack(v_x, stride=stride)
        v_y_move = neighbour_stack(v_y, stride=stride)
        # compute mean
        v_x_mean = np.nanmean(v_x_move, axis=0)
        v_y_mean = np.nanmean(v_y_move, axis=0)
        v_x[np.isnan(v_x)] = v_x_mean[np.isnan(v_x)]
        v_y[np.isnan(v_y)] = v_y_mean[np.isnan(v_y)]
        # all values with stride distance from edge have to be made NaN
        v_x[0:stride, :] = np.nan; v_x[-stride:, :] = np.nan; v_x[:, 0:stride] = np.nan; v_x[:, -stride:] = np.nan;
        v_y[0:stride, :] = np.nan;v_y[-stride:, :] = np.nan;v_y[:, 0:stride] = np.nan;v_y[:, -stride:] = np.nan;
    ds["v_x"][:] = v_x
    ds["v_y"][:] = v_y
    return ds

folder = r"/home/hcwinsemius/Media/projects/pyorc/piv"
src_1 = os.path.join(folder, "velocity.nc")
src_2 = os.path.join(folder, "velocity_filter_func.nc")
# out_fn = os.path.join(folder, "velocity_filter_func.nc")

ds1 = xr.open_dataset(src_1)
ds2 = xr.open_dataset(src_2)
# ds_g = ds.groupby("time")
#
#
# # a = ds_g.apply(spatial_nan_filter, **{"max_nan_frac": 0.76})
# # ds_g = a.groupby("time")
# a = ds_g.apply(spatial_median_filter, **{"max_rel_diff": 0.5})
# ds_g = a.groupby("time")
# a = ds_g.apply(replace_outliers, **{"max_iter": 3, "stride": 1 })

v_x1, v_y1 = ds1["v_x"].values, ds1["v_y"].values
v_x2, v_y2 = ds2["v_x"].values, ds2["v_y"].values# v_x = ds["v_x"][:, 35, 55]
# first on one frame
f = plt.figure(figsize=(10, 6))
ax1 = plt.subplot(111)
ax1.hist(v_x1[:, 35, 55], np.linspace(-3, 3, 61), zorder=0)
print(np.nanmedian(v_x1[:, 35, 55]))

ax1.hist(v_x2[:, 35, 55], np.linspace(-3, 3, 61), zorder=1)
print(np.nanmedian(v_x2[:, 35, 55]))
plt.xlabel("velocity [m s-1]")
plt.ylabel("No. of occurrences [-]")
# plot another thing
plt.show()

