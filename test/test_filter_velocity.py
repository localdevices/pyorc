import os
import xarray as xr
import matplotlib.pyplot as plt
import openpiv.validation
from OpenRiverCam import piv, io
import cv2
import numpy as np

import openpiv.tools
folder = r"/home/hcwinsemius/Media/projects/OpenRiverCam/piv"
src = os.path.join(folder, "velocity.nc")


fn1 = os.path.join(folder, "..", "ortho_proj", "proj_0001_000004.tif")
fn2 = os.path.join(folder, "..", "ortho_proj", "proj_0002_000008.tif")


frame_a = openpiv.tools.imread(fn1)
frame_b = openpiv.tools.imread(fn2)


# frame_a = cv2.imread(fn1)
# frame_b = cv2.imread(fn2)

# frame_a = io._corr_color(frame_a, alpha=None, beta=None, gamma=0.4)
# frame_b = io._corr_color(frame_b, alpha=None, beta=None, gamma=0.4)

# cols, rows, _u, _v, _sig2noise = piv.piv(
#     frame_a,
#     frame_b,
#     res_x=0.01,
#     res_y=0.01,
#     window_size=50,
#     sig2noise_method="peak2mean",
#     search_area_size=50,
#     overlap=15,
#     dt=0.004,
# )

# open file for reading

ds = xr.open_dataset(src)

# make a bunch of good plots

# n_frames = 4  # amount of frame pairs to plot
# d_frames = 2  # amount of frame pairs to skip for each plot

angle = np.arctan2(ds["v_x"], ds["v_y"])
angle_mean = angle.mean(dim="time")
angle_std = angle.std(dim="time")
# angle_mean.plot(vmin=0.25*np.pi, vmax=0.75*np.pi)
# _u, _v, mask = openpiv.validation.sig2noise_val(_u, _v, _sig2noise, threshold = 1.1 )
#
# filter 1: angle more or less in left-right direction
ds["v_x"] = ds["v_x"].where(angle_mean > 0.25*np.pi).where(angle_mean < 0.75*np.pi)
ds["v_y"] = ds["v_y"].where(angle_mean > 0.25*np.pi).where(angle_mean < 0.75*np.pi)
#
s = (ds["v_x"]**2 + ds["v_x"]**2)**0.5
s_std = s.std(dim="time")
s_mean = s.mean(dim="time")
s_var = s_std/s_mean

ds["v_x"] = ds["v_x"].where(s_var > 0.5)
ds["v_y"] = ds["v_y"].where(s_var > 0.5)

# filter_2 std/mean


# _u[s<0.025] = np.nan
# _v[s<0.025] = np.nan
# # plt.imshow(_sig2noise, vmin=1., vmax=1.5, alpha=0.5, cmap="Greys");plt.colorbar()
# # # plt.imshow(frame_a) #, cmap="Greys")
_u, _v, s2n = ds["v_x"].mean(dim="time").values, ds["v_y"].mean(dim="time").values, ds["s2n"][1].values
_u, _v, s2n = ds["v_x"][0].values, ds["v_y"][0].values, ds["s2n"][0].values
# plt.imshow(s2n, vmin=1., vmax=1.5);plt.colorbar()
# plt.imshow(angle_std.values,vmin=0.6);plt.colorbar()
plt.imshow(s_var.values);plt.colorbar()

# angle_list = angle_std.values.flatten()
# angle_list[angle_list<0.002] = np.nan
# plt.hist(angle_list, bins=25)

plt.quiver(_u, _v, color='r')
# plt.imshow(mask) # # , vmin=1.0, vmax=1.1);


# ds.s2n[0].plot(vmax=1.3, vmin=1.)  # plot
plt.show()


