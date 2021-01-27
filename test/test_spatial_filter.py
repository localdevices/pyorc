import os
import xarray as xr
from openpiv import validation
import matplotlib.pyplot as plt

def local_median_val(ds):
    v_x, v_y = ds["v_x"].values, ds["v_y"].values
    u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds["v_x"][:] = u
    ds["v_y"][:] = v
    return ds

folder = r"/home/hcwinsemius/Media/projects/OpenRiverCam/piv"
src = os.path.join(folder, "velocity_filter_func.nc")
# out_fn = os.path.join(folder, "velocity_filter_func.nc")

ds = xr.open_dataset(src)
ds_g = ds.groupby("time")

a = ds_g.apply(local_median_val)


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