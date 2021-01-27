import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from OpenRiverCam import piv
import cv2


def movie(f, ax, fns, ds, movie_fn, video_args):
    def init():
        # im_data = openpiv.tools.imread(fns[0])
        im_data = cv2.imread(fns[0])
        im.set_data(np.zeros(im_data.shape))
        _u = ds["v_x"][0].values
        # line.set_data([], [])
        q.set_UVC(np.zeros(_u.shape), np.zeros(_u.shape))
        return ax  # line,

    def animate(i):
        print(i)
        im_data = cv2.imread(fns[i + 1])
        # im_data = openpiv.tools.imread(fns[i+1])
        _u, _v = ds["v_x"][i].values, ds["v_y"][i].values
        im.set_data(im_data)
        q.set_UVC(_u, _v)

        # point.set_data(x[idx_line], y[idx_line])
        return ax

    im_data = cv2.imread(fns[0])
    im = ax.imshow(im_data)
    x, y = ds["x"].values, ds["y"].values
    _u, _v = ds["v_x"][0].values, ds["v_y"][0].values
    # make a local mesh
    xi, yi = np.meshgrid(x / 0.01, np.flipud(y) / 0.01)
    q = ax.quiver(xi, yi, _u, _v, color="w", alpha=0.5, scale=100, width=0.0015)

    anim = FuncAnimation(
        f, animate, init_func=init, frames=len(fns)-1, interval=20, blit=False
    )  # interval=40 defaults to 25fps (40ms per frame)
    anim.save(movie_fn, **video_args)


import numpy as np

folder = r"/home/hcwinsemius/Media/projects/OpenRiverCam/piv"
src = os.path.join(folder, "velocity.nc")
out_fn = os.path.join(folder, "velocity_filter_func.nc")


# set the arguments for storing video (used by anim.save(...))
video_args = {
    "fps": 10,
    "extra_args": ["-vcodec", "libx264"],
    #               'extra_args': ['-pix_fmt', 'yuv420p'],
    "dpi": 120,
}
fns = glob.glob(os.path.join(folder, "..", "ortho_proj_color", "proj*.*"))
fns.sort()


# open file for reading
ds = xr.open_dataset(src)

ds = piv.piv_filter(ds)

ds.to_netcdf(out_fn)
angle = np.arctan2(ds["v_x"], ds["v_y"])
angle_mean = angle.mean(dim="time")
# ds["v_x"].where(np.abs(angle - angle_mean) < 0.25*np.pi)
#
# animation
f = plt.figure(figsize=(16, 9), frameon=False)
f.set_size_inches(16, 9, True)
f.patch.set_facecolor("k")
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
ax = plt.subplot(111)

movie_fn = "example_PIV_high_res_func.mp4"
movie(f, ax, fns, ds, movie_fn, video_args=video_args)
