import openpiv.tools
import openpiv.pyprocess
import xarray as xr
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime, timedelta
from rasterio.plot import reshape_as_raster
import pyorc as ORC
import cv2
from matplotlib.animation import FuncAnimation, FFMpegWriter
from process import *
# import example data
from example_data import movie

folder = r"/home/hcwinsemius/Media/projects/pyorc"
video_args = {
    "fps": 4,
    "extra_args": ["-vcodec", "libx264"],
    #               'extra_args': ['-pix_fmt', 'yuv420p'],
    "dpi": 120,
}
# example data is loaded in dictionary "movie" with a placeholder for the video to investigate, below we adapt the
# settings.

# change the movie file (all frames are parsed under a subfolder called 'frames')
movie['file']['bucket'] = r'/home/hcwinsemius/Media/projects/pyorc/buckets/20210327_113200' # s3 bucket (or in our case a local folder) in which the movie is located
movie['file']['identifier'] = 'clip_schedule_20210327_113240.mkv'
movie['h_a'] = 0.62  # this value is simply what you read on the staff gauge

movie['camera_config']['resolution'] = 0.01
movie['camera_config']['aoi_window_size'] = 30

dst = os.path.join(movie['file']['bucket'], 'results')

# make destination folder if not existing
if not(os.path.isdir(dst)):
    os.makedirs(dst)


# # extract and project frames
proj_frames(movie, dst)
# # compute piv
compute_piv(movie, dst, piv_kwargs={})

# filter_piv(movie, dst, filter_temporal_kwargs={"kwargs_corr": {"tolerance": -100}})
filter_piv(movie, dst)
#
compute_q(movie, dst)

fn = os.path.join(dst, "velocity_filter.nc")
# retrieve velocities over cross section only (ds_points has time, points as dimension)
ds_points = pyorc.io.interp_coords(
    fn, *zip(*movie["bathymetry"]["coords"])
)

# add the effective velocity perpendicular to cross-section
ds_points["v_eff"] = pyorc.piv.vector_to_scalar(
    ds_points["v_x"], ds_points["v_y"]
)

v_eff_mean = ds_points["v_eff"].mean(dim="points")
means = []
for nr_samples in range(25, 150, 25):
    for n in range(1000):
        idx = np.random.randint(0, 124, 25)
        means.append(v_eff_mean[idx].mean())



make_video(movie, dst, video_args)

