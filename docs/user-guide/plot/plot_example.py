import xarray as xr
import pandas as pd
import pyorc
from matplotlib.colors import Normalize
ds = xr.open_dataset("examples/ngwerere/ngwerere_masked.nc")

# also open the original video file, only one frame needed
video_file = "examples/ngwerere/ngwerere_20191103.mp4"

video = pyorc.Video(video_file, start_frame=0, end_frame=1)

# borrow the camera config from the velocimetry results
video.camera_config = ds.velocimetry.camera_config

# get the frame as rgb
da_rgb = video.get_frames(method="rgb")

# read a cross section
cross_section = pd.read_csv("examples/ngwerere/ngwerere_cross_section.csv")
x = cross_section["x"]
y = cross_section["y"]
z = cross_section["z"]

# get data over the cross section
ds_points = ds.velocimetry.get_transect(x, y, z, crs=32735, rolling=4)

# get vertically integrated velocities
ds_points_q = ds_points.transect.get_q(fill_method="log_interp")

# plot the rgb frame first. We use the "camera" mode to plot the camera perspective.
norm = Normalize(vmin=0., vmax=0.6, clip=False)

# plot the first frame and return the mappable
p = da_rgb[0].frames.plot(mode="camera")

# extract mean velocity and plot in camera projection
ds.mean(dim="time", keep_attrs=True).velocimetry.plot(
    ax=p.axes,  # use the axes already created with the first mappable
    mode="camera",  # show camera perspective
    cmap="rainbow",  # choose a colormap
    scale=200,  # quiver scale (larger means smaller arrow lengths)
    width=0.001,  # width of quiver arrows
    alpha=0.3,  # transparency (smaller is more transparent)
    norm=norm,  # color scale
)

# plot velocimetry point results in camera projection
ds_points_q.isel(quantile=2).transect.plot(
    ax=p.axes,  # refer to the axes already created
    mode="camera",  # show camera perspective
    cmap="rainbow",  # choose a colormap
    scale=100,  # quiver scale (larger means smaller arrow lengths)
    width=0.003,  # width of quiver arrows
    norm=norm,  # color scale
    add_colorbar=True  # as final touch, add a colorbar
)
