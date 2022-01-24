import cv2
import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.animation import FuncAnimation, FFMpegWriter
from pyorc import cv, io, helpers, const
from rasterio.transform import Affine



def project(frames):
    """
    Project frames DataArray, derived from a Video object with a complete CameraConfig, into a projected
    frames DataArray, with information from the CameraConfig. This requires that the CameraConfig contains full gcp
    information and a coordinate reference system (crs).

    :param frames: DataArray with frames, and typical attributes derived from the CameraConfig M, proj_transform and crs
    :return: frames: DataArray with projected frames and x and y axis in local coordinate system (origin: top-left)

    """
    # retrieve the M and shape from the frames attributes
    if not(hasattr(frames, "M")):
        raise AttributeError(f'Attribute "M" is not available in frames')
    if not(hasattr(frames, "shape")):
        raise AttributeError(f'Attribute "shape" is not available in frames')
    M = helpers.deserialize_attr(frames, "M", np.array, args_parse=False)
    shape = helpers.deserialize_attr(frames, "proj_shape", list)
    get_ortho = dask.delayed(cv.get_ortho)
    imgs = [get_ortho(frame, M, tuple(np.flipud(shape)), flags=cv2.INTER_AREA) for frame in frames]
    # prepare axes
    time = frames.time
    y = np.flipud(np.linspace(frames.resolution/2, frames.resolution*(shape[0]-0.5), shape[0]))
    x = np.linspace(frames.resolution/2, frames.resolution*(shape[1]-0.5), shape[1])
    cols, rows = np.meshgrid(
        np.arange(len(x)),
        np.arange(len(y))
    )
    xs, ys, lons, lats = io.get_xs_ys(
        cols,
        rows,
        helpers.deserialize_attr(frames, "proj_transform", Affine, args_parse=True),
        frames.crs
    )

    # prepare attributes

    attrs = {
        "M": M,
        "proj_transform": frames.proj_transform,
        "crs": frames.crs,
        "resolution": frames.resolution
    }
    # create DataArray and return
    coords = {
        "time": time,
        "y": y,
        "x": x
    }
    if "rgb" in frames.coords:
        coords["rgb"] = np.array([0, 1, 2])
        shape = (*shape, 3)
    da = helpers.delayed_to_da(
        imgs,
        shape,
        "uint8",
        coords=coords,
        attrs=frames.attrs
    )
    del coords["time"]
    if "rgb" in frames.coords:
        del coords["rgb"]
    da = helpers.add_xy_coords(da, [xs, ys, lons, lats], coords, const.GEOGRAPHICAL_ATTRS)
    return da

def landmask(frames, dilate_iter=10, samples=15):
    """
    Attempt to mask out land from water, by assuming that the time standard deviation over mean of land is much higher
    than that of water. An automatic threshold using Otsu thresholding is used to separate and a dilation operation is
    used to make the land mask a little bit larger than the exact defined pixels.

    :param frames: DataArray with frames
    :param dilate_iter: number of dilation iterations to use, to dilate land mask
    :param samples: amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
        number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
    :return: xr.DataArray with filtered frames

    """
    time_interval = round(len(frames)/samples)
    assert(time_interval != 0), f"Amount of frames is too small to provide {samples} samples"
    # ensure attributes are kept
    xr.set_options(keep_attrs=True)
    # compute standard deviation over mean, assuming this value is low over water, and high over land
    std_norm = (frames[::time_interval].std(axis=0) / frames[::time_interval].mean(axis=0)).load()
    # retrieve a simple 3x3 equal weight kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilate the std_norm by some dilation iterations
    dilate_std_norm = cv2.dilate(std_norm.values, kernel, iterations=dilate_iter)
    # rescale result to typical uint8 0-255 range
    img = cv2.normalize(dilate_std_norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
        np.uint8)
    # threshold with Otsu thresholding
    ret, thres = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask is where thres is
    mask = thres != 255
    # make mask 3-dimensional
    return (frames * mask) # .astype(bool)


def normalize(frames, samples=15):
    """
    Remove the mean of sampled frames. This is typically used to remove non-moving background from foreground, and helps
    to increase contrast when river bottoms are visible, or when the objective contains partly illuminated and partly
    shaded parts.

    :param frames: DataArray with frames
    :param samples: amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
        number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
    :return: xr.DataArray with filtered frames
    """
    time_interval = round(len(frames)/samples)
    assert(time_interval != 0), f"Amount of frames is too small to provide {samples} samples"
    # ensure attributes are kept
    xr.set_options(keep_attrs=True)
    # normalize = dask.delayed(cv2.normalize)
    mean = frames[::time_interval].mean(axis=0).load()
    frames_reduce = frames - mean
    # frames_min = frames_reduce.min(axis=-1).min(axis=-1)
    # frames_max = frames_reduce.max(axis=-1).min(axis=-1)
    # frames_norm = ((frames_reduce - frames_min)/(frames_max-frames_min)*255).astype("uint8")
    frames_thres = np.maximum(frames_reduce, 0)
    # # normalize
    frames_norm = (frames_thres*255/frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
    frames_norm = frames_norm.where(mean!=0, 0)
    return frames_norm

def reduce_rolling(frames, int=25):
    roll_mean = frames.rolling(time=25).mean()
    assert(len(frames) >= int), f"Amount of frames is smaller than requested rolling interval of {int} samples"
    # ensure attributes are kept
    xr.set_options(keep_attrs=True)
    # normalize = dask.delayed(cv2.normalize)
    frames_reduce = frames - roll_mean
    frames_thres = np.maximum(frames_reduce, 0)
    # # normalize
    frames_norm = (frames_thres*255/frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
    frames_norm = frames_norm.where(roll_mean!=0, 0)
    return frames_norm

def animation(fn, frames, video_args=const.VIDEO_ARGS, **kwargs):
    """
    Create a video of the result, using defined settings passed to imshow

    :param attr:
    :return:
    """

    def init():
        im_data = frames[0]
        im.set_data(np.zeros(im_data.shape))
        return ax

    def animate(i):
        im_data = frames[i].load()
        im.set_data(im_data.values)
        return ax

    # retrieve the dataset
    f = plt.figure(figsize=(16, 9), frameon=False)
    f.set_size_inches(16, 9, True)
    f.patch.set_facecolor("k")
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = plt.subplot(111)
    im_data = frames[0].load()
    im = ax.imshow(im_data.values, **kwargs)
    anim = FuncAnimation(
        f, animate, init_func=init, frames=frames.shape[0], interval=20, blit=False
    )
    anim.save(fn, **video_args)

