import cv2
import dask
import dask.array as da

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from orc import cv, piv, io
from matplotlib.animation import FuncAnimation, FFMpegWriter

VIDEO_ARGS = {
    "fps": 25,
    "extra_args": ["-vcodec", "libx264"],
    "dpi": 120,
}

PIV_NAMES = ["v_x", "v_y", "s2n", "corr"]

def delayed_to_da(delayed_das, shape, dtype, time, y, x, attrs={}, name=None):
    """
    Convert a list of delayed 2D arrays (assumed to be time steps of grids) into a 3D DataArray with all axes
    :param delayed_das: Delayed dask data arrays (2D)
    :param shape: tuple, foreseen shape of data arrays (rows, cols)
    :param dtype: string or dtype, e.g. "uint8" of data arrays
    :param time:
    :param y:
    :param x:
    :param attrs:
    :return:
    """
    assert(len(time)==len(delayed_das)), f"Length of time axis {len(time)} is not equal to amount of data arrays {len(delayed_das)}"
    assert(len(y)==shape[0]), f"Length of y-axis {len(y)} is not equal to expected shape in y-direction {shape[0]}"
    assert(len(x)==shape[1]), f"Length of x-axis {len(x)} is not equal to expected shape in x-direction {shape[1]}"
    data_array = [da.from_delayed(
        d,
        dtype=dtype,
        shape=shape
    ) for d in delayed_das]
    dims = ("time", "y", "x")
    return xr.DataArray(
        da.stack(data_array, axis=0),
        dims=dims,
        coords={
            "time": time,
            "y": y,
            "x": x
        },
        attrs=attrs,
        name=name
    )

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
    M = frames.M
    shape = frames.proj_shape
    get_ortho = dask.delayed(cv.get_ortho)
    imgs = [get_ortho(frame, M, tuple(np.flipud(shape)), flags=cv2.INTER_AREA) for frame in frames]
    # prepare axes
    time = frames.time
    y = np.flipud(np.linspace(frames.resolution/2, frames.resolution*(shape[0]-0.5), shape[0]))
    x = np.linspace(frames.resolution/2, frames.resolution*(shape[1]-0.5), shape[1])
    # prepare attributes

    attrs = {
        "M": M,
        "proj_transform": frames.proj_transform,
        "crs": frames.crs,
        "resolution": frames.resolution
    }
    # create DataArray and return
    return delayed_to_da(imgs, shape, "uint8", time, y, x, attrs=frames.attrs)


def landmask(frames, dilate_iter=10, samples=15):
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
    # frames_min = frames_reduce.min(axis=-1).min(axis=-1)
    # frames_max = frames_reduce.max(axis=-1).min(axis=-1)
    # frames_norm = ((frames_reduce - frames_min)/(frames_max-frames_min)*255).astype("uint8")
    frames_thres = np.maximum(frames_reduce, 0)
    # # normalize
    frames_norm = (frames_thres*255/frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
    frames_norm = frames_norm.where(roll_mean!=0, 0)
    return frames_norm

def compute_piv(frames, res_x=0.01, res_y=0.01, search_area_size=30, correlation=True, window_size=None, overlap=None, **kwargs):
    # forward the computation to piv
    dask_piv = dask.delayed(piv.piv, nout=6)
    v_x, v_y, s2n, corr = [], [], [], []
    for frame_a, frame_b in zip(frames[0:-1], frames[1:]):
        dt = frame_b.time - frame_a.time
        # determine time difference dt between frames
        cols, rows, _v_x, _v_y, _s2n, _corr = dask_piv(
            frame_a,
            frame_b,
            res_x=frames.resolution,
            res_y=frames.resolution,
            dt=float(dt.values),
            search_area_size=frames.window_size,
            **kwargs,
        )
        # result = dask_piv(
        #     frame_a,
        #     frame_b,
        #     res_x=frames.resolution,
        #     res_y=frames.resolution,
        #     dt=float(dt.values),
        #     search_area_size=frames.window_size,
        #     **kwargs,
        # )
        v_x.append(_v_x), v_y.append(_v_y), s2n.append(_s2n), corr.append(_corr)
    # compute one sample for the spacing
    cols, rows, _v_x, _v_y, _s2n, _corr = piv.piv(
        frame_a,
        frame_b,
        res_x=frames.resolution,
        res_y=frames.resolution,
        dt=float(dt.values),
        search_area_size=frames.window_size,
        **kwargs,
    )
    attrs = frames.attrs
    time = (frames.time[0:-1].values + frames.time[1:].values)/2  # as we use frame to frame differences, one time step gets lost
    x, y = io.get_axes(cols, rows, frames.resolution)
    xs, ys, lons, lats = io.get_xs_ys(cols, rows, frames.proj_transform, frames.crs)
    v_x, v_y, s2n, corr = [
        delayed_to_da(
            data,
            (len(y), len(x)),
            np.float32,
            time,
            y,
            x,
            attrs=attrs,
            name=name
        ) for data, name in zip((v_x, v_y, s2n, corr), PIV_NAMES)]
    ds = xr.merge([v_x, v_y, s2n, corr])
    return ds
    #
    # # prepare dataset
    # dataset = io.to_dataset(
    #     [v_x, v_y, s2n, corr],
    #     var_names,
    #     x,
    #     y,
    #     time=time,
    #     lat=lats,
    #     lon=lons,
    #     xs=xs,
    #     ys=ys,
    #     attrs=var_attrs,
    # )
    #
    # # structure into properly defined DataArrays
    # v_x = [da.from_delayed(img, dtype="uint8", shape=tuple(np.flipud(shape))) for img in imgs]
    # time = frames.time
    # y = np.flipud(np.linspace(frames.resolution/2, frames.resolution*(data_array[0].shape[0]-0.5), data_array[0].shape[0]))
    # x = np.linspace(frames.resolution/2, frames.resolution*(data_array[0].shape[1]-0.5), data_array[0].shape[1])
    # dims = ("time", "y", "x")
    # attrs = {
    #     "M": M,
    #     "proj_transform": frames.proj_transform,
    #     "crs": frames.crs,
    #
    # }
    # return xr.DataArray(
    #     da.stack(data_array, axis=0),
    #     dims=dims,
    #     coords={
    #         "time": time,
    #         "y": y,
    #         "x": x
    #     },
    #     attrs=attrs
    # )



    # finally read GeoTiff transform from the first file

    raise NotImplementedError

    # def piv(
    #         frame_a, frame_b, res_x=0.01, res_y=0.01, search_area_size=30, correlation=True, window_size=None,
    #         overlap=None,
    #         **kwargs
    # ):
    #

def animation(fn, frames, video_args=VIDEO_ARGS, **kwargs):
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

