import copy

import cv2
import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
from .orcbase import ORCBase
from .plot import _frames_plot
from .. import cv, helpers, const, piv_process


@xr.register_dataarray_accessor("frames")
class Frames(ORCBase):
    """
    Frames functionalities that can be applied on ``xarray.DataArray``

    """
    def __init__(self, xarray_obj):
        """
        Initialize a frames ``xarray.DataArray``

        :param xarray_obj: xarray dataarray containing frames data fields (from ``pyorc.Video.get_frames``)
        """
        super(Frames, self).__init__(xarray_obj)

    def get_piv(self, **kwargs):
        """
        Perform PIV computation on projected frames. Only a pipeline graph to computation is setup. Call a result to
        trigger actual computation.

        :param kwargs: dict, keyword arguments to pass to dask_piv, used to control the manner in which
            openpiv.pyprocess is called.
        :return: Velocimetry object (xr.Dataset), containing the PIV results in a lazy dask.array form.
        """
        camera_config = copy.deepcopy(self.camera_config)
        if "window_size" in kwargs:
            camera_config.window_size = kwargs["window_size"]
            del kwargs["window_size"]
        # forward the computation to piv
        dask_piv = dask.delayed(piv_process.piv, nout=6)
        v_x, v_y, s2n, corr = [], [], [], []
        frames_a = self._obj[0:-1]
        frames_b = self._obj[1:]
        for frame_a, frame_b in zip(frames_a, frames_b):
            # select the time difference in seconds
            dt = frame_b.time - frame_a.time
            # perform lazy piv graph computation
            cols, rows, _v_x, _v_y, _s2n, _corr = dask_piv(
                frame_a,
                frame_b,
                res_x=camera_config.resolution,
                res_y=camera_config.resolution,
                dt=float(dt.values),
                search_area_size=camera_config.window_size,
                **kwargs,
            )
            # append to result
            v_x.append(_v_x), v_y.append(_v_y), s2n.append(_s2n), corr.append(_corr)
        # compute one sample for the spacing
        cols, rows, _v_x, _v_y, _s2n, _corr = piv_process.piv(
            frame_a,
            frame_b,
            res_x=camera_config.resolution,
            res_y=camera_config.resolution,
            dt=float(dt.values),
            search_area_size=camera_config.window_size,
            **kwargs,
        )
        # extract global attributes from origin
        global_attrs = self._obj.attrs
        time = (self._obj.time[0:-1].values + self._obj.time[1:].values) / 2  # frame to frame differences
        # retrieve the x and y-axis belonging to the results
        x, y = helpers.get_axes(cols, rows, self.camera_config.resolution)
        # convert in projected and latlon coordinates
        xs, ys = helpers.get_xs_ys(
            cols,
            rows,
            camera_config.transform,
        )
        if hasattr(camera_config, "crs"):
            lons, lats = helpers.get_lons_lats(xs, ys, camera_config.crs)
        else:
            lons = None
            lats = None
        M = camera_config.get_M(self.h_a, reverse=True)
        # compute row and column position of vectors in original reprojected background image col/row coordinates
        xp, yp = helpers.xy_to_perspective(*np.meshgrid(x, np.flipud(y)), self.camera_config.resolution, M)
        # ensure y coordinates start at the top in the right orientation (different from order of a CRS)
        shape_y, shape_x = self.camera_shape
        yp = shape_y - yp
        coords = {
            "time": time,
            "y": y,
            "x": x
        }
        # establish the full xr.Dataset
        v_x, v_y, s2n, corr = [
            helpers.delayed_to_da(
                data,
                (len(y), len(x)),
                np.float32,
                coords=coords,
                attrs=attrs,
                name=name
            ) for data, (name, attrs) in zip((v_x, v_y, s2n, corr), const.PIV_ATTRS.items())]
        ds = xr.merge([v_x, v_y, s2n, corr])
        del coords["time"]
        # prepare the xs, ys, lons and lats grids for geographical projections and add to xr.Dataset
        ds = ds.velocimetry._add_xy_coords(
            [xp, yp, xs, ys, lons, lats],
            coords,
            {**const.PERSPECTIVE_ATTRS, **const.GEOGRAPHICAL_ATTRS}
        )
        # add piv object functionality and attrs to dataset and return
        ds = xr.Dataset(ds, attrs=global_attrs)
        # in case window_size was changed, overrule the camera_config attribute
        ds.attrs.update(camera_config=camera_config.to_json())
        # set encoding
        ds.velocimetry.set_encoding()
        return ds


    def project(self, resolution=None):
        """
        Project frames into a projected frames object, with information from the camera_config attr.
        This requires that the CameraConfig contains full gcp information and a coordinate reference system (crs).

        :return: frames: xr.DataArray with projected frames and x and y in local coordinate system (origin: top-left)

        """
        # retrieve the M and shape from camera config with said h_a
        camera_config = copy.deepcopy(self.camera_config)
        if resolution is not None:
            camera_config.resolution = resolution
        M = camera_config.get_M(self.h_a)
        shape = camera_config.shape
        # get orthoprojected frames as delayed objects
        get_ortho = dask.delayed(cv.get_ortho)
        imgs = [get_ortho(frame, M, tuple(np.flipud(shape)), flags=cv2.INTER_AREA) for frame in self._obj]
        # prepare axes
        time = self._obj.time
        y = np.flipud(np.linspace(
            camera_config.resolution/2,
            camera_config.resolution*(shape[0]-0.5),
            shape[0])
        )
        x = np.linspace(
            camera_config.resolution/2,
            camera_config.resolution*(shape[1]-0.5), shape[1]
        )
        cols, rows = np.meshgrid(
            np.arange(len(x)),
            np.arange(len(y))
        )
        # retrieve all coordinates we may ever need for further analysis or plotting
        xs, ys = helpers.get_xs_ys(
            cols,
            rows,
            camera_config.transform,
        )
        if hasattr(camera_config, "crs"):
            lons, lats = helpers.get_lons_lats(xs, ys, camera_config.crs)
        else:
            lons = None
            lats = None
        # Setup coordinates
        coords = {
            "time": time,
            "y": y,
            "x": x
        }
        # add a coordinate if RGB frames are used
        if "rgb" in self._obj.coords:
            coords["rgb"] = np.array([0, 1, 2])
            shape = (*shape, 3)
        # prepare a dask data array
        frames_proj = helpers.delayed_to_da(
            imgs,
            shape,
            "uint8",
            coords=coords,
            attrs=self._obj.attrs,
            object_type=xr.DataArray
        )
        # remove time coordinate for the spatial variables (and rgb in case rgb frames are used)
        del coords["time"]
        if "rgb" in self._obj.coords:
            del coords["rgb"]
        # add coordinate meshes to projected frames and return
        frames_proj = frames_proj.frames._add_xy_coords([xs, ys, lons, lats], coords, const.GEOGRAPHICAL_ATTRS)
        # in case resolution was changed, overrule the camera_config attribute
        frames_proj.attrs.update(camera_config = camera_config.to_json())
        return frames_proj


    def landmask(self, dilate_iter=10, samples=15):
        """
        Attempt to mask out land from water, by assuming that the time standard deviation over mean of land is much
        higher than that of water. An automatic threshold using Otsu thresholding is used to separate and a dilation
        operation is used to make the land mask slightly larger than the exact defined pixels.

        :param dilate_iter: int, number of dilation iterations to use, to dilate land mask
        :param samples: int, amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
        :return: xr.DataArray, filtered frames

        """
        time_interval = round(len(self._obj)/samples)
        assert(time_interval != 0), f"Amount of frames is too small to provide {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        # compute standard deviation over mean, assuming this value is low over water, and high over land
        std_norm = (self._obj[::time_interval].std(axis=0) / self._obj[::time_interval].mean(axis=0)).load()
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
        return (self._obj * mask)


    def normalize(self, samples=15):
        """
        Remove the mean of sampled frames. This is typically used to remove non-moving background from foreground, and
        helps to increase contrast when river bottoms are visible, or when the objective contains partly illuminated and
        partly shaded parts.

        :param samples: int, amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
        :return: xr.DataArray, filtered frames
        """
        time_interval = round(len(self._obj) / samples)
        assert (time_interval != 0), f"Amount of frames is too small to provide {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        mean = self._obj[::time_interval].mean(axis=0).load().astype("float32")
        frames_reduce = self._obj.astype("float32") - mean
        frames_min = frames_reduce.min(axis=-1).min(axis=-1)
        frames_max = frames_reduce.max(axis=-1).max(axis=-1)
        frames_norm = ((frames_reduce - frames_min) / (frames_max - frames_min) * 255).astype("uint8")
        return frames_norm


    def edge_detect(self, stride_1=7, stride_2=9):
        """
        Convert frames in edges, using a band convolution filter. The filter uses two slightly differently convolved
        images and computes their difference to detect edges.

        :param stride_1: int, stride to use for first gaussian blur filter
        :param stride_2: int, stride to use for second gaussian blur filter
        :return: xr.DataArray, filtered frames (i.e. difference between first and second gaussian convolution)
        """
        def convert_edge(img, stride_1, stride_2):
            if not(isinstance(img, np.ndarray)):
                img = img.values
            # load values here
            blur1 = cv2.GaussianBlur(img.astype("float32"), (stride_1, stride_1), 0)
            blur2 = cv2.GaussianBlur(img.astype("float32"), (stride_2, stride_2), 0)
            edges = blur2 - blur1
            return edges

        shape = self._obj[0].shape  # single-frame shape does not change
        da_convert_edge = dask.delayed(convert_edge)
        imgs = [da_convert_edge(frame.values, stride_1, stride_2) for frame in self._obj]
        # prepare axes
        # Setup coordinates
        coords = {
            "time": self._obj.time,
            "y": self._obj.y,
            "x": self._obj.x
        }
        # add a coordinate if RGB frames are used
        frames_edge = helpers.delayed_to_da(
            imgs,
            shape,
            "float32",
            coords=coords,
            attrs=self._obj.attrs,
            name="edges",
            object_type=xr.DataArray
        )
        if "xp" in self._obj:
            frames_edge["xp"] = self._obj["xp"]
            frames_edge["yp"] = self._obj["yp"]
        return frames_edge


    def reduce_rolling(self, samples=25):
        """
        Remove a rolling mean from the frames (very slow, so in most cases, it is recommended to use `normalize`).

        :param samples: number of samples per rolling
        :return: xr.DataArray, filtered frames
        """
        roll_mean = self._obj.rolling(time=samples).mean()
        assert (len(self._obj) >= samples), f"Amount of frames is smaller than requested rolling of {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        # normalize = dask.delayed(cv2.normalize)
        frames_reduce = self._obj - roll_mean
        frames_thres = np.maximum(frames_reduce, 0)
        # # normalize
        frames_norm = (frames_thres * 255 / frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
        frames_norm = frames_norm.where(roll_mean != 0, 0)
        return frames_norm


    def to_ani(self, fn, figure_kwargs=const.FIGURE_ARGS, video_kwargs=const.VIDEO_ARGS, anim_kwargs=const.ANIM_ARGS, **kwargs):
        """
        Store an animation of the frames in the object

        :param fn: str, filename to which animation is stored
        :param figure_kwargs: dict, keyword args passed to ``matplotlib.pyplot.figure``
        :param video_kwargs:  dict, optional, keyword arguments passed to ``save`` method of animation,
            containing parameters such as the frame rate, dpi and codec settings to use.
        :param anim_kwargs: dict, optional, keyword arguments passed to ``matplotlib.animation.FuncAnimation``
            to control the animation.
        :param kwargs: dict, optional, keyword arguments to pass to ``matplotlib.pyplot.imshow``
        :return:
        """
        def init():
            # set imshow data to values in the first frame
            im.set_data(self._obj[0])
            return ax  # line,

        def animate(i):
            # set imshow data to values in the current frame
            im.set_data(self._obj[i])
            return ax

        # setup a standard 16/9 borderless window with black background
        f = plt.figure(**figure_kwargs)
        f.set_size_inches(16, 9, True)
        f.patch.set_facecolor("k")
        f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax = plt.subplot(111)

        im = ax.imshow(self._obj[0], **kwargs)
        anim = FuncAnimation(
            f, animate, init_func=init, frames=tqdm(range(len(self._obj))), **anim_kwargs
        )
        anim.save(fn, **video_kwargs)

    plot = _frames_plot