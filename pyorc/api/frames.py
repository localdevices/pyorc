import copy

import cv2
import dask
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr

from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
from .orcbase import ORCBase
from .plot import _frames_plot
from .. import cv, helpers, const, piv_process


@xr.register_dataarray_accessor("frames")
class Frames(ORCBase):
    """Frames functionalities that can be applied on xr.DataArray"""
    def __init__(self, xarray_obj):
        """
        Initialize a frames xr.DataArray

        Parameters
        ----------
        xarray_obj: xr.DataArray
            frames data fields (from pyorc.Video.get_frames)
        """
        super(Frames, self).__init__(xarray_obj)

    def get_piv(self, **kwargs):
        """Perform PIV computation on projected frames. Only a pipeline graph to computation is setup. Call a result to
        trigger actual computation. The dataset returned contains velocity information for each interrogation window
        including "v_x" for x-direction velocity components, "v_y" for y-direction velocity component, "corr" for
        the maximum correlation [-] found in the interrogation window and s2n [-] for the signal to noise ratio

        Parameters
        ----------
        **kwargs : keyword arguments to pass to dask_piv, used to control the manner in which openpiv.pyprocess
            is called.

        Returns
        -------
        ds : xr.Dataset
            PIV results in a lazy dask.array form in DataArrays "v_x", "v_y", "corr" and "s2n".

        """
        camera_config = copy.deepcopy(self.camera_config)
        dt = self._obj["time"].diff(dim="time")

        # add a number of kwargs for the piv function
        if "window_size" in kwargs:
            camera_config.window_size = kwargs["window_size"]
        kwargs["search_area_size"] = camera_config.window_size
        kwargs["window_size"] = camera_config.window_size
        kwargs["res_x"] = camera_config.resolution
        kwargs["res_y"] = camera_config.resolution
        # set an overlap if not provided in kwargs
        if not("overlap" in kwargs):
            kwargs["overlap"] = int(round(camera_config.window_size) / 2)
        # first get rid of coordinates that need to be recalculated
        coords_drop = list(set(self._obj.coords) - set(self._obj.dims))
        obj = self._obj.drop_vars(coords_drop)
        # get frames and shifted frames in time
        frames1 = obj.shift(time=1)[1:]
        frames2 = obj[1:]
        # get the cols and rows coordinates of the expected results
        cols, rows = piv_process.get_piv_size(
            image_size=frames1[0].shape,
            search_area_size=kwargs["search_area_size"],
            overlap=kwargs["overlap"]
        )
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

        # new approach to getting M (from bbox coordinates)
        src = camera_config.get_bbox(camera=True, h_a=self.h_a).exterior.coords[0:4]
        dst_xy = camera_config.get_bbox().exterior.coords[0:4]
        # get geographic coordinates bbox corners
        dst = cv.transform_to_bbox(dst_xy, camera_config.bbox, camera_config.resolution)
        M = cv.get_M_2D(src, dst, reverse=True)
        # TODO: remove when above 4 lines work
        # M = camera_config.get_M(self.h_a, to_bbox_grid=True, reverse=True)
        # compute row and column position of vectors in original reprojected background image col/row coordinates
        xp, yp = helpers.xy_to_perspective(
            *np.meshgrid(x, np.flipud(y)),
            self.camera_config.resolution,
            M
        )
        # ensure y coordinates start at the top in the right orientation (different from order of a CRS)
        shape_y, shape_x = self.camera_shape
        yp = shape_y - yp
        coords = {
            "y": y,
            "x": x
        }
        # retrieve all data arrays
        v_x, v_y, s2n, corr = xr.apply_ufunc(
            piv_process.piv,
            frames1,
            frames2,
            dt,
            kwargs=kwargs,
            input_core_dims=[["y", "x"], ["y", "x"], []],
            output_core_dims=[["new_y", "new_x"]] * 4,
            dask_gufunc_kwargs={
                "output_sizes": {
                    "new_y": len(y),
                    "new_x": len(x)
                },
            },
            output_dtypes=[np.float32] * 4,
            vectorize=True,
            keep_attrs=True,
            dask="parallelized",
        )
        # merge all DataArrays in one Dataset
        ds = xr.merge([
            v_x.rename("v_x"),
            v_y.rename("v_y"),
            s2n.rename("s2n"),
            corr.rename("corr")
        ]).rename({
            "new_x": "x",
            "new_y": "y"
        })
        # add y and x-axis values
        ds["y"] = y
        ds["x"] = x

        # add all 2D-coordinates
        ds = ds.velocimetry._add_xy_coords(
            [xp, yp, xs, ys, lons, lats],
            coords,
            {**const.PERSPECTIVE_ATTRS, **const.GEOGRAPHICAL_ATTRS}
        )
        # in case window_size was changed, overrule the camera_config attribute
        ds.attrs.update(camera_config=camera_config.to_json())
        # set encoding
        ds.velocimetry.set_encoding()
        return ds

    def project(self, resolution=None):
        """Project frames into a projected frames object, with information from the camera_config attr.
        This requires that the CameraConfig contains full gcp information. If a CRS is provided, also "lat" and "lon"
        variables will be added to the output, containing geographical latitude and longitude coordinates.

        Parameters
        ----------
        resolution : float, optional
            resolution to project to. If not provided, this will be taken from the camera config in the metadata
             (Default value = None)

        Returns
        -------
        frames: xr.DataArray
             projected frames and x and y in local coordinate system (origin: top-left), lat and lon if a crs was
             provided.
        """
        camera_config = copy.deepcopy(self.camera_config)
        if resolution is not None:
            camera_config.resolution = resolution
        # convert bounding box coords into row/column space
        shape = camera_config.shape
        # get camera perspective bbox corners
        src = camera_config.get_bbox(camera=True, h_a=self.h_a).exterior.coords[0:4]
        dst_xy = camera_config.get_bbox().exterior.coords[0:4]
        # get geographic coordinates bbox corners
        dst = cv.transform_to_bbox(dst_xy, camera_config.bbox, camera_config.resolution)
        M = cv.get_M_2D(src, dst)

        # prepare all coordinates
        y = np.flipud(np.linspace(
            camera_config.resolution / 2,
            camera_config.resolution * (shape[0] - 0.5),
            shape[0])
        )
        x = np.linspace(
            camera_config.resolution / 2,
            camera_config.resolution * (shape[1] - 0.5), shape[1]
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
        coords = {
            "y": y,
            "x": x,
        }
        f = xr.apply_ufunc(
            cv.get_ortho, self._obj,
            kwargs={
                "M": M,
                "shape": tuple(np.flipud(shape)),
                "flags": cv2.INTER_AREA
            },
            input_core_dims=[["y", "x"]],
            output_core_dims=[["new_y", "new_x"]],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "new_y": len(y),
                    "new_x": len(x)
                },
            },
            output_dtypes=[self._obj.dtype],
            vectorize=True,
            exclude_dims=set(("y", "x")),
            dask="parallelized",
            keep_attrs=True
        ).rename({
            "new_y": "y",
            "new_x": "x"
        })
        f["y"] = y
        f["x"] = x
        # assign coordinates
        f = f.frames._add_xy_coords([xs, ys, lons, lats], coords, const.GEOGRAPHICAL_ATTRS)
        if "rgb" in f.dims and len(f.dims) == 4:
            # ensure that "rgb" is the last dimension
            f = f.transpose("time", "y", "x", "rgb")
        # in case resolution was changed, overrule the camera_config attribute
        f.attrs.update(camera_config=camera_config.to_json())

        return f

    def landmask(self, dilate_iter=10, samples=15):
        """Attempt to mask out land from water, by assuming that the time standard deviation over mean of land is much
        higher than that of water. An automatic threshold using Otsu thresholding is used to separate and a dilation
        operation is used to make the land mask slightly larger than the exact defined pixels.

        Parameters
        ----------
        dilate_iter : int, optional
            number of dilation iterations to use, to dilate land mask (Default value = 10)
        samples : int, optional
            amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation (Default value = 15)

        Returns
        -------
        da : xr.DataArray
            filtered frames

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
        """Remove the mean of sampled frames. This is typically used to remove non-moving background from foreground, and
        helps to increase contrast when river bottoms are visible, or when the objective contains partly illuminated and
        partly shaded parts.

        Parameters
        ----------
        samples : int, optional
            amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).

        Returns
        -------
        da : xr.DataArray
            filtered frames

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

    def edge_detect(self, wdw_1=1, wdw_2=2):
        """Convert frames in edges, using a band convolution filter. The filter uses two slightly differently convolved
        images and computes their difference to detect edges.

        Parameters
        ----------
        wdw_1 : int, optional
            window size to use for first gaussian blur filter (Default value = 2)
        wdw_2 : int, optional
            stride to use for second gaussian blur filter (Default value = 4)

        Returns
        -------
        da : xr.DataArray
            filtered frames (i.e. difference between first and second gaussian convolution)

        """

        # shape = self._obj[0].shape  # single-frame shape does not change
        # da_convert_edge = dask.delayed(cv._convert_edge)
        stride_1 = wdw_1 * 2 + 1
        stride_2 = wdw_2 * 2 + 1
        return xr.apply_ufunc(
            cv._convert_edge,
            self._obj, stride_1, stride_2,
            input_core_dims=[["y", "x"], [], []],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            #     exclude_dims=set(("y",)),
            #         kwargs={"stride_1": 5, "stride_2": 9},
            dask="parallelized",
            keep_attrs=True
        )

    def reduce_rolling(self, samples=25):
        """Remove a rolling mean from the frames (very slow, so in most cases, it is recommended to use
        Frames.normalize).

        Parameters
        ----------
        samples : int, optional
            number of samples per rolling (Default value = 25)

        Returns
        -------
        da : xr.DataArray
            filtered frames

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

    def to_ani(
            self,
            fn,
            figure_kwargs=const.FIGURE_ARGS,
            video_kwargs=const.VIDEO_ARGS,
            anim_kwargs=const.ANIM_ARGS,
            progress_bar=True,
            **kwargs
    ):
        """Store an animation of the frames in the object

        Parameters
        ----------
        fn : str
            path to file to which animation is stored
        figure_kwargs : dict, optional
            keyword args passed to ``matplotlib.pyplot.figure`` (Default value = const.FIGURE_ARGS)
        video_kwargs : dict, optional
            keyword arguments passed to ``save`` method of animation, containing parameters such as the frame rate,
            dpi and codec settings to use. (Default value = const.VIDEO_ARGS)
        anim_kwargs : dict, optional
            keyword arguments passed to ``matplotlib.animation.FuncAnimation``
            to control the animation. (Default value = const.ANIM_ARGS)
        **kwargs : keyword arguments to pass to ``matplotlib.pyplot.imshow``

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
        if progress_bar:
            frames = tqdm(range(len(self._obj)), position=0, leave=True)
        else:
            frames = range(len(self._obj))
        anim = FuncAnimation(
            f, animate, init_func=init, frames=frames, **anim_kwargs
        )
        anim.save(fn, **video_kwargs)

    def to_video(
            self,
            fn,
            video_format=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=None
    ):
        """
        Write frames to a video file without any layout

        Parameters
        ----------
        fn : str
            Path to output file
        video_format : cv2.VideoWriter_fourcc, optional
            A VideoWriter preference, default is cv2.VideoWriter_fourcc(*"mp4v")
        fps : float, optional
            Frames per second, if not provided, derived from original video

        Returns
        -------
        None
        """
        if fps is None:
            # estimate it from the time differences
            fps = 1/(self._obj["time"][1].values - self._obj["time"][0].values)
        h = self._obj.shape[1]
        w = self._obj.shape[2]
        out = cv2.VideoWriter(fn, video_format, fps, (w, h))
        pbar = tqdm(self._obj, position=0, leave=True)
        pbar.set_description("Writing frames")
        for n, f in enumerate(pbar):
            if len(f.shape) == 3:
                img = cv2.cvtColor(f.values, cv2.COLOR_RGB2BGR)
            else:
                img = f.values
                if n == 0:
                    # make a scale between 0 and 255, only with first frame
                    img_min = img.min(axis=0).min(axis=0)
                    img_max = img.max(axis=0).max(axis=0)
                img = np.uint8(255*((img - img_min)/(img_max - img_min)))
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out.write(img)
        out.release()

    plot = _frames_plot
