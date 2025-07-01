"""Frames methods for pyorc."""

import copy
from typing import Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ffpiv import window
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from pyorc import const, cv, helpers, project
from pyorc.velocimetry import ffpiv, openpiv

from .orcbase import ORCBase
from .plot import _frames_plot

__all__ = ["Frames"]


@xr.register_dataarray_accessor("frames")
class Frames(ORCBase):
    """Frames functionalities that can be applied on xr.DataArray."""

    def __init__(self, xarray_obj):
        """Initialize a frames `xr.DataArray`.

        Parameters
        ----------
        xarray_obj: xr.DataArray
            frames data fields (from pyorc.Video.get_frames)

        """
        super(Frames, self).__init__(xarray_obj)

    def get_piv_coords(
        self, window_size: tuple[int, int], search_area_size: tuple[int, int], overlap: tuple[int, int]
    ) -> tuple[dict, dict]:
        """Get Particle Image Velocimetry (PIV) coordinates and mesh grid projections.

        This function calculates the PIV coordinates and the corresponding mesh grid
        projections based on the provided window size, search area size, and overlap.
        The results include both projected coordinates (xp, yp) and the respective
        longitude and latitude values (if available).

        Parameters
        ----------
        window_size : (int, int)
            The size of the window for the PIV analysis.
        search_area_size : (int, int)
            The size of the search area for the PIV analysis.
        overlap : (int, int)
            The overlap ratio between consecutive windows.

        Returns
        -------
        coords : dict
            A dictionary containing the PIV local non-geographical projection coordinates:
                - 'y': The y-axis coordinates.
                - 'x': The x-axis coordinates.
        mesh_coords : dict
            A dictionary containing the mesh grid projections and coordinates:
                - 'xp': The projected x-coordinates.
                - 'yp': The projected y-coordinates.
                - 'xs': The x-coordinates in the image space.
                - 'ys': The y-coordinates in the image space.
                - 'lon': The longitude values (if CRS is provided).
                - 'lat': The latitude values (if CRS is provided).

        """
        dim_size = self._obj[0].shape

        # get the cols and rows coordinates of the expected results
        cols_vector, rows_vector = window.get_rect_coordinates(
            dim_size=dim_size,
            window_size=window_size,
            search_area_size=search_area_size,
            overlap=overlap,
        )
        cols, rows = np.meshgrid(cols_vector, rows_vector)
        # retrieve the x and y-axis belonging to the results
        x, y = helpers.get_axes(cols_vector, rows_vector, self._obj.x.values, self._obj.y.values)
        # convert in projected and latlon coordinates
        xs, ys = helpers.get_xs_ys(
            cols,
            rows,
            self.camera_config.transform,
        )
        if hasattr(self.camera_config, "crs"):
            lons, lats = helpers.get_lons_lats(xs, ys, self.camera_config.crs)
        else:
            lons = None
            lats = None
        # calculate projected coordinates
        z = self.camera_config.h_to_z(self.h_a)
        zs = np.ones(xs.shape) * z
        xp, yp = self.camera_config.project_grid(xs, ys, zs, swap_y_coords=True)
        # package the coordinates
        coords = {"y": y, "x": x}
        mesh_coords = {"xp": xp, "yp": yp, "xs": xs, "ys": ys, "lon": lons, "lat": lats}
        return coords, mesh_coords

    def get_piv(
        self,
        window_size: Optional[tuple[int, int]] = None,
        overlap: Optional[tuple[int, int]] = None,
        engine: str = "numba",
        **kwargs,
    ) -> xr.Dataset:
        """Perform PIV computation on projected frames.

        Only a pipeline graph to computation is set up. Call a result to trigger actual computation. The dataset
        returned contains velocity information for each interrogation window including "v_x" for x-direction velocity
        components, "v_y" for y-direction velocity component, "corr" for the maximum correlation [-] found in the
        interrogation window and s2n [-] for the signal to noise ratio.

        Parameters
        ----------
        window_size : (int, int), optional
            size of interrogation windows in pixels (y, x)
        overlap : (int, int), optional
            amount of overlap between interrogation windows in pixels (y, x)
        engine : str, optional
            select the compute engine, can be "openpiv" (default), "numba", or "numpy". "numba" will give the fastest
            performance but is still experimental. It can boost performance by almost an order of magnitude compared
            to openpiv or numpy. both "numba" and "numpy" use the FF-PIV library as back-end.
        **kwargs : dict
            keyword arguments to pass to the piv engine. For "numba" and "numpy" the argument `chunks` can be provided
            with an integer defining in how many batches of work the total velocimetry problem should be subdivided.

        Returns
        -------
        xr.Dataset
            PIV results in a lazy dask.array form in DataArrays "v_x", "v_y", "corr" and "s2n".

        See Also
        --------
        OpenPIV project: https://github.com/OpenPIV/openpiv-python
        FF-PIV project: https://github.com/localdevices/ffpiv

        """
        camera_config = copy.deepcopy(self.camera_config)
        dt = self._obj["time"].diff(dim="time")
        # Use window_size from camera_config unless provided in the method
        if window_size is not None:
            camera_config.window_size = window_size
        window_size = (
            2 * (camera_config.window_size,)
            if isinstance(camera_config.window_size, int)
            else camera_config.window_size
        )
        # ensure window size is a round number
        window_size = window.round_to_even(window_size)
        search_area_size = window_size
        # set an overlap if not provided in kwargs
        if overlap is None:
            overlap = 2 * (int(round(camera_config.window_size) / 2),)

        # get all required coordinates for the PIV result
        coords, mesh_coords = self.get_piv_coords(window_size, search_area_size, overlap)
        # provide kwargs for OpenPIV analysis
        if engine == "openpiv":
            import warnings

            warnings.warn(
                '"openpiv" is deprecated, please use "numba" or "numpy" as engine',
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs = {
                **kwargs,
                "search_area_size": search_area_size[0],
                "window_size": window_size[0],
                "overlap": overlap[0],
                "res_x": camera_config.resolution,
                "res_y": camera_config.resolution,
            }
            ds = openpiv.get_openpiv(self._obj, coords["y"], coords["x"], dt, **kwargs)
        elif engine in ["numba", "numpy"]:
            kwargs = {
                **kwargs,
                "search_area_size": search_area_size,
                "window_size": window_size,
                "overlap": overlap,
                "res_x": camera_config.resolution,
                "res_y": camera_config.resolution,
            }
            ds = ffpiv.get_ffpiv(self._obj, coords["y"], coords["x"], dt, engine=engine, **kwargs)
        else:
            raise ValueError(f"Selected PIV engine {engine} does not exist.")
        # add all 2D-coordinates
        ds = ds.velocimetry.add_xy_coords(mesh_coords, coords, {**const.PERSPECTIVE_ATTRS, **const.GEOGRAPHICAL_ATTRS})
        # ensure all metadata is transferred
        ds.attrs = self._obj.attrs
        # in case window_size was changed, overrule the camera_config attribute
        ds.attrs.update(camera_config=camera_config.to_json())
        # set encoding
        ds.velocimetry.set_encoding()
        return ds

    def project(
        self,
        method: Literal["cv", "numpy"] = "numpy",
        resolution: Optional[float] = None,
        reducer: Optional[str] = "mean",
    ):
        """Project frames into a projected frames object, with information from the camera_config attr.

        This requires that the CameraConfig contains full gcp information. If a CRS is provided, also "lat" and "lon"
        variables will be added to the output, containing geographical latitude and longitude coordinates.

        Parameters
        ----------
        method : str, optional
            can be `numpy` or `cv` (default `numpy`).
            With `numpy` each individual orthoprojected grid cell is mapped to the image pixels space. For oversampled
            areas, this is also done vice versa. Undersampled areas result in nearest-neighbour interpolations, whilst
            for oversampled, this results in a mean value (if the user uses `mean` as reducer). With `cv` (opencv)
            resampling is performed by first undistorting images, and then by resampling to the
            desired grid. With heavily distorted images and part of the area of interest outside of the field
            of view, the orthoprojection of the corners may end up in the wrong space.
            We recommend using `numpy` and only use cv with simple cases such as nadir-looking videos.
        resolution : float, optional
            resolution to project to. If not provided, this will be taken from the camera config in the metadata
             (Default value = None)
        reducer: str, optional
            Currently only used when `method="numpy"`. Default "mean", resulting in averaging of oversampled grid cells.

        Returns
        -------
        frames: xr.DataArray
             projected frames and x and y in local coordinate system (origin: top-left), lat and lon if a crs was
             provided.

        """
        cc = copy.deepcopy(self.camera_config)
        if resolution is not None:
            cc.resolution = resolution
        # convert bounding box coords into row/column space
        shape = cc.shape
        # prepare y and x axis of targfet
        y = np.flipud(np.linspace(cc.resolution / 2, cc.resolution * (shape[0] - 0.5), shape[0]))
        x = np.linspace(cc.resolution / 2, cc.resolution * (shape[1] - 0.5), shape[1])
        cols, rows = np.meshgrid(np.arange(len(x)), np.arange(len(y)))
        xs, ys = helpers.get_xs_ys(
            cols,
            rows,
            cc.transform,
        )
        if hasattr(cc, "crs"):
            lons, lats = helpers.get_lons_lats(xs, ys, cc.crs)
        else:
            lons = None
            lats = None
        coords = {
            "y": y,
            "x": x,
        }
        ## PROJECTION PREPARATIONS
        # ========================
        z = cc.get_z_a(self.h_a)
        if not (hasattr(project, f"project_{method}")):
            raise ValueError(f"Selected projection method {method} does not exist.")
        proj_method = getattr(project, f"project_{method}")
        da_proj = proj_method(self._obj, cc, x, y, z, reducer)
        # ensure no missing values are persisting
        da_proj = da_proj.fillna(0.0)

        # assign coordinates
        da_proj = da_proj.frames.add_xy_coords(
            {"xs": xs, "ys": ys, "lon": lons, "lat": lats}, coords, const.GEOGRAPHICAL_ATTRS
        )
        if "rgb" in da_proj.dims and len(da_proj.dims) == 4:
            # ensure that "rgb" is the last dimension and dtype is int
            da_proj = da_proj.transpose("time", "y", "x", "rgb")
            da_proj = da_proj.astype("uint8")
        # in case resolution was changed, overrule the camera_config attribute
        da_proj.attrs.update(camera_config=cc.to_json())
        return da_proj

    def normalize(self, samples=15):
        """Remove the temporal mean of sampled frames.

        This is typically used to remove non-moving background from foreground, and helps to increase contrast when
        river bottoms are visible, or when the objective contains partly illuminated and partly shaded parts.

        Parameters
        ----------
        samples : int, optional
            amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).

        Returns
        -------
        xr.DataArray
            normalized frames

        """
        time_interval = round(len(self._obj) / samples)
        assert time_interval != 0, f"Amount of frames is too small to provide {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        mean = self._obj[::time_interval].mean(axis=0).load().astype("float32")
        frames_reduce = self._obj.astype("float32") - mean
        frames_min = frames_reduce.min(axis=-1).min(axis=-1)
        frames_max = frames_reduce.max(axis=-1).max(axis=-1)
        frames_norm = ((frames_reduce - frames_min) / (frames_max - frames_min) * 255).astype("uint8")
        return frames_norm

    def edge_detect(self, wdw_1=1, wdw_2=2):
        """Highlight edges of frame intensities, using a band convolution filter.

        The filter uses two slightly differently convolved images and computes their difference to detect edges.

        Parameters
        ----------
        wdw_1 : int, optional
            window size to use for first gaussian blur filter (Default value = 2)
        wdw_2 : int, optional
            stride to use for second gaussian blur filter (Default value = 4)

        Returns
        -------
        xr.DataArray
            filtered frames (i.e. difference between first and second gaussian convolution)

        """
        # shape = self._obj[0].shape  # single-frame shape does not change
        # da_convert_edge = dask.delayed(cv._convert_edge)
        stride_1 = wdw_1 * 2 + 1
        stride_2 = wdw_2 * 2 + 1
        return xr.apply_ufunc(
            cv._convert_edge,
            self._obj,
            stride_1,
            stride_2,
            input_core_dims=[["y", "x"], [], []],
            output_core_dims=[["y", "x"]],
            vectorize=True,
            #     exclude_dims=set(("y",)),
            #         kwargs={"stride_1": 5, "stride_2": 9},
            dask="parallelized",
            keep_attrs=True,
        )

    def minmax(self, min=-np.Inf, max=np.Inf):
        """Minimum / maximum intensity filter.

        All pixels will be thresholded to a minimum and maximum value.

        Parameters
        ----------
        min : float, optional
            minimum value to bound intensities to. If not provided, no minimum bound is used.
        max : float, optional
            maximum value to bound intensities to. If not provided, no maximum bound is used.

        Returns
        -------
        xr.DataArray
            Treated frames

        """
        return np.maximum(np.minimum(self._obj, max), min)

    def range(self):
        """Return the range of pixel values through time.

        Returned array does not have a time dimension. This filter is typically used to detect
        widely changing pixels, e.g. to distinguish moving water from land.

        Returns
        -------
        xr.DataArray
            Single image (with coordinates) with minimum-maximum range in time [x, y]

        """
        range_da = (self._obj.max(dim="time", keep_attrs=True) - self._obj.min(dim="time", keep_attrs=True)).astype(
            self._obj.dtype
        )  # ensure dtype out is same as dtype in
        return range_da

    def reduce_rolling(self, samples=25):
        """Remove a rolling mean from the frames.

        Very slow method, so in most cases, it is recommended to use Frames.normalize instead.

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
        assert len(self._obj) >= samples, f"Amount of frames is smaller than requested rolling of {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        # normalize = dask.delayed(cv2.normalize)
        frames_reduce = self._obj - roll_mean
        frames_thres = np.maximum(frames_reduce, 0)
        # # normalize
        frames_norm = (frames_thres * 255 / frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
        frames_norm = frames_norm.where(roll_mean != 0, 0)
        return frames_norm

    def time_diff(self, thres=2, abs=False):
        """Apply a difference over time.

        Method subtracts frame 1 from frame 2, frame 2 from frame 3, etcetera.
        This method is very efficient to highlight moving objects when the video is very stable. If the video
        is very unstable this filter may lead to very bad results.

        Parameters
        ----------
        thres : float, optional
            obsolute value intensity threshold to set values to zero when intensity is lower than this threshold
            default: 2.
        abs : boolean, optional
            if set to True (default: False) apply absolute value on result

        Returns
        -------
        da : xr.DataArray
            filtered frames

        """
        frames_diff = self._obj.astype(np.float32).diff(dim="time")
        frames_diff = frames_diff.where(np.abs(frames_diff) > thres)
        frames_diff.attrs = self._obj.attrs
        # frames_diff -= frames_diff.min(dim=["x", "y"])
        frames_diff = frames_diff.fillna(0.0)
        if abs:
            return np.abs(frames_diff)
        return frames_diff

    def smooth(self, wdw=1):
        """Smooth each frame with a Gaussian kernel.

        Parameters
        ----------
        wdw : int, optional
            window height or width applied. if set to 1 (default) then the total window is 3x3 (i.e. 2 * 1 + 1). When
            set to 2, the total window is 5x5 (i.e. 2 * 2 + 1). Very effective to apply before ``Frames.time_diff``.
            The value for ``wdw`` shouild be chosen such that the moving features of interest are not removed from
            the view. This can be based on a visual interpretation of a result.

        Returns
        -------
        da : xr.DataArray
            filtered frames

        """
        stride = wdw * 2 + 1
        f = xr.apply_ufunc(
            cv._smooth,
            self._obj,
            stride,
            input_core_dims=[["y", "x"], []],
            output_core_dims=[["y", "x"]],
            output_dtypes=[np.float32],
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
        )
        return f

    def to_ani(
        self,
        fn,
        figure_kwargs=const.FIGURE_ARGS,
        video_kwargs=const.VIDEO_ARGS,
        anim_kwargs=const.ANIM_ARGS,
        progress_bar=True,
        **kwargs,
    ):
        """Store an animation of the frames in the object.

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
        progress_bar : bool, optional
            print a progress bar while storing video (default: True)
        **kwargs : dict
            keyword arguments to pass to ``matplotlib.pyplot.imshow``

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
        anim = FuncAnimation(f, animate, init_func=init, frames=frames, **anim_kwargs)
        anim.save(fn, **video_kwargs)

    def to_video(self, fn, video_format=None, fps=None, progress=True):
        """Write frames to a video file without any layout.

        Frames from the input object are written into a video file. The format and frame
        rate can be customized as per user preference or derived automatically from the
        input object.

        Parameters
        ----------
        fn : str
            Path to the output video file.
        video_format : cv2.VideoWriter_fourcc, optional
            The desired video file format codec. If not provided, defaults to
            `cv2.VideoWriter_fourcc(*"mp4v")`.
        fps : float, optional
            Frames per second for the output video. If not specified, it is estimated
            from the time differences in the input frames.
        progress : bool, optional
            Display a progress bar while writing the video frames. (default: True)

        """
        if video_format is None:
            # set to a default
            video_format = cv2.VideoWriter_fourcc(*"mp4v")
        if fps is None:
            # estimate it from the time differences
            fps = 1 / (self._obj["time"].diff(dim="time").values.mean())
        h = self._obj.shape[1]
        w = self._obj.shape[2]
        out = cv2.VideoWriter(fn, video_format, fps, (w, h))
        with tqdm(total=len(self._obj), position=0, leave=True, disable=not (progress)) as pbar:
            pbar.set_description("Writing frames")
            first_frame = True
            for n_start in range(0, len(self._obj), self._obj.chunksize):
                frames_chunk = self._obj.isel(time=slice(n_start, n_start + self._obj.chunksize))
                frames_chunk.load()  # load in memory only once
                for f in frames_chunk:
                    if len(f.shape) == 3:
                        img = cv2.cvtColor(np.uint8(f.values), cv2.COLOR_RGB2BGR)
                    else:
                        img = f.values
                        if first_frame:
                            first_frame = False
                            # make a scale between 0 and 255, only with first frame
                            img_min = img.min(axis=0).min(axis=0)
                            img_max = img.max(axis=0).max(axis=0)
                        img = np.uint8(255 * ((img - img_min) / (img_max - img_min)))
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    out.write(img)
                    pbar.update(1)
        out.release()

    plot = _frames_plot
