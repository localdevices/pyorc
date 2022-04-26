import cv2
import dask
import numpy as np
import xarray as xr

from matplotlib.collections import QuadMesh
from .orcbase import ORCBase
from .. import cv, helpers, const, piv_process


@xr.register_dataarray_accessor("frames")
class Frames(ORCBase):
    def __init__(self, xarray_obj):
        super(Frames, self).__init__(xarray_obj)

    def get_piv(self, **kwargs):
        """
        Perform PIV computation on projected frames. Only a pipeline graph to computation is setup. Call a result to
        trigger actual computation.

        :param kwargs: dict, keyword arguments to pass to dask_piv, used to control the manner in which
            openpiv.pyprocess is called.
        :return: Velocimetry object (xr.Dataset), containing the PIV results in a lazy dask.array form.
        """
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
                res_x=self.camera_config.resolution,
                res_y=self.camera_config.resolution,
                dt=float(dt.values),
                search_area_size=self.camera_config.window_size,
                **kwargs,
            )
            # append to result
            v_x.append(_v_x), v_y.append(_v_y), s2n.append(_s2n), corr.append(_corr)
        # compute one sample for the spacing
        cols, rows, _v_x, _v_y, _s2n, _corr = piv_process.piv(
            frame_a,
            frame_b,
            res_x=self.camera_config.resolution,
            res_y=self.camera_config.resolution,
            dt=float(dt.values),
            search_area_size=self.camera_config.window_size,
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
            self.camera_config.transform,
        )
        if hasattr(self.camera_config, "crs"):
            lons, lats = helpers.get_lons_lats(xs, ys, self.camera_config.crs)
        else:
            lons = None
            lats = None
        M = self.camera_config.get_M(self.h_a, reverse=True)
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
        ds = ds.velocimetry.add_xy_coords(
            [xp, yp, xs, ys, lons, lats],
            coords,
            {**const.PERSPECTIVE_ATTRS, **const.GEOGRAPHICAL_ATTRS}
        )
        # add piv object functionality and attrs to dataset and return
        ds = xr.Dataset(ds, attrs=global_attrs)
        ds.velocimetry.set_encoding()
        return ds

    # def plot(self, ax=None, mode="local", **kwargs):
    #     """
    #     plot a frame. this can be useful for plotting as background for a velocimetry result. Plotting can be done in
    #     three modes:
    #     - "local": a simple planar view plot, with a local coordinate system in meters, with the top-left coordinate
    #       being the 0, 0 point, and ascending coordinates towards the right and bottom.
    #     - "geographical": a geographical plot, requiring the package `cartopy`, the results are plotted on a geographical
    #       axes, so that combinations with tile layers such as OpenStreetMap, or shapefiles can be made.
    #     - "camera": i.e. seen from the camera perspective. This is the most intuitive view for end users.
    #
    #     :param ax: pre-defined axes object. If not set, a new axes will be prepared. In case `mode=="geographical"`, a
    #         cartopy GeoAxes needs to be provided, or will be made in case ax is not set.
    #     :param mode: can be "local", "geographical", or "camera". For "geographical" a frames set from frames.project
    #         should be used that contains "lon" and "lat" coordinates. For "camera", a non-projected frames set should
    #         be used.
    #     :param kwargs: dict, plotting parameters to be passed to matplotlib.pyplot.pcolormesh, for plotting the
    #         background frame.
    #     """
    #     # prepare axes
    #     if "time" in self._obj.coords:
    #         if self._obj.time.size > 1:
    #             raise AttributeError(f'Object contains dimension "time" with length {len(self._obj.time)}. Reduce dataset by selecting one time step or taking a median, mean or other statistic.')
    #     ax = plot_orc.prepare_axes(ax=ax, mode=mode)
    #     f = ax.figure  # handle to figure
    #     if mode == "local":
    #         x = "x"
    #         y = "y"
    #     elif mode == "geographical":
    #         # import some additional packages
    #         import cartopy.crs as ccrs
    #         # add transform for GeoAxes
    #         kwargs["transform"] = ccrs.PlateCarree()
    #         x = "lon"
    #         y = "lat"
    #     else:
    #         # mode is camera
    #         x = "xp"
    #         y = "yp"
    #     assert all(v in self._obj.coords for v in [x, y]), f'required coordinates "{x}" and/or "{y}" are not available'
    #     if (len(self._obj.shape) == 3 and self._obj.shape[-1] == 3):
    #         # looking at an rgb image
    #         facecolors = self._obj.values.reshape(self._obj.shape[0] * self._obj.shape[1], 3) / 255
    #         facecolors = np.hstack([facecolors, np.ones((len(facecolors), 1))])
    #         quad = ax.pcolormesh(self._obj[x], self._obj[y], self._obj.mean(dim="rgb"), shading="nearest",
    #                              facecolors=facecolors, **kwargs)
    #         # remove array values, override .set_array, needed in case GeoAxes is provided, because GeoAxes asserts if array has dims
    #         QuadMesh.set_array(quad, None)
    #     else:
    #         ax.pcolormesh(self._obj[x], self._obj[y], self._obj, **kwargs)
    #     # fix axis limits to min and max of extent of frames
    #     ax.set_xlim([self._obj[x].min(), self._obj[x].max()])
    #     ax.set_ylim([self._obj[y].min(), self._obj[y].max()])
    #     return ax

    def project(self):
        """
        Project frames into a projected frames object, with information from the camera_config attr.
        This requires that the CameraConfig contains full gcp information and a coordinate reference system (crs).

        :return: frames: xr.DataArray with projected frames and x and y in local coordinate system (origin: top-left)

        """
        # retrieve the M and shape from camera config with said h_a
        M = self.camera_config.get_M(self.h_a)
        shape = self.camera_config.shape
        # get orthoprojected frames as delayed objects
        get_ortho = dask.delayed(cv.get_ortho)
        imgs = [get_ortho(frame, M, tuple(np.flipud(shape)), flags=cv2.INTER_AREA) for frame in self._obj]
        # prepare axes
        time = self._obj.time
        y = np.flipud(np.linspace(
            self.camera_config.resolution/2,
            self.camera_config.resolution*(shape[0]-0.5),
            shape[0])
        )
        x = np.linspace(
            self.camera_config.resolution/2,
            self.camera_config.resolution*(shape[1]-0.5), shape[1]
        )
        cols, rows = np.meshgrid(
            np.arange(len(x)),
            np.arange(len(y))
        )
        # retrieve all coordinates we may ever need for further analysis or plotting
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

        # frames_proj = Frames(da)
        # remove time coordinate for the spatial variables (and rgb in case rgb frames are used)
        del coords["time"]
        if "rgb" in self._obj.coords:
            del coords["rgb"]
        # add coordinate meshes to projected frames and return
        frames_proj = frames_proj.frames.add_xy_coords([xs, ys, lons, lats], coords, const.GEOGRAPHICAL_ATTRS)
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
        # normalize = dask.delayed(cv2.normalize)
        mean = self._obj[::time_interval].mean(axis=0).load()
        frames_reduce = self._obj - mean
        #     frames_norm = cv2.normalize(frames_reduce)
        # frames_min = frames_reduce.min(axis=-1).min(axis=-1)
        # frames_max = frames_reduce.max(axis=-1).min(axis=-1)
        # frames_norm = ((frames_reduce - frames_min) / (frames_max - frames_min) * 255).astype("uint8")

        frames_thres = np.maximum(frames_reduce, 0)
        # normalize
        frames_norm = (frames_thres*255/frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
        frames_norm = frames_norm.where(mean!=0, 0)
        return frames_norm


    def edge_detection(self, stride_1=7, stride_2=9):
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
            blur1 = cv2.GaussianBlur(img, (stride_1, stride_1), 0)
            blur2 = cv2.GaussianBlur(img, (stride_2, stride_2), 0)
            edges = blur2 - blur1
            mask = edges == 0
            edges = ((edges - edges.min()) / (edges.max() - edges.min()) * 255).astype("uint8")
            edges = cv2.equalizeHist(edges)
            edges[mask] = 0
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
            "uint8",
            coords=coords,
            attrs=self._obj.attrs,
            name="edges",
            object_type=Frames
        )
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
