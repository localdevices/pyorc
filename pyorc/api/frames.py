import cv2
import dask
import numpy as np
import xarray as xr

from .. import cv, helpers, const, piv_process
from .piv import Velocimetry

class Frames(xr.DataArray):
    __slots__ = ()
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # check for camera_config
        self.name = "frames"
        if hasattr(self, "camera_config"):
            if isinstance(self.camera_config, str):
                # convert into a camera_config object
                from pyorc import get_camera_config
                self.attrs["camera_config"] = get_camera_config(self.camera_config)
        if hasattr(self, "camera_shape"):
            if isinstance(self.camera_shape, str):
                self.attrs["camera_shape"] = helpers.deserialize_attr(self, "camera_shape", np.array)

    def add_xy_coords(
        self,
        xy_coord_data,
        coords,
        attrs_dict
    ):
        """
        add coordinate variables with x and y dimensions (2d) to existing xr.Dataset.

        :param ds: xr.Dataset with at least y and x dimensions
        :param xy_coord_data: list, one or several arrays with 2-dimensional coordinates
        :param coords: tuple with strings, indicating the dimensions of the data in xy_coord_data
        :param attrs_dict: list of dicts, containing attributes belonging to xy_coord_data, must have equal length as xy_coord_data
        :return: xr.Dataset, with added coordinate variables.
        """
        dims = tuple(coords.keys())
        xy_coord_data = [
            xr.DataArray(
                data,
                dims=dims,
                coords=coords,
                attrs=attrs,
                name=name
            ) for data, (name, attrs) in zip(xy_coord_data, attrs_dict.items())]
        # assign the coordinates
        frames_coord = self.assign_coords({
            k: (dims, v) for k, v in zip(attrs_dict, xy_coord_data)
        })
        # add the attributes (not possible with assign_coords
        for k, v in zip(attrs_dict, xy_coord_data):
            frames_coord[k].attrs = v.attrs
        return frames_coord

    def get_piv(self, **kwargs):
        """
        Perform PIV computation on projected frames. Only a pipeline graph to computation is setup. Call a result to
        trigger actual computation.

        :param kwargs: dict, keyword arguments to pass to dask_piv, used to control the manner in which openpiv.pyprocess
            is called.
        :return: Velocimetry object (xr.Dataset), containing the PIV results in a lazy dask.array form.
        """
        # forward the computation to piv
        dask_piv = dask.delayed(piv_process.piv, nout=6)
        v_x, v_y, s2n, corr = [], [], [], []
        frames_a = self[0:-1]
        frames_b = self[1:]
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
        global_attrs = self.attrs
        time = (self.time[0:-1].values + self.time[1:].values) / 2  # frame to frame differences
        # retrieve the x and y-axis belonging to the results
        x, y = helpers.get_axes(cols, rows, self.camera_config.resolution)
        # convert in projected and latlon coordinates
        xs, ys, lons, lats = helpers.get_xs_ys(
            cols,
            rows,
            self.camera_config.transform,
            self.camera_config.crs
        )
        M = self.camera_config.get_M_reverse(self.h_a)
        # compute row and column position of vectors in original reprojected background image col/row coordinates
        xp, yp = helpers.xy_to_perspective(*np.meshgrid(x, np.flipud(y)), self.camera_config.resolution, M)
        # dirty trick to ensure y coordinates start at the top in the right orientation
        shape_y, shape_x = self.camera_shape
        yp = shape_y - yp
        coords = {
            "time": time,
            "y": y,
            "x": x
        }
        # here establish the full xr.Dataset
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
        ds = helpers.add_xy_coords(
            ds,
            [xp, yp, xs, ys, lons, lats],
            coords,
            {**const.PERSPECTIVE_ATTRS, **const.GEOGRAPHICAL_ATTRS}
        )
        # add piv object functionality and attrs to dataset and return
        return Velocimetry(ds, attrs=global_attrs)

    def project(self):
        """
        Project frames into a projected frames object, with information from the camera_config attr.
        This requires that the CameraConfig contains full gcp information and a coordinate reference system (crs).

        :param frames: xr.DataArray with frames, and typical attributes derived from the CameraConfig M, proj_transform and crs
        :return: frames: xr.DataArray with projected frames and x and y axis in local coordinate system (origin: top-left)

        """
        # # read json into CameraConfig object
        # camera_config = helpers.get_camera_config_from_ds(frames)
        # retrieve the M and shape from camera config with said h_a
        M = self.camera_config.get_M(self.h_a)
        shape = self.camera_config.shape
        # get orthoprojected frames as delayed objects
        get_ortho = dask.delayed(cv.get_ortho)
        imgs = [get_ortho(frame, M, tuple(np.flipud(shape)), flags=cv2.INTER_AREA) for frame in self]
        # prepare axes
        time = self.time
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
        xs, ys, lons, lats = helpers.get_xs_ys(
            cols,
            rows,
            self.camera_config.transform,
            self.camera_config.crs
        )
        # Setup coordinates
        coords = {
            "time": time,
            "y": y,
            "x": x
        }
        # add a coordinate if RGB frames are used
        if "rgb" in self.coords:
            coords["rgb"] = np.array([0, 1, 2])
            shape = (*shape, 3)
        # prepare a dask data array
        frames_proj = helpers.delayed_to_da(
            imgs,
            shape,
            "uint8",
            coords=coords,
            attrs=self.attrs,
            object_type=Frames
        )

        # frames_proj = Frames(da)
        # remove time coordinate for the spatial variables (and rgb in case rgb frames are used)
        del coords["time"]
        if "rgb" in self.coords:
            del coords["rgb"]
        # add coordinate meshes to projected frames and return
        return frames_proj.add_xy_coords([xs, ys, lons, lats], coords, const.GEOGRAPHICAL_ATTRS)

    def landmask(self, dilate_iter=10, samples=15):
        """
        Attempt to mask out land from water, by assuming that the time standard deviation over mean of land is much higher
        than that of water. An automatic threshold using Otsu thresholding is used to separate and a dilation operation is
        used to make the land mask a little bit larger than the exact defined pixels.

        :param frames: xr.DataArray with frames
        :param dilate_iter: int, number of dilation iterations to use, to dilate land mask
        :param samples: int, amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
        :return: xr.DataArray with filtered frames

        """
        time_interval = round(len(self)/samples)
        assert(time_interval != 0), f"Amount of frames is too small to provide {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        # compute standard deviation over mean, assuming this value is low over water, and high over land
        std_norm = (self[::time_interval].std(axis=0) / self[::time_interval].mean(axis=0)).load()
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
        return (self * mask) # .astype(bool)


    def normalize(self, samples=15):
        """
        Remove the mean of sampled frames. This is typically used to remove non-moving background from foreground, and helps
        to increase contrast when river bottoms are visible, or when the objective contains partly illuminated and partly
        shaded parts.

        :param frames: xr.DataArray with frames
        :param samples: int, amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
        :return: xr.DataArray with filtered frames
        """
        time_interval = round(len(self) / samples)
        assert (time_interval != 0), f"Amount of frames is too small to provide {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        # normalize = dask.delayed(cv2.normalize)
        mean = self[::time_interval].mean(axis=0).load()
        frames_reduce = self - mean
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

        :param frames: xr.DataArray with frames
        :param samples: int, amount of samples to retrieve from frames for estimating standard deviation and mean. Set to a lower
            number to speed up calculation, default: 15 (which is normally sufficient and fast enough).
        :return: xr.DataArray with filtered frames
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

        shape = self[0].shape  # single-frame shape does not change
        da_convert_edge = dask.delayed(convert_edge)
        imgs = [da_convert_edge(frame.values, stride_1, stride_2) for frame in self]
        # prepare axes
        # Setup coordinates
        coords = {
            "time": self.time,
            "y": self.y,
            "x": self.x
        }
        # add a coordinate if RGB frames are used
        frames_edge = helpers.delayed_to_da(
            imgs,
            shape,
            "uint8",
            coords=coords,
            attrs=self.attrs,
            name="edges",
            object_type=Frames
        )
        frames_edge["xp"] = self["xp"]
        frames_edge["yp"] = self["yp"]
        return frames_edge

    def reduce_rolling(self, samples=25):
        """
        Remove a rolling mean from the frames (very slow, so in most cases, it is recommended to use `normalize` instead).

        :param frames: xr.DataArray with frames
        :param samples: number of samples per rolling
        :return: xr.DataArray with filtered frames
        """
        roll_mean = self.rolling(time=samples).mean()
        assert (len(self) >= samples), f"Amount of frames is smaller than requested rolling interval of {samples} samples"
        # ensure attributes are kept
        xr.set_options(keep_attrs=True)
        # normalize = dask.delayed(cv2.normalize)
        frames_reduce = self - roll_mean
        frames_thres = np.maximum(frames_reduce, 0)
        # # normalize
        frames_norm = (frames_thres * 255 / frames_thres.max(axis=-1).max(axis=-1)).astype("uint8")
        frames_norm = frames_norm.where(roll_mean != 0, 0)
        return frames_norm

    #

    def to_netcdf(self, *args, **kwargs): # -> Union[bytes, "Delayed", None]:
#       # before writing, serialize the camera_config
        self.attrs["camera_config"] = self.camera_config.to_json()
        super().to_netcdf(*args, **kwargs)


def open_frames(fn, *args, **kwargs):
    if not("chunks" in kwargs):
        # add at least the time chunk
        kwargs["chunks"] = {"time": 1}
    ds = xr.open_dataset(fn, *args, **kwargs)
    assert("frames" in ds), f"Opened dataset does not contain frames"
    return Frames(ds["frames"])

