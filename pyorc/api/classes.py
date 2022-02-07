import cv2
import dask
import dask.array as da

import json
import matplotlib.pyplot as plt
import numpy as np
import rasterio.transform
import shapely.wkt
import xarray as xr

from pyorc import cv, io, helpers, const
from pyproj import CRS, Transformer

class Video(cv2.VideoCapture):
    def __init__(
            self,
            fn,
            camera_config,
            h_a,
            start_frame=None,
            end_frame=None,
            *args,
            **kwargs
    ):
        """
        Video class, strongly inherited from cv2.VideoCapture. This class extends cv2.VideoCapture by adding a
        camera configuration to it, and a start and end frame to read from the video. Several methods are added to
        read frames into memory or into a xr.DataArray with attributes. These can then be processed with other pyorc
        API functionalities.

        :param fn: str, locally stored video file
        :param camera_config: CameraConfig object, containing all information about the camera, lens parameters, lens
            position, ground control points with GPS coordinates, and all referencing information (see CameraConfig),
            needed to reproject frames on a horizontal geographically referenced plane.
        :param h_a: float, actual height [m], measured in local vertical reference (e.g. a staff gauge in view of the
            camera)
        :param start_frame: int, first frame to use in analysis
        :param end_frame: int, last frame to use in analysis
        :param args: list or tuple, arguments to pass to cv2.VideoCapture on initialization.
        :param kwargs: dict, keyword arguments to pass to cv2.VideoCapture on initialization.
        """

        super().__init__(*args, **kwargs)
        # explicitly open file for reading
        self.open(fn)
        # set end and start frame
        self.frame_count = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame is not None:
            if (start_frame > self.frame_count and self.frame_count > 0):
                raise ValueError("Start frame is larger than total amount of frames")
        if end_frame is not None:
            if end_frame < start_frame:
                raise ValueError(
                    f"Start frame {start_frame} is larger than end frame {end_frame}"
                )
        self.end_frame = end_frame
        self.start_frame = start_frame
        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.frame_number = 0
        # set other properties
        self.h_a = h_a
        # get the perspective transformation matrix (movie dependent)
        self.M = camera_config.get_M(self.h_a)
        self.M_reverse = camera_config.get_M_reverse(self.h_a)
        self.lens_pars = camera_config.lens_pars
        self.fn = fn
        self.transform = camera_config.transform
        self.resolution = camera_config.resolution
        self.shape = camera_config.shape
        self.window_size = camera_config.window_size
        self.crs = camera_config.crs
        self._stills = {}  # here all stills are stored lazily

    @property
    def end_frame(self):
        return self._end_frame

    @end_frame.setter
    def end_frame(self, end_frame=None):
        if end_frame is None:
            self._end_frame = self.frame_count
        else:
            self._end_frame = min(self.frame_count, end_frame)

    @property
    def start_frame(self):
        return self._start_frame

    @start_frame.setter
    def start_frame(self, start_frame=None):
        if start_frame is None:
            self._start_frame = 0
        else:
            self._start_frame = start_frame

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        if (np.isinf(fps)) or (fps <= 0):
            raise ValueError(f"FPS in video is {fps} which is not a valid value. Repair the video file before use")
        self._fps = fps

    @property
    def corners(self):
        return self._corners

    @corners.setter
    def corners(self, corners):
        self._corners = corners

    def get_frame(self, n, lens_pars=None, grayscale=False):
        """
        Retrieve one frame. Frame will be corrected for lens distortion if lens parameters are given.

        :param n: int, frame number to retrieve
        :param lens_pars: dict, containing lens parameters k1, c and f
        :param grayscale: bool, if set to `True`, frame will be grayscaled
        :return: np.ndarray containing frame
        """

        cap = cv2.VideoCapture(self.fn)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        try:
            ret, img = cap.read()
        except:
            raise IOError(f"Cannot read")
        if ret:
            if lens_pars is not None:
                # apply lens distortion correction
                img = io._corr_lens(img, **lens_pars)
            if grayscale:
                # apply gray scaling, contrast- and gamma correction
                # img = _corr_color(img, alpha=None, beta=None, gamma=0.4)
                img = img.mean(axis=2)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_count = n + 1
        cap.release()
        return img

    def get_frames(self, **kwargs):
        """
        Get a xr.DataArray, containing a dask array of frames, from `start_frame` until `end_frame`, expected to be read
        lazily. The xr.DataArray will contain all coordinate variables and attributes, needed for further processing
        steps.

        :param kwargs: dict, keyword arguments to pass to `get_frame`. Currently only `grayscale` is supported.
        :return: xr.DataArray, containing all requested frames
        """
        get_frame = dask.delayed(self.get_frame, pure=True)  # Lazy version of get_frame
        frames = [get_frame(n=n, lens_pars=self.lens_pars, **kwargs) for n in range(self.start_frame, self.end_frame)]
        sample = frames[0].compute()
        data_array = [da.from_delayed(
            frame,
            dtype=sample.dtype,
            shape=sample.shape
        ) for frame in frames]
        time = np.arange(len(data_array))*1/self.fps
        y = np.flipud(np.arange(data_array[0].shape[0]))
        # y = np.arange(data_array[0].shape[0])
        x = np.arange(data_array[0].shape[1])
        # perspective column and row coordinate grids
        xp, yp = np.meshgrid(x, y)
        coords = {
            "time": time,
            "y": y,
            "x": x
        }
        if len(sample.shape) == 3:
            coords["rgb"] = np.array([0, 1, 2])

        dims = tuple(coords.keys())
        attrs = {
            "M": str(self.M.tolist()),
            "M_reverse": str(self.M_reverse.tolist()),
            "proj_transform": str(list(self.transform)[0:6]),
            "proj_shape": str(self.shape),
            "camera_shape": str([len(y), len(x)]),
            "crs": self.crs,
            "resolution": self.resolution,
            "window_size": self.window_size
        }
        data_array = xr.DataArray(
            da.stack(data_array, axis=0),
            dims=dims,
            coords=coords,
            attrs=attrs
        )
        del coords["time"]
        if len(sample.shape) == 3:
            del coords["rgb"]
        # add coordinate grids
        data_array = helpers.add_xy_coords(data_array, [xp, yp], coords, const.PERSPECTIVE_ATTRS)
        return data_array

class CameraConfig:
    def __init__(self,
                 crs,
                 window_size=15,
                 resolution=0.01,
                 lens_position=None,
                 bbox=None,
                 transform=None,
                 shape=None,
                 corners=None,
                 gcps=None,
                 lens_pars=None
                 ):
        """
        Initiate a CameraConfig object with several (default) settings. This object allows for treatment of movies
        with defined settings

        :param crs: int, dict, or str, optional Coordinate Reference System. Accepts EPSG codes (int or str);
            proj (str or dict) or wkt (str). Only used if the data has no native CRS.
        :param id:
        :param resolution:
        :param aoi_window_size:
        """

        assert(isinstance(window_size, int)), 'window_size must be of type "int"'
        if crs is not None:
            try:
                crs = CRS.from_user_input(crs)
            except:
                raise ValueError(f"crs {crs} is not a valid Coordinate Reference System")
            assert(crs.is_geographic == 0), "Provided crs must be projected with units like [m]"
        self.crs = crs.to_wkt()
        if resolution is not None:
            self.resolution = resolution
        if lens_position is not None:
            self.set_lens_position(*lens_position)
        if gcps is not None:
            self.set_gcps(**gcps)
        if lens_pars is not None:
            self.set_lens_pars(**lens_pars)
        if bbox is not None:
            self.bbox = bbox
        if transform is not None:
            self.transform = transform
        if shape is not None:
            self.shape = shape
        if window_size is not None:
            self.window_size = window_size
        # override the transform and bbox with the set corners
        if (corners is not None and hasattr(self, "gcps")):
            self.set_bbox(corners)
            # get the desired target shape and transform for future videos using this CameraConfig
            self.set_transform_from_bbox()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        if isinstance(transform, rasterio.transform.Affine):
            self._transform = transform
        else:
            # try to set it via rasterio Affine, assuming it is a (minimal) 6 value list
            self._transform = rasterio.transform.Affine(*transform[0:6])

    def get_M(self, h_a):
        """
        Establish a transformation matrix for a certain actual water level `h_a`. This is done by mapping where the
        ground control points, measured at `h_ref` will end up with new water level `h_a`, given the lens position.

        :param h_a: actual water level [m]
        :return: np.ndarray, containing 2x3 transformation matrix.
        """
        # map where the destination points are with the actual water level h_a.
        dst_a = cv._get_gcps_a(
            self.lens_position,
            h_a,
            self.gcps["dst"],
            self.gcps["z_0"],
            self.gcps["h_ref"],
        )
        # lookup where the destination points are in row/column space
        dst_colrow_a = cv._transform_to_bbox(dst_a, shapely.wkt.loads(self.bbox), self.resolution)

        # retrieve and return M for destination row and col
        return cv._get_M(src=self.gcps["src"], dst=dst_colrow_a)

    def get_M_reverse(self, h_a):
        """
        Establish the reverse transformation matrix for certain actual water level. This is used to be able to
        transform coordinates from the real space to the camera perspective (reverse of `get_M`).

        :param h_a: actual water level [m]
        :return: np.ndarray, containing 2x3 transformation matrix.
        """
        dst_a = cv._get_gcps_a(
            self.lens_position,
            h_a,
            self.gcps["dst"],
            self.gcps["z_0"],
            self.gcps["h_ref"],
        )
        dst_colrow_a = cv._transform_to_bbox(dst_a, shapely.wkt.loads(self.bbox), self.resolution)

        # retrieve M reverse for destination row and col
        return cv._get_M(src=dst_colrow_a, dst=self.gcps["src"])

    def set_bbox(self, corners):
        self.bbox = cv.get_aoi(self.gcps["src"], self.gcps["dst"], corners).__str__()

    def set_transform_from_bbox(self):
        # estimate size of required grid
        bbox = shapely.wkt.loads(self.bbox)
        transform = cv._get_transform(bbox, resolution=self.resolution)
        # TODO: alter method to determine window_size based on how PIV is done. If only squares are possible, then this can be one single nr.
        cols, rows = cv._get_shape(
            bbox, resolution=self.resolution, round=10
        )
        self._transform = transform
        self.shape = (rows, cols)

    def set_lens_pars(self, k1=0, c=2, f=4):
        """
        Set the lens parameters of the given CameraConfig

        :param k1: float, lens curvature [-], zero (default) means no curvature
        :param c: float, optical centre [1/n], whgere n is the fraction of the lens diameter, 2.0 (default) means in the centre
        :param f: float, focal length [mm], typical values could be 2.8, or 4 (default)
        """
        assert(isinstance(k1, (int, float))), "k1 must be a float"
        assert(isinstance(c, (int, float))), "k1 must be a float"
        assert(isinstance(f, (int, float))), "k1 must be a float"
        self.lens_pars = {
            "k1": k1,
            "c": c,
            "f": f
        }


    def set_gcps(self, src, dst, h_ref, z_0, crs=None):
        """
        Set ground control points for the given CameraConfig

        :param src: list of lists, containing 4 x, y pairs of columns and rows in the frames of the original video
        :param dst: list of lists, containing 4 x, y pairs of real world coordinates in the given coordinate reference
            system.
        :param h_ref: water level, belonging to the 4 control points in `dst`. This is the water level as measured by
            a local reference (e.g. gauge plate) during the surveying of the control points. Control points must be
            taken on the water surface.
        :param z_0: same as `h_ref` but then as measured by a global reference system such as a geoid or ellipsoid used
            by a GPS device. All other surveyed points (lens position and cross section) must have the same vertical
            reference.
        :param crs: coordinate reference system, used to measure the control points (e.g. 4326 for WGS84 lat-lon).
            the destination control points will automatically be reprojected to the local crs of the CameraConfig.
        """
        assert(isinstance(src, list)), f"src must be a list of 4 numbers"
        assert(isinstance(dst, list)), f"dst must be a list of 4 numbers"
        assert (len(src) == 4), f"4 source points are expected in src, but {len(src)} were found"
        assert (len(dst) == 4), f"4 destination points are expected in dst, but {len(dst)} were found"
        assert(isinstance(h_ref, (float, int))), "h_ref must contain a float number"
        assert(isinstance(z_0, (float, int))), "z_0 must contain a float number"
        assert(all(isinstance(x, int) for p in src for x in p)), "src contains non-int parts"
        assert(all(isinstance(x, (float, int)) for p in dst for x in p)), "dst contains non-float parts"
        if crs is not None:
            if self.crs is None:
                raise ValueError("CameraConfig does not contain a crs, ")
            _x, _y = zip(*dst)
            x, y = helpers.xy_transform(_x, _y, crs, CRS.from_wkt(self.crs))
            # replace transformed coordinates
            dst = list(zip(x, y))
        self.gcps = {
            "src": src,
            "dst": dst,
            "h_ref": h_ref,
            "z_0": z_0,
        }


    def set_lens_position(self, x, y, z, crs=None):
        """
        Set the geographical position of the lens of current CameraConfig.

        :param x:
        :param y:
        :param z:
        :param crs: coordinate reference system, used to measure the lens position (e.g. 4326 for WGS84 lat-lon).
            the position's x and y coordinates will automatically be reprojected to the local crs of the CameraConfig.
        """

        if crs is not None:
            if self.crs is None:
                raise ValueError("CameraConfig does not contain a crs, ")
            x, y = helpers.xy_transform(x, y, crs, self.crs)

        self.lens_position = [x, y, z]

    def plot(self, figsize=(13, 8), ax=None, tiles=None, buffer=0.0002, zoom_level=18, tiles_kwargs={}):
        """
        Plot the geographical situation of the CameraConfig. This is very useful to check if the CameraConfig seems
        to be in the right location. Requires cartopy to be installed.

        :return: ax
        """
        try:
            import cartopy
            import cartopy.io.img_tiles as cimgt
            import cartopy.crs as ccrs
            from shapely.geometry import LineString, Point
            from shapely import ops
        except:
            raise ModuleNotFoundError(
                'Geographic plotting requires cartopy. Please install it with "conda install cartopy" and try again')
        # make a transformer to lat lon
        transform = Transformer.from_crs(CRS.from_user_input(self.crs), CRS.from_epsg(4326), always_xy=True).transform

        if ax is None:
            if tiles is not None:
                tiler = getattr(cimgt, tiles)()
                crs = tiler.crs
            else:
                crs = ccrs.PlateCarree()
            f = plt.figure(figsize=figsize)
            # make point collection
            if not(hasattr(self, "gcps")):
                raise("No GCPs found yet, please populate the gcps attribute with set_gcps first.")
            points = [Point(x, y) for x, y in self.gcps["dst"]]
            if hasattr(self, "lens_position"):
                points.append(Point(self.lens_position[0], self.lens_position[1]))

            points_lonlat = [ops.transform(transform, p) for p in points]

            xmin, ymin, xmax, ymax = list(np.array(LineString(points_lonlat).bounds))
            bbox = [xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer]
            ax = plt.subplot(projection=crs)
            ax.set_extent(bbox, crs=ccrs.PlateCarree())
            if tiles is not None:
                tiler = getattr(cimgt, tiles)()
                ax.add_image(tiler, zoom_level, **tiles_kwargs)
        x = [p.x for p in points_lonlat]
        y = [p.y for p in points_lonlat]
        ax.plot(x[0:4], y[0:4], ".", markersize=16, transform=ccrs.PlateCarree(), zorder=2, label="Control points")  #
        if hasattr(self, "lens_position"):
            ax.plot(x[-1], y[-1], ".", markersize=16, transform=ccrs.PlateCarree(), zorder=2, label="Lens position")  # transform=ccrs.PlateCarree()
        bbox = shapely.wkt.loads(self.bbox)
        bbox_trans = ops.transform(transform, bbox)
        ax.add_geometries([bbox_trans], ccrs.PlateCarree(), alpha=0.5,  label="Area of interest")

        ax.legend()
        return ax

    def to_file(self, fn):
        """
        Write the CameraConfig object to json structure
        :return:
        """

        with open(fn, "w") as f:
            f.write(json.dumps(self, default=lambda o: o.to_dict(), indent=4))

    def to_dict(self):
        """
        Return the CameraConfig object as dictionary
        :return:
        """

        dict = self.__dict__
        # replace underscore keys for keys without underscore
        for k in list(dict.keys()):
            if k[0] == "_":
                dict[k[1:]] = dict.pop(k)

        return dict


def load_camera_config(fn):
    """
    Load a CameraConfig from a geojson file.

    :param fn: str, filename with camera config.
    :return: CameraConfig object
    """
    with open(fn, "r") as f:
        dict = json.loads(f.read())
    return CameraConfig(**dict)


