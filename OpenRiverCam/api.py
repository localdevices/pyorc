import cv2
import dask
import dask.array as da
import json
import matplotlib.pyplot as plt
import numpy as np
from OpenRiverCam import cv, io
from pyproj import CRS, Transformer

def from_file(fn):
    with open(fn, "r") as f:
        dict = json.loads(f.read())
    return CameraConfig(**dict)

class Video(cv2.VideoCapture):
    def __init__(
            self,
            fn,
            h_a,
            resolution=0.01,
            window_size=15,
            corners=None,
            start_frame=None,
            end_frame=None,
            *args,
            **kwargs
    ):

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
        # set other properties
        self.h_a = h_a
        self.resolution = resolution
        self.window_size = window_size
        if corners is not None:
            self.corners = corners
        self.fn = fn
        self.transform = None

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

    def set_corners(self, x, y):
        """
        The coordinates must be provided in the following order:
        - upstream left-bank
        - downstream left-bank
        - downstream right-bank
        - upstream right-bank

        :param x: list, x-coordinates of corners
        :param y: list, y-coordinates of corners

        :return:
        """
        self._corners = {}
        keys = ["up_left", "down_left", "down_right", "up_right"]

        for k, _x, _y in zip(keys, x, y):
            self._corners[k] = [_x, _y]

    def set_frames(self, cam_config, **kwargs):
        """
        Make a dask array of frames, expected to be read

        :return: frame
        """
        get_frame = dask.delayed(io.get_frame, pure=True)  # Lazy version of get_frame
        assert(isinstance(cam_config, CameraConfig))

        frames = [get_frame(self, n=n, lens_pars=cam_config.lens_pars, **kwargs) for n in range(self.start_frame, self.end_frame)]
        sample = frames[0].compute()
        data_array = [da.from_delayed(
            frame,
            dtype=sample.dtype,
            shape=sample.shape
        ) for frame in frames]

        self._frames = da.stack(data_array, axis=0)

    def set_project_frames(self, cam_config):
        get_ortho = dask.delayed(cv.get_ortho)
        bbox = cv.get_aoi(cam_config.gcps["src"], cam_config.gcps["dst"], self.corners)

        # get the perspective transformation matrix
        M = cam_config.get_M(bbox, self.h_a, self.resolution)

        # get the desired target shape and transform
        transform, shape = cam_config.get_transform(bbox, self.resolution)
        # store the transform
        self.transform = transform

        imgs = [get_ortho(frame, M, shape, flags=cv2.INTER_AREA) for frame in self.frames]

        # retrieve one sample to setup a lazy dask array
        sample = imgs[0].compute()
        data_array = [da.from_delayed(
            img,
            dtype=sample.dtype,
            shape=sample.shape
        ) for img in imgs]
        self._proj_frames = da.stack(data_array, axis=0)


    @property
    def frames(self):
        return self._frames

        #
        # iter = io.frames(self, lens_pars=cam_config.lens_pars, **kwargs)
        # return iter


class CameraConfig:
    def __init__(self,
                 crs,
                 aoi_window_size=15,
                 lens_position=None,
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

        assert(isinstance(aoi_window_size, int)), 'aoi_window_size must be of type "int"'
        if crs is not None:
            try:
                crs = CRS.from_user_input(crs)
            except:
                raise ValueError(f"crs {crs} is not a valid Coordinate Reference System")
            assert(crs.is_geographic == 0), "Provided crs must be projected with units like [m]"
        self.crs = crs.to_wkt()
        if lens_position is not None:
            self.set_lens_position(*lens_position)
        if gcps is not None:
            self.set_gcps(**gcps)
        if lens_pars is not None:
            self.set_lens_pars(**lens_pars)

    def get_M(self, bbox, h_a, resolution):
        dst_a = cv._get_gcps_a(
            self.lens_position,
            h_a,
            self.gcps["dst"],
            self.gcps["z_0"],
            self.gcps["h_ref"],
        )
        dst_colrow_a = cv._transform_to_bbox(dst_a, bbox, resolution)

        # retrieve M for destination row and col
        return cv._get_M(src=self.gcps["src"], dst=dst_colrow_a)

    def get_transform(self, bbox, resolution):
        # estimate size of required grid
        transform = cv._get_transform(bbox, resolution=resolution)
        # TODO: alter method to determine window_size based on how PIV is done. If only squares are possible, then this can be one single nr.
        cols, rows = cv._get_shape(
            bbox, resolution=resolution, round=10
        )
        return transform, (cols, rows)

    def set_lens_pars(self, k1=0, c=2, f=4):
        """
        Set the lens parameters of the given CameraConfig

        :param k1: float, lens curvature [-], zero (default) means no curvature
        :param c: float, optical centre [1/n], whgere n is the fraction of the lens diameter, 2.0 (default) means in the centre
        :param f: float, focal length [mm], typical values could be 2.8, or 4 (default)
        :return: None
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

        :param src:
        :param dst:
        :param h_ref:
        :param z_0:
        :param crs:
        :return: None
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
            try:
                crs = CRS.from_user_input(crs)
            except:
                raise ValueError(f"crs {crs} is not a valid Coordinate Reference System")
            transform = Transformer.from_crs(crs, CRS.from_wkt(self.crs), always_xy=True)
            # transform dst coordinates to local projection
            _x, _y = zip(*dst)
            x, y = transform.transform(_x, _y)
            # replace them
            dst = list(zip(x, y))
        self.gcps = {
            "src": src,
            "dst": dst,
            "h_ref": h_ref,
            "z_0": z_0,
        }


    def set_lens_position(self, x, y, z, crs=None):
        """
        Set the geographical position of the lens of current CameraConfig

        :param x:
        :param y:
        :param z:
        :param crs:
        :return:
        """

        if crs is not None:
            if self.crs is None:
                raise ValueError("CameraConfig does not contain a crs, ")
            try:
                crs = CRS.from_user_input(crs)
            except:
                raise ValueError(f"crs {crs} is not a valid Coordinate Reference System")
            transform = Transformer.from_crs(crs, self.crs, always_xy=True)
            # transform dst coordinates to local projection
            x, y = transform.transform(x, y)
            # replace them

        self.lens_position = [x, y, z]

    def plot(self, figsize=(13, 8), ax=None, tiles=None, buffer=0.0002, zoom_level=18, tiles_kwargs={}):
        """
        Plot the geographical situation of the CameraConfig

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
        plt.legend()
        return ax

    def to_file(self, fn):
        """
        Write the CameraConfig object to json structure
        :return:
        """
        with open(fn, "w") as f:
            f.write(json.dumps(self, default=lambda o: o.__dict__, indent=4))

    def to_dict(self):
        """
        Return the CameraConfig object as dictionary
        :return:
        """
        return self.__dict__

