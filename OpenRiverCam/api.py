from pyproj import CRS, Transformer
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from OpenRiverCam import cv, io

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
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.open(fn)
        self.h_a = h_a
        self.resolution = resolution,
        self.window_size = window_size
        if corners is not None:
            self.corners = corners

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
        self.corners = {}
        keys = ["up_left", "down_left", "down_right", "up_right"]

        for k, _x, _y in zip(keys, x, y):
            self.corners[k] = [_x, _y]

    def read_project(self, cam_config, **kwargs):
        """
        Read and immediately project a frame

        :return: frame
        """
        assert(isinstance(cam_config, CameraConfig))
        bbox = cv.get_aoi(cam_config.gcps["src"], cam_config.gcps["dst"], self.corners)
        iter = io.frames(self, lens_pars=cam_config.lens_pars, **kwargs)
        return iter


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

