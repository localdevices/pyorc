import json
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from .. import cv, helpers


class CameraConfig:
    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return self.to_json()

    def __init__(
            self,
            crs=None,
            window_size=15,
            resolution=0.01,
            lens_position=None,
            corners=None,
            gcps=None,
            lens_pars=None
    ):
        """
        CameraConfig object with several settings. This object allows for treatment of movies
        with defined settings.

        :param crs: int, dict, or str, optional. Coordinate Reference System. Accepts EPSG codes (int or str);
            proj (str or dict) or wkt (str). Only used if the data has no native CRS.
        :param window_size: int, pixel size of interrogation window (default: 15)
        :param resolution: float, resolution in m. of projected pixels
        :param lens_position: list of 3 floats, containing x, y, z coordinate of lens position in CRS
        :param corners: list of 4 lists with [x, y] coordinates defining corners of area of interest in camera cols/rows
        :param gcps: dict, containing:
            "src": list of 4 lists, with column, row locations in objective of control points.
            "dst": list of 4 lists, with x, y locations (local or global coordinate reference system) of control points.
            "h_ref": float, measured water level [m] in local reference system (e.g. from staff gauge or pressure gauge)
                during gcp survey.
            "z_0": float, water level [m] in global reference system (e.g. from used GPS system CRS). This must be in
                the same vertical reference as the measured bathymetry and other survey points.
            "crs": int, str or CRS object, CRS in which "dst" points are measured. If None, a local coordinate system is
                assumed (e.g. from spirit level).
        :param lens_pars: dict, containing:
            "k1": float, barrel lens distortion parameter (default: 0.)
            "c": float, optical center (default: 2.)
            "f": float, focal length (default: 1.)
        """
        assert (isinstance(window_size, int)), 'window_size must be of type "int"'
        if crs is not None:
            try:
                crs = CRS.from_user_input(crs)
            except CRSError:
                raise CRSError(f'crs "{crs}" is not a valid Coordinate Reference System')
            assert (crs.is_geographic == 0), "Provided crs must be projected with units like [m]"
            self.crs = crs.to_wkt()
        if resolution is not None:
            self.resolution = resolution
        if lens_position is not None:
            self.set_lens_position(*lens_position)
        if gcps is not None:
            self.set_gcps(**gcps)
        if lens_pars is not None:
            self.set_lens_pars(**lens_pars)
        if window_size is not None:
            self.window_size = window_size
        # override the transform and bbox with the set corners
        if corners is not None:
            self.set_corners(corners)

    @property
    def bbox(self):
        assert (hasattr(self, "corners")), "CameraConfig object has no corners, set these with CameraConfig.set_corners"
        return cv.get_aoi(self.gcps["src"], self.gcps["dst"], self.corners).__str__()

    @property
    def shape(self):
        cols, rows = cv.get_shape(
            shapely.wkt.loads(self.bbox),
            resolution=self.resolution,
            round=10
        )
        return rows, cols

    @property
    def transform(self):
        bbox = shapely.wkt.loads(self.bbox)
        return cv.get_transform(bbox, resolution=self.resolution)

    def get_depth(self, z, h_a=None):
        """
        Retrieve depth for measured bathymetry points using the camera configuration and an actual water level, measured
        in local reference (e.g. staff gauge).

        :param z: measured bathymetry points
        :param h_a: float, optional, actual water level measured [m], if not set, assumption is that a single video
            is processed and thus changes in water level are not relevant.
        :return:
        """
        if h_a is None:
            assert(self.gcps["h_ref"] is None), "No actual water level is provided, but a reference water level is provided"
            h_a = 0.
            h_ref = 0.
        else:
            h_ref = self.gcps["h_ref"]
        z_pressure = np.maximum(self.gcps["z_0"] - h_ref + h_a, z)
        return z_pressure - z

    def z_to_h(self, z):
        """
        Convert z coordinates of bathymetry to height coordinates in local reference (e.g. staff gauge)

        :param z: measured bathymetry point
        :return:
        """
        if self.gcps["h_ref"] is None:
            h_ref = 0.
        else:
            h_ref = self.gcps["h_ref"]
        h = z + h_ref - self.gcps["z_0"]
        return h


    def get_M(self, h_a=None, reverse=False):
        """
        Establish a transformation matrix for a certain actual water level `h_a`. This is done by mapping where the
        ground control points, measured at `h_ref` will end up with new water level `h_a`, given the lens position.

        :param h_a: actual water level [m]
        :param reverse: bool, optional, if set, the reverse matrix is prepared, which can be used to transform projected
            coordinates back to the original camera perspective.
        :return: np.ndarray, containing 2x3 transformation matrix.
        """
        # map where the destination points are with the actual water level h_a.
        if h_a is None or self.gcps["h_ref"] is None:
            # fill in the same value for h_ref and h_a
            h_ref = 0.
            h_a = 0.
        else:
            h_ref = self.gcps["h_ref"]
        dst_a = cv.get_gcps_a(
            self.lens_position,
            h_a,
            self.gcps["dst"],
            self.gcps["z_0"],
            h_ref,
        )

        # lookup where the destination points are in row/column space
        dst_colrow_a = cv.transform_to_bbox(dst_a, shapely.wkt.loads(self.bbox), self.resolution)
        if reverse:
            # retrieve and return M reverse for destination row and col
            return cv.get_M(src=dst_colrow_a, dst=self.gcps["src"])
        else:
            # retrieve and return M for destination row and col
            return cv.get_M(src=self.gcps["src"], dst=dst_colrow_a)

    def set_corners(self, corners):
        assert (np.array(corners).shape == (4,
                                            2)), f"a list of lists of 4 coordinates must be given, resulting in (4, 2) shape. Current shape is {corners.shape}"
        self.corners = corners

    def set_lens_pars(self, k1=0, c=2, f=4):
        """
        Set the lens parameters of the given CameraConfig

        :param k1: float, lens curvature [-], zero (default) means no curvature
        :param c: float, optical centre [1/n], where n is the fraction of the lens diameter, 2.0 (default) means in the
            centre.
        :param f: float, focal length [mm], typical values could be 2.8, or 4 (default).
        """
        assert (isinstance(k1, (int, float))), "k1 must be a float"
        assert (isinstance(c, (int, float))), "k1 must be a float"
        assert (isinstance(f, (int, float))), "k1 must be a float"
        self.lens_pars = {
            "k1": k1,
            "c": c,
            "f": f
        }

    def set_gcps(self, src, dst, z_0, h_ref=None, crs=None):
        """
        Set ground control points for the given CameraConfig

        :param src: list of lists, containing 4 x, y pairs of columns and rows in the frames of the original video
        :param dst: list of lists, containing 4 x, y pairs of real world coordinates in the given coordinate reference
            system.
        :param z_0: same as `h_ref` but then as measured by a global reference system such as a geoid or ellipsoid used
            by a GPS device. All other surveyed points (lens position and cross section) must have the same vertical
            reference.
        :param h_ref: float, optional. Water level, belonging to the 4 control points in `dst`. This is the water level
            as measured by a local reference (e.g. gauge plate) during the surveying of the control points. Control
            points must be taken on the water surface. If a single movie is processed, h_ref can be left out.
        :param crs: coordinate reference system, used to measure the control points (e.g. 4326 for WGS84 lat-lon).
            the destination control points will automatically be reprojected to the local crs of the CameraConfig.
        """
        assert (isinstance(src, list)), f"src must be a list of 4 numbers"
        assert (isinstance(dst, list)), f"dst must be a list of 4 numbers"
        assert (len(src) == 4), f"4 source points are expected in src, but {len(src)} were found"
        assert (len(dst) == 4), f"4 destination points are expected in dst, but {len(dst)} were found"
        if h_ref is not None:
            assert (isinstance(h_ref, (float, int))), "h_ref must contain a float number"
        assert (isinstance(z_0, (float, int))), "z_0 must contain a float number"
        assert (all(isinstance(x, (float, int)) for p in src for x in p)), "src contains non-int parts"
        assert (all(isinstance(x, (float, int)) for p in dst for x in p)), "dst contains non-float parts"
        if crs is not None:
            if not (hasattr(self, "crs")):
                raise ValueError(
                    'CameraConfig does not contain a crs, so gcps also cannot contain a crs. Ensure that the provided destination coordinates are in a locally defined coordinate reference system, e.g. established with a spirit level.')
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

    def plot(self, figsize=(13, 8), ax=None, tiles=None, buffer=0.0005, zoom_level=19, tiles_kwargs={}):
        """
        Plot the geographical situation of the CameraConfig. This is very useful to check if the CameraConfig seems
        to be in the right location. Requires cartopy to be installed.

        :param figsize: tuple, optional, width and height of figure
        :param ax: axes, optional, if not provided, axes is setup
        :param tiles: str, optional, name of tiler service to use (called as attribute from cartopy.io.img_tiles)
        :param buffer: float, optional buffer in lat-lon around points, used to set extent (default: 0.0005)
        :param zoom_level: int, optional, zoom level of image tiler service (default: 18)
        :param tiles_kwargs: dict, optional, keyword arguments to pass to ax,add_image when tiles are added
        :return: ax
        """
        # define plot kwargs
        from shapely.geometry import LineString, Point
        from shapely import ops
        if not (hasattr(self, "gcps")):
            raise ValueError("No GCPs found yet, please populate the gcps attribute with set_gcps first.")
        # prepare points for plotting
        points = [Point(x, y) for x, y in self.gcps["dst"]]
        if hasattr(self, "corners"):
            bbox = shapely.wkt.loads(self.bbox)
        if hasattr(self, "lens_position"):
            points.append(Point(self.lens_position[0], self.lens_position[1]))
        # transform points in case a crs is provided
        if hasattr(self, "crs"):
            # make a transformer to lat lon
            transform = Transformer.from_crs(CRS.from_user_input(self.crs), CRS.from_epsg(4326), always_xy=True).transform
            points = [ops.transform(transform, p) for p in points]
            if hasattr(self, "corners"):
                bbox = ops.transform(transform, bbox)
        xmin, ymin, xmax, ymax = list(np.array(LineString(points).bounds))
        extent = [xmin - buffer, xmax + buffer, ymin - buffer, ymax + buffer]
        x = [p.x for p in points]
        y = [p.y for p in points]

        if ax is None:
            f = plt.figure(figsize=figsize)
            if hasattr(self, "crs"):
                try:
                    import cartopy
                    import cartopy.io.img_tiles as cimgt
                    import cartopy.crs as ccrs
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        'Geographic plotting requires cartopy. Please install it with "conda install cartopy" and try again.')
                if tiles is not None:
                    tiler = getattr(cimgt, tiles)(**tiles_kwargs)
                    crs = tiler.crs
                else:
                    crs = ccrs.PlateCarree()
                # make point collection
                ax = plt.subplot(projection=crs)
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                if tiles is not None:
                    ax.add_image(tiler, zoom_level, zorder=1)
        if hasattr(ax, "add_geometries"):
            plot_kwargs = dict(transform= ccrs.PlateCarree())
        else:
            plot_kwargs = {}
        ax.plot(x[0:4], y[0:4], ".", label="Control points", markersize=12, markeredgecolor="w", zorder=2, **plot_kwargs)
        if hasattr(self, "lens_position"):
            ax.plot(x[-1], y[-1], ".", label="Lens position", markersize=12, zorder=2, markeredgecolor="w", **plot_kwargs)
        if hasattr(self, "corners"):
            patch = PolygonPatch(bbox, alpha=0.5, zorder=2, edgecolor="w", label="Area of interest", **plot_kwargs)
            ax.add_patch(patch)
        ax.legend()
        return ax

    def to_dict(self):
        """
        Return the CameraConfig object as dictionary

        :return: dict, containing CameraConfig components
        """

        d = self.__dict__
        # replace underscore keys for keys without underscore
        for k in list(d.keys()):
            if k[0] == "_":
                d[k[1:]] = d.pop(k)

        return d

    def to_file(self, fn):
        """
        Write the CameraConfig object to json structure

        :return: None
        """

        with open(fn, "w") as f:
            f.write(self.to_json())

    def to_json(self):
        """
        Convert CameraConfig object to string

        :return: json string with CameraConfig components
        """
        return json.dumps(self, default=lambda o: o.to_dict(), indent=4)


def get_camera_config(s):
    """
    Read camera config from string
    :param s: json string containing camera config
    :return: CameraConfig object
    """
    d = json.loads(s)
    return CameraConfig(**d)


def load_camera_config(fn):
    """
    Load a CameraConfig from a geojson file.

    :param fn: str, filename with camera config.
    :return: CameraConfig object
    """
    with open(fn, "r") as f:
        camera_config = get_camera_config(f.read())
    return camera_config
