import json
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt

from matplotlib import patches
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from .. import cv, helpers


class CameraConfig:
    """
    Camera configuration containing information about the perspective of the camera with respect to real world
    coordinates
    """
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

        Parameters
        ----------
        crs : int, dict or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str) proj (str or dict) or wkt (str). Only used if
            the data has no native CRS.
        window_size : int
            pixel size of interrogation window (default: 15)
        resolution : float
            resolution in m. of projected pixels
        lens_position : list of floats (3),
            x, y, z coordinate of lens position in CRS
        corners : list of lists of floats (2)
            [x, y] coordinates defining corners of area of interest in camera cols/rows
        gcps : dict
            Can contain "src": list of 4 lists, with column, row locations in objective of control points,
            "dst": list of 4 lists, with x, y locations (local or global coordinate reference system) of control points,
            "h_ref": float, measured water level [m] in local reference system (e.g. from staff gauge or pressure gauge)
            during gcp survey,
            "z_0": float, water level [m] in global reference system (e.g. from used GPS system CRS). This must be in
            the same vertical reference as the measured bathymetry and other survey points,
            "crs": int, str or CRS object, CRS in which "dst" points are measured. If None, a local coordinate system is
            assumed (e.g. from spirit level).
        lens_pars : dict, optional
            Lens parameters, containing: "k1": float, barrel lens distortion parameter (default: 0.),
            "c": float, optical center (default: 2.),
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
        """
        Returns geographical bbox fitting around corners points of area of interest in camera perspective

        Returns
        -------
        bbox : shapely.geometry.Polygon
            bbox of area of interest

        """
        assert (hasattr(self, "corners")), "CameraConfig object has no corners, set these with CameraConfig.set_corners"
        return cv.get_aoi(self.gcps["src"], self.gcps["dst"], self.corners).__str__()

    @property
    def shape(self):
        """
        Returns rows and columns in projected frames from ``Frames.project``

        Returns
        -------
        rows : int
            Amount of rows in projected frame
        cols : int
            Amount of columns in projected frame
        """
        cols, rows = cv._get_shape(
            shapely.wkt.loads(self.bbox),
            resolution=self.resolution,
            round=10
        )
        return rows, cols

    @property
    def transform(self):
        """
        Returns Affine transform of projected frames from ``Frames.project``

        Returns
        -------
        transform : rasterio.transform.Affine object

        """
        bbox = shapely.wkt.loads(self.bbox)
        return cv._get_transform(bbox, resolution=self.resolution)

    def get_depth(self, z, h_a=None):
        """Retrieve depth for measured bathymetry points using the camera configuration and an actual water level, measured
        in local reference (e.g. staff gauge).

        Parameters
        ----------
        z : list of floats
            measured bathymetry point depths
        h_a : float, optional
            actual water level measured [m], if not set, assumption is that a single video
            is processed and thus changes in water level are not relevant. (default: None)

        Returns
        -------
        depths : list of floats

        """
        if h_a is None:
            assert(self.gcps["h_ref"] is None), "No actual water level is provided, but a reference water level is provided"
            h_a = 0.
            h_ref = 0.
        else:
            h_ref = self.gcps["h_ref"]
        z_pressure = np.maximum(self.gcps["z_0"] - h_ref + h_a, z)
        return z_pressure - z


    def get_dist_shore(self, x, y, z, h_a=None):
        """Retrieve depth for measured bathymetry points using the camera configuration and an actual water level, measured
        in local reference (e.g. staff gauge).

        Parameters
        ----------
        z : list of floats
            measured bathymetry point depths
        h_a : float, optional
            actual water level measured [m], if not set, assumption is that a single video
            is processed and thus changes in water level are not relevant. (default: None)

        Returns
        -------
        depths : list of floats

        """
        # retrieve depth
        depth = self.get_depth(z, h_a=h_a)
        if h_a is None:
            assert(self.gcps["h_ref"] is None), "No actual water level is provided, but a reference water level is provided"
            h_a = 0.
            h_ref = 0.
        else:
            h_ref = self.gcps["h_ref"]
        z_dry = depth <= 0
        z_dry[[0, -1]] = True
        # compute distance to nearest dry points with Pythagoras
        dist_shore = np.array([(((x[z_dry] - _x) ** 2 + (y[z_dry] - _y) ** 2) ** 0.5).min() for _x, _y, in zip(x, y)])
        return dist_shore


    def get_dist_wall(self, x, y, z, h_a=None):
        depth = self.get_depth(z, h_a=h_a)
        dist_shore = self.get_dist_shore(x, y, z, h_a=h_a)
        dist_wall = (dist_shore**2 + depth**2)**0.5
        return dist_wall

    def z_to_h(self, z):
        """Convert z coordinates of bathymetry to height coordinates in local reference (e.g. staff gauge)

        Parameters
        ----------
        z : float
            measured bathymetry point

        Returns
        -------
        h : float
        """
        h_ref = 0 if self.gcps["h_ref"] is None else self.gcps["h_ref"]
        h = z + h_ref - self.gcps["z_0"]
        return h


    def get_M(self, h_a=None, reverse=False):
        """Establish a transformation matrix for a certain actual water level `h_a`. This is done by mapping where the
        ground control points, measured at `h_ref` will end up with new water level `h_a`, given the lens position.

        Parameters
        ----------
        h_a : float, optional
            actual water level [m] (Default: None)
        reverse : bool, optional
            if True, the reverse matrix is prepared, which can be used to transform projected
            coordinates back to the original camera perspective. (Default: False)

        Returns
        -------
        M : np.ndarray
            2x3 transformation matrix

        """
        # map where the destination points are with the actual water level h_a.
        if h_a is None or self.gcps["h_ref"] is None:
            # fill in the same value for h_ref and h_a
            dst_a = self.gcps["dst"]
        else:
            h_ref = self.gcps["h_ref"]
            lens_position = self.lens_position
            dst_a = cv._get_gcps_a(
                lens_position,
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
        """
        Assign corner coordinates to camera configuration

        Parameters
        ----------
        corners : list of lists (4)
            [columns, row] coordinates in camera perspective

        """
        assert (np.array(corners).shape == (4,
                                            2)), f"a list of lists of 4 coordinates must be given, resulting in (4, 2) shape. Current shape is {corners.shape}"
        self.corners = corners

    def set_lens_pars(self, k1=0, c=2, f=4):
        """Set the lens parameters of the given CameraConfig

        Parameters
        ----------
        k1 : float
            lens curvature [-], zero (default) means no curvature
        c : float
            optical centre [1/n], where n is the fraction of the lens diameter, 2.0 (default) means in the
            centre.
        f : float, optional
            focal length [mm], typical values could be 2.8, or 4 (default).


        """
        assert (isinstance(k1, (int, float))), "k1 must be a float"
        assert (isinstance(c, (int, float))), "k1 must be a float"
        assert (isinstance(f, (int, float))), "k1 must be a float"
        self.lens_pars = {
            "k1": k1,
            "c": c,
            "f": f
        }

    def set_gcps(self, src, dst, z_0=None, h_ref=None, crs=None):
        """Set ground control points for the given CameraConfig

        Parameters
        ----------
        src : list of lists (4)
            [x, y] pairs of columns and rows in the frames of the original video
        dst : list of lists (4)
            [x, y] pairs of real world coordinates in the given coordinate reference system.
        z_0 : float
            Water level measured in global reference system such as a geoid or ellipsoid used
            by a GPS device. All other surveyed points (lens position and cross section) must have the same vertical
            reference.
        h_ref :  float, optional
            Water level, belonging to the 4 control points in `dst`. This is the water level
            as measured by a local reference (e.g. gauge plate) during the surveying of the control points. Control
            points must be taken on the water surface. If a single movie is processed, h_ref can be left out.
            (Default: None)
        crs : int, dict or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str) proj (str or dict) or wkt (str). CRS used to
            measure the control points (e.g. 4326 for WGS84 lat-lon). Destination control points will automatically be
            reprojected to the local crs of the CameraConfig. (Default: None)

        """
        assert (isinstance(src, list)), f"src must be a list of 4 numbers"
        assert (isinstance(dst, list)), f"dst must be a list of 4 numbers"
        assert (len(src) == 4), f"4 source points are expected in src, but {len(src)} were found"
        assert (len(dst) == 4), f"4 destination points are expected in dst, but {len(dst)} were found"
        if h_ref is not None:
            assert (isinstance(h_ref, (float, int))), "h_ref must contain a float number"
        if z_0 is not None:
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
        """Set the geographical position of the lens of current CameraConfig.

        Parameters
        ----------
        x : float
            x-coordinate
        y : float
            y-coordinate
        z : float
            z-coordinate
        crs : int, dict or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str) proj (str or dict) or wkt (str). CRS used to
            measure the lens position (e.g. 4326 for WGS84 lat-lon). The position's x and y coordinates will
            automatically be reprojected to the local crs of the CameraConfig.
        """

        if crs is not None:
            if self.crs is None:
                raise ValueError("CameraConfig does not contain a crs, ")
            x, y = helpers.xy_transform(x, y, crs, self.crs)

        self.lens_position = [x, y, z]

    def plot(self, figsize=(13, 8), ax=None, tiles=None, buffer=0.0005, zoom_level=19, tiles_kwargs={}):
        """Plot the geographical situation of the CameraConfig. This is very useful to check if the CameraConfig seems
        to be in the right location. Requires cartopy to be installed.

        Parameters
        ----------
        figsize : tuple, optional
            width and height of figure (Default value = (13)
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None)
        tiles : str, optional
            name of tiler service to use (called as attribute from cartopy.io.img_tiles) (Default: None)
        buffer : float, optional
            buffer in lat-lon around points, used to set extent (default: 0.0005)
        zoom_level : int, optional
            zoom level of image tiler service (default: 18)
        **tiles_kwargs
            additional keyword arguments to pass to ax.add_image when tiles are added
        8) :
            

        Returns
        -------
        ax : plt.axes

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
            import cartopy.crs as ccrs
            plot_kwargs = dict(transform= ccrs.PlateCarree())
        else:
            plot_kwargs = {}
        ax.plot(x[0:4], y[0:4], ".", label="Control points", markersize=12, markeredgecolor="w", zorder=2, **plot_kwargs)
        if hasattr(self, "lens_position"):
            ax.plot(x[-1], y[-1], ".", label="Lens position", markersize=12, zorder=2, markeredgecolor="w", **plot_kwargs)
        if hasattr(self, "corners"):
            bbox_x, bbox_y = bbox.exterior.xy
            bbox_coords = list(zip(bbox_x, bbox_y))
            patch = patches.Polygon(bbox_coords, alpha=0.5, zorder=2, edgecolor="w", label="Area of interest", **plot_kwargs)
            ax.add_patch(patch)
        ax.legend()
        return ax

    def to_dict(self):
        """Return the CameraConfig object as dictionary

        Returns
        -------
        camera_config_dict : dict
            serialized CameraConfig

        """

        d = self.__dict__
        # replace underscore keys for keys without underscore
        for k in list(d.keys()):
            if k[0] == "_":
                d[k[1:]] = d.pop(k)

        return d

    def to_file(self, fn):
        """Write the CameraConfig object to json structure
        
        Parameters
        ----------
        fn : str
            Path to file to write camera config to

        """

        with open(fn, "w") as f:
            f.write(self.to_json())

    def to_json(self):
        """Convert CameraConfig object to string
        
        Returns
        -------
        json_str : str
            json string with CameraConfig components
        """
        return json.dumps(self, default=lambda o: o.to_dict(), indent=4)


def get_camera_config(s):
    """Read camera config from string

    Parameters
    ----------
    s : str
        json string containing camera config

    Returns
    -------
    cam_config : CameraConfig

    """
    d = json.loads(s)
    return CameraConfig(**d)


def load_camera_config(fn):
    """Load a CameraConfig from a geojson file.

    Parameters
    ----------
    fn : str
        path to file with camera config in json format.

    Returns
    -------
    cam_config : CameraConfig

    """
    with open(fn, "r") as f:
        camera_config = get_camera_config(f.read())
    return camera_config
