import copy
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely import ops, wkt
from shapely.geometry import Polygon, LineString, Point

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
        return str(self.to_json())

    def __repr__(self):
        return self.to_json()

    def __init__(
            self,
            height,
            width,
            crs=None,
            window_size=10,
            resolution=0.05,
            bbox=None,
            camera_matrix=None,
            dist_coeffs=None,
            lens_position=None,
            corners=None,
            gcps=None,
            lens_pars=None,
            calibration_video=None

    ):
        """

        Parameters
        ----------
        height : int
            height of frame in pixels
        width : int
            width of frame in pixels
        crs : int, dict or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str) proj (str or dict) or wkt (str). Only used if
            the data has no native CRS.
        window_size : int
            pixel size of interrogation window (default: 15)
        resolution : float, optional
            resolution in m. of projected pixels (default: 0.01)
        bbox : shapely.geometry.Polygon, optional
            bounding box in geographical coordinates
        lens_position : list of floats (3),
            x, y, z coordinate of lens position in CRS
        corners : list of lists of floats (2)
            [x, y] coordinates defining corners of area of interest in camera cols/rows, bbox will be computed from this
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
        calibration_video : str, optional
            local path to video file containing a checkerboard pattern. Must be 9x6 if called directly, otherwise use
            ``.calibrate_camera`` explicitly and provide ``chessboard_size`` explicitly. When used, an automated camera
            calibration on the video file will be attempted.
        """
        assert(isinstance(height, int)), 'height must be provided as type "int"'
        assert(isinstance(width, int)), 'width must be provided as type "int"'
        assert (isinstance(window_size, int)), 'window_size must be of type "int"'
        self.height = height
        self.width = width
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
        else:
            self.set_lens_pars()
        if calibration_video is not None:
            self.set_lens_calibration(calibration_video, plot=False)
        # camera matrix and dist coeffs can also be set hard, this overrules the lens_pars option
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        if bbox is not None:
            self.bbox = bbox
        if window_size is not None:
            self.window_size = window_size
        # override the transform and bbox with the set corners
        if corners is not None:
            self.set_bbox_from_corners(corners)

    @property
    def bbox(self):
        """
        Returns geographical bbox fitting around corners points of area of interest in camera perspective

        Returns
        -------
        bbox : shapely.geometry.Polygon
            bbox of area of interest

        """
        return self._bbox

    @bbox.setter
    def bbox(self, pol):
        self._bbox = pol

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, camera_matrix):
        self._camera_matrix = camera_matrix.tolist() if isinstance(camera_matrix, np.ndarray) else camera_matrix

    @property
    def dist_coeffs(self):
        return self._dist_coeffs

    @dist_coeffs.setter
    def dist_coeffs(self, dist_coeffs):
        self._dist_coeffs = dist_coeffs.tolist() if isinstance(dist_coeffs, np.ndarray) else dist_coeffs

    @property
    def gcp_reduced(self):
        """
        Get the location of gcp destination points, reduced with their mean for better local projection

        Returns
        -------
        gcp_reduced : np.ndarray
            Reduced coordinate (x, y) or (x, y, z) of gcp destination points
        """
        return np.array(self.gcps["dst"]) - self.gcp_mean

    @property
    def gcp_mean(self):
        """
        Get the mean location of gcp destination points

        Returns
        -------
        gcp_mean : np.ndarray
            mean coordinate (x, y) or (x, y, z) of gcp destination points
        """
        return np.array(self.gcps["dst"]).mean(axis=0)

    @property
    def gcp_dims(self):
        """

        Returns
        -------
        dims : int
            amount of dimensions of gcps (can be 2 or 3)

        """
        return len(self.gcps["dst"][0])

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
            self.bbox,
            resolution=self.resolution,
            round=1
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
        return cv._get_transform(self.bbox, resolution=self.resolution)

    def set_lens_calibration(
        self,
        fn,
        chessboard_size=(9, 6),
        max_imgs=30,
        plot=True,
        progress_bar=True,
        **kwargs
    ):
        """
        Calibrates and sets the properties ``camera_matrix`` and ``dist_coeffs`` using a video of a chessboard pattern.
        Follows methods described on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        Parameters
        ----------
        fn : str
            path to video file
        df : int, optional
            amount of frames to skip after a valid frame with corner points is found, defaults to fps of video.
        chessboard_size : tuple, optional
            amount of internal corner points expected on chessboard pattern, default is (9, 6).
        max_imgs : int, optional
            maximum amount of images to use for calibration (default: 30).
        tolerance : float, optional
            error tolerance alowed for reprojection of corner points (default: 0.1, if set to None, no filtering will
            be done). images that exceed the tolerance are excluded from calibration. This is to remove images with
            spuriously defined points, or blurry images.
        plot : bool, optional
            if set, chosen frames will be plotted for the user to inspect on-=the-fly with a one-second delay
            (default: True).
        progress_bar : bool, optional
            if set, a progress bar going through the frames is plotted (default: True).


        """
        assert(os.path.isfile(fn)), f"Video calibration file {fn} not found"
        camera_matrix, dist_coeffs = cv.calibrate_camera(
            fn,
            chessboard_size,
            max_imgs,
            plot,
            progress_bar,
            **kwargs
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def get_bbox(self, camera=False, h_a=None, redistort=False):
        """

        Parameters
        ----------
        camera : bool, optional
            If set, the bounding box will be returned as row and column coordinates in the camera perspective.
            In this case ``h_a`` may be set to provide the right water level, to estimate the bounding box for.
        h_a : float, optional
            If set with ``camera=True``, then the bbox coordinates will be transformed to the camera perspective,
            using h_a as a present water level. In case a video with higher (lower) water levels is used, this
            will result in a different perspective plane than the control video.
        redistort : bool, optional
            If set in combination with ``camera``, the bbox will be redistorted in the camera objective using the
            distortion coefficients and camera matrix. Not used in orthorectification because this occurs by default
            on already undistorted images.

        Returns
        -------
        A bounding box, that in the used CRS is perfectly rectangular, and aligned in the up/downstream direction.
        It can (and certainly will) be rotated with respect to a typical bbox with xlim, ylim coordinates.
        If user sets ``camera=True`` then the geographical bounding box will be converted into a camera perspective,
        using the homography belonging to the available ground control points and current water level.

        This can then be used to reconstruct the grid for velocimetry calculations.
        """
        bbox = self.bbox
        if camera:
            coords = np.array(bbox.exterior.coords)
            # convert to perspective rowcol coordinates
            M = self.get_M(reverse=True, h_a=h_a)
            # reduce coords by control point mean
            coords -= self.gcp_mean[0:2]
            # TODO: re-distort if needed
            corners = cv2.perspectiveTransform(np.float32([coords]), M)[0]
            if redistort:
                # for visualization on still distorted frames this can be done. DO NOT do this if used for
                # orthorectification, as this typically occurs on undistorted images.
                corners = cv.undistort_points(corners, self.camera_matrix, self.dist_coeffs, reverse=True)

            bbox = Polygon(corners)
        return bbox

    def get_depth(self, z, h_a=None):
        """
        Retrieve depth for measured bathymetry points using the camera configuration and an actual water level,
        measured in local reference (e.g. staff gauge).

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
            h_a = self.gcps["h_ref"]
        z_pressure = np.maximum(self.gcps["z_0"] - self.gcps["h_ref"] + h_a, z)
        return z_pressure - z

    def get_dst_a(self, h_a=None):
        """
        h_a : float, optional
            actual water level measured [m], if not set, assumption is that a single video
            is processed and thus changes in water level are not relevant. (default: None)

        Returns
        -------
        Actual locations of control points (in case these are only x, y) given the current set water level and
        the camera location
        """
        # map where the destination points are with the actual water level h_a.
        if h_a is None or h_a == self.gcps["h_ref"]:
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
        return dst_a

    def get_dist_shore(self, x, y, z, h_a=None):
        """
        Retrieve depth for measured bathymetry points using the camera configuration and an actual water level, measured
        in local reference (e.g. staff gauge).

        Parameters
        ----------
        x : list of floats
            measured bathymetry point x-coordinates
        y : list of floats
            measured bathymetry point y-coordinates
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
            assert(self.gcps["h_ref"] is None), "No actual water level is provided, but a reference water level is " \
                                                "provided "
            # h_a = 0.
            # h_ref = 0.
        # else:
            # h_ref = self.gcps["h_ref"]
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

    def get_M(self, h_a=None, to_bbox_grid=False, reverse=False):
        """Establish a transformation matrix for a certain actual water level `h_a`. This is done by mapping where the
        ground control points, measured at `h_ref` will end up with new water level `h_a`, given the lens position.

        Parameters
        ----------
        h_a : float, optional
            actual water level [m] (Default: None)
        to_bbox_grid : bool, optional
            if set, the M will be computed in row, column values of the target bbox, with set resolution
        reverse : bool, optional
            if True, the reverse matrix is prepared, which can be used to transform projected
            coordinates back to the original camera perspective. (Default: False)

        Returns
        -------
        M : np.ndarray
            2x3 transformation matrix

        """
        src = cv.undistort_points(self.gcps["src"], self.camera_matrix, self.dist_coeffs)
        if to_bbox_grid:
            # lookup where the destination points are in row/column space
            # dst_a is the destination point locations position with the actual water level
            dst_a = cv.transform_to_bbox(
                self.get_dst_a(h_a),
                self.bbox,
                self.resolution
            )
            dst_a = np.array(dst_a)
        else:
            # in case we are dealing with a 2D 4-point, then reproject points on water surface, else keep 3D points
            dst_a = self.get_dst_a(h_a) if self.gcp_dims == 2 else self.gcps["dst"]
            # reduce dst_a with its mean to get much more accurate projection result in case x and y order of
            # magnitude is very large
            dst_a -= self.gcp_mean
        # src_a = self.get_src(**lens_pars)
        if dst_a.shape[-1] == 3:
            # compute the water level in the coordinate system reduced with the mean gcp coordinate
            z_a = self.get_z_a(h_a)
            z_a -= self.gcp_mean[-1]
            # treating 3D homography
            return cv.get_M_3D(
                src=src,
                dst=dst_a,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                z=z_a,
                reverse=reverse
            )
        else:
            return cv.get_M_2D(
                src=src,
                dst=dst_a,
                reverse=reverse
            )

    def get_z_a(self, h_a=None):
        """
        h_a : float, optional
            actual water level measured [m], if not set, assumption is that a single video
            is processed and thus changes in water level are not relevant. (default: None)

        Returns
        -------
        Actual locations of control points (in case these are only x, y) given the current set water level and
        the camera location
        """
        if h_a is None:
            return self.gcps["z_0"]
        else:
            return self.gcps["z_0"] + (h_a - self.gcps["h_ref"])

    def set_bbox_from_corners(self, corners):
        """
        Establish bbox based on a set of camera perspective corner points Assign corner coordinates to camera
        configuration

        Parameters
        ----------
        corners : list of lists (4)
            [columns, row] coordinates in camera perspective

        """
        assert (np.array(corners).shape == (4,
                                            2)), f"a list of lists of 4 coordinates must be given, resulting in (4, " \
                                                 f"2) shape. Current shape is {corners.shape} "

        # get homography, this is the only place besides self.get_M where cv.get_M is used.
        M_gcp = self.get_M()
        # TODO: make derivation dependent on 3D or 2D point availability
        # if self.gcps["src"].shape == 3:
        #     # TODO: homography from solvepnp
        bbox = cv.get_aoi(M_gcp, corners, resolution=self.resolution)
        # bbox is offset by self.gcp_mean. Regenerate bbox after adding offset
        bbox_xy = np.array(bbox.exterior.coords)
        bbox_xy += self.gcp_mean[0:2]
        bbox = Polygon(bbox_xy)
        self.bbox = bbox

    def set_lens_pars(self, k1=0, c=2, f=4):
        """Set the lens parameters of the given CameraConfig

        Parameters
        ----------
        k1 : float, optional
            lens curvature [-], zero (default) means no curvature
        c : float, optional
            optical centre [1/n], where n is the fraction of the lens diameter, 2.0 (default) means in the
            centre.
        f : float, optional
            focal length [mm], typical values could be 2.8, or 4 (default).


        """
        assert (isinstance(k1, (int, float))), "k1 must be a float"
        assert (isinstance(c, (int, float))), "k1 must be a float"
        assert (isinstance(f, (int, float))), "k1 must be a float"
        self.dist_coeffs = cv._get_dist_coefs(k1)
        self.camera_matrix = cv._get_cam_mtx(self.height, self.width, c=c, f=f)

    def set_gcps(self, src, dst, z_0, h_ref=None, crs=None):
        """
        Set ground control points for the given CameraConfig

        Parameters
        ----------
        src : list of lists (4 or 6+)
            [x, y] pairs of columns and rows in the frames of the original video
        dst : list of lists (4 or 6+)
            [x, y] or [x, y, z] pairs of real world coordinates in the given coordinate reference system.
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
        assert (isinstance(src, list)), f"src must be a list of (x, y) or (x, y, z) coordinates"
        assert (isinstance(dst, list)), f"dst must be a list of (x, y) or (x, y, z) coordinates"
        if np.array(dst).shape[1] == 2:
            assert (len(src) == 4), f"4 source points are expected in src, but {len(src)} were found"
            assert (len(dst) == 4), f"4 destination points are expected in dst, but {len(dst)} were found"
        else:
            assert(len(src) == len(dst)), f"Amount of (x, y, z) coordinates in src ({len(src)}) and dst ({len(dst)} must be equal"
            assert(len(src) >= 6), f"for (x, y, z) points, at least 6 pairs must be available, only {len(src)} provided"
        if h_ref is not None:
            assert (isinstance(h_ref, (float, int))), "h_ref must contain a float number"
        if z_0 is not None:
            assert (isinstance(z_0, (float, int))), "z_0 must be provided as type float"
        assert (all(isinstance(x, (float, int)) for p in src for x in p)), "src contains non-int parts"
        assert (all(isinstance(x, (float, int)) for p in dst for x in p)), "dst contains non-float parts"
        if crs is not None:
            if not (hasattr(self, "crs")):
                raise ValueError(
                    'CameraConfig does not contain a crs, so gcps also cannot contain a crs. Ensure that the provided '
                    'destination coordinates are in a locally defined coordinate reference system, e.g. established '
                    'with a spirit level.')
            dst = helpers.xyz_transform(dst, crs, CRS.from_wkt(self.crs))
        # if there is no h_ref, then no local gauge system, so set h_ref to zero
        if h_ref is None:
            h_ref = 0.
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
            x, y = helpers.xyz_transform([[x, y]], crs, self.crs)[0]
        self.lens_position = [x, y, z]

    def plot(
            self,
            figsize=(13, 8),
            ax=None,
            tiles=None,
            buffer=0.0005,
            zoom_level=19,
            camera=False,
            tiles_kwargs={}
    ):
        """
        Plot the geographical situation of the CameraConfig. This is very useful to check if the CameraConfig seems
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
        camera : bool, optional
            If set to True, all camera config information will be back projected to the original camera objective.
        **tiles_kwargs
            additional keyword arguments to pass to ax.add_image when tiles are added
        8) :
            

        Returns
        -------
        ax : plt.axes

        """
        # initiate transform
        transformer = None
        # if there is an axes, get the extent
        xlim = ax.get_xlim() if ax is not None else None
        ylim = ax.get_ylim() if ax is not None else None

        # prepare points for plotting
        if camera:
            points = [Point(x, y) for x, y in self.gcps["src"]]
        else:
            points = [Point(p[0], p[1]) for p in self.gcps["dst"]]

        if not camera:
            if hasattr(self, "lens_position") and not camera:
                points.append(Point(self.lens_position[0], self.lens_position[1]))
            # transform points in case a crs is provided
            if hasattr(self, "crs"):
                # make a transformer to lat lon
                transformer = Transformer.from_crs(
                    CRS.from_user_input(self.crs),
                    CRS.from_epsg(4326),
                    always_xy=True).transform
                points = [ops.transform(transformer, p) for p in points]
            xmin, ymin, xmax, ymax = list(np.array(LineString(points).bounds))
            extent = [xmin - buffer, xmax + buffer, ymin - buffer, ymax + buffer]
        x = [p.x for p in points]
        y = [p.y for p in points]

        if ax is None:
            f = plt.figure(figsize=figsize)
            if (hasattr(self, "crs") and not(camera)):
                ax = helpers.get_geo_axes(tiles=tiles, extent=extent, zoom_level=zoom_level, **tiles_kwargs)
            else:
                ax = plt.subplot()
        if hasattr(ax, "add_geometries"):
            import cartopy.crs as ccrs
            plot_kwargs = dict(transform=ccrs.PlateCarree())
        else:
            plot_kwargs = {}
        ax.plot(
            x[0:len(self.gcps["dst"])],
            y[0:len(self.gcps["dst"])],
            ".",
            label="Control points",
            markersize=12,
            markeredgecolor="w",
            zorder=2,
            **plot_kwargs
        )
        if len(x) > len(self.gcps["dst"]):
            ax.plot(
                x[-1],
                y[-1],
                ".",
                label="Lens position",
                markersize=12,
                zorder=2,
                markeredgecolor="w",
                **plot_kwargs
            )
        patch_kwargs = {
            **plot_kwargs,
            "alpha": 0.5,
            "zorder": 2,
            "edgecolor": "w",
            "label": "Area of interest",
            **plot_kwargs
        }
        if hasattr(self, "bbox"):
            self.plot_bbox(ax=ax, camera=camera, transformer=transformer, **patch_kwargs)
        if camera:
            # make sure that zero is on the top
            ax.set_aspect("equal")
            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        ax.legend()
        return ax

    def plot_bbox(self, ax=None, camera=False, transformer=None, h_a=None, **kwargs):
        """
        Plot bounding box for orthorectification in a geographical projection (``camera=False``) or the camera
        Field Of View (``camera=True``).

        Parameters
        ----------
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None)
        camera : bool, optional
            If set to True, all camera config information will be back projected to the original camera objective.
        transformer : pyproj transformer transformation function, optional
            used to reproject bbox to axes object projection (e.g. lat lon)
        h_a : float, optional
            If set with ``camera=True``, then the bbox coordinates will be transformed to the camera perspective,
            using h_a as a present water level. In case a video with higher (lower) water levels is used, this
            will result in a different perspective plane than the control video.

        Returns
        -------
        p : matplotlib.patch mappable
        """
        # collect information to plot
        bbox = self.get_bbox(camera=camera, h_a=h_a)
        if camera is False and transformer is not None:
            # geographical projection is needed
            bbox = ops.transform(transformer, bbox)

        bbox_x, bbox_y = bbox.exterior.xy
        bbox_coords = list(zip(bbox_x, bbox_y))
        patch = patches.Polygon(
            bbox_coords,
            **kwargs
        )
        p = ax.add_patch(patch)
        return p

    def to_dict(self):
        """Return the CameraConfig object as dictionary

        Returns
        -------
        camera_config_dict : dict
            serialized CameraConfig

        """

        d = copy.deepcopy(self.__dict__)
        # replace underscore keys for keys without underscore
        for k in list(d.keys()):
            if k[0] == "_":
                d[k[1:]] = d.pop(k)

        return d

    def to_dict_str(self):
        d = self.to_dict()
        # convert anything that is not string in string
        dict_str = {k: v if not(isinstance(v, Polygon)) else v.__str__() for k, v in d.items()}
        return dict_str

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
        return json.dumps(self, default=lambda o: o.to_dict_str(), indent=4)


depr_warning_height_width = """
Your camera configuration does not have a property "height" and/or "width", probably because your configuration file is 
from an older < 0.3.0 version. Please rectify this by editing your .json config file. The top of your file should e.g.
look as follows for a HD video:
{
    "height": 1080,
    "width": 1920,
    "crs": ....
    ...
}
"""


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
    if not "height" in d or not "width" in d:
        raise IOError(depr_warning_height_width)
    # ensure the bbox is a Polygon object
    if "bbox" in d:
        if isinstance(d["bbox"], str):
            d["bbox"] = wkt.loads(d["bbox"])
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
