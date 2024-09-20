import copy
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shapely.geometry
from shapely import ops, wkt
from shapely.geometry import Polygon, LineString, Point

from matplotlib import patches
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

from typing import Any, Dict, List, Optional, Tuple, Union

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
            height: int,
            width: int,
            crs: Optional[Any] = None,
            window_size: int = 10,
            resolution: float = 0.05,
            bbox: Optional[Union[shapely.geometry.Polygon, str]] = None,
            camera_matrix: Optional[List[List[float]]] = None,
            dist_coeffs: Optional[List[List[float]]] = None,
            lens_position: Optional[List[float]] = None,
            corners: Optional[List[List[float]]] = None,
            gcps: Optional[Dict[str, Union[List, float]]] = None,
            lens_pars: Optional[Dict[str, float]] = None,
            calibration_video: Optional[str] = None,
            is_nadir: Optional[bool] = False,
            stabilize: Optional[List[List]] = None,
            rotation: Optional[int] = None,
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
            Can contain "src": list of lists, with column, row locations in objective of control points,
            "dst": list of lists, with x, y or x, y, z locations (local or global coordinate reference system) of
            control points,
            "h_ref": float, measured water level [m] in local reference system (e.g. from staff gauge or pressure gauge)
            during gcp survey,
            "z_0": float, water level [m] in global reference system (e.g. from used GPS system CRS). This must be in
            the same vertical reference as the measured bathymetry and other survey points,
            "crs": int, str or CRS object, CRS in which "dst" points are measured. If None, a local coordinate system is
            assumed (e.g. from spirit level).
        lens_pars (deprecated) : dict, optional
            Lens parameters, containing: "k1": float, barrel lens distortion parameter (default: 0.),
            "c": float, optical center (default: 2.),
            "focal_length": float, focal length (default: width of image frame)
        calibration_video : str, optional
            local path to video file containing a checkerboard pattern. Must be 9x6 if called directly, otherwise use
            ``.calibrate_camera`` explicitly and provide ``chessboard_size`` explicitly. When used, an automated camera
            calibration on the video file will be attempted.
        is_nadir : bool, optional
            If set, the video is assumed to be taken at sub-drone position and only two control points are needed
            for camera configuration
        stabilize : list (of lists), optional
            contains [x, y] pixels defining a polygon enclosing moving (water) areas. Areas outside of the polygon
            are used for stabilization of the video, if this polygon is defined.
        rotation : int [90, 180, 270]
            enforces a rotation of the video of 90, 180 or 2780 degrees clock-wise.
        """
        assert(isinstance(height, int)), 'height must be provided as type "int"'
        assert(isinstance(width, int)), 'width must be provided as type "int"'
        assert (isinstance(window_size, int)), 'window_size must be of type "int"'
        self.height = height
        self.width = width
        self.is_nadir = is_nadir
        if crs is not None:
            try:
                crs = CRS.from_user_input(crs)
            except CRSError:
                raise CRSError(f'crs "{crs}" is not a valid Coordinate Reference System')
            assert (crs.is_geographic == 0), "Prodstvided crs must be projected with units like [m]"
            self.crs = crs.to_wkt()
        if resolution is not None:
            self.resolution = resolution
        if lens_position is not None:
            self.set_lens_position(*lens_position)
        else:
            self.lens_position = None
        if gcps is not None:
            self.set_gcps(**gcps)
        if camera_matrix is None or dist_coeffs is None:
            if self.is_nadir:
                # with nadir, no perspective can be constructed, hence, camera matrix and dist coeffs will be set to default values
                self.camera_matrix = cv._get_cam_mtx(self.height, self.width)
                self.dist_coeffs = cv.DIST_COEFFS
            # camera pars are incomplete and need to be derived
            else:
                self.set_intrinsic(
                    camera_matrix=camera_matrix,
                    lens_pars=lens_pars
                )
        else:
            # camera matrix and dist coeffs can also be set hard, this overrules the lens_pars option
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
        if calibration_video is not None:
            self.set_lens_calibration(calibration_video, plot=False)
        if bbox is not None:
            self.bbox = bbox
        if window_size is not None:
            self.window_size = window_size
        # override the transform and bbox with the set corners
        if corners is not None:
            self.set_bbox_from_corners(corners)
        if stabilize is not None:
            self.stabilize = stabilize
        if rotation is not None:
            self.rotation = rotation

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
        if isinstance(pol, str):
            self._bbox = wkt.loads(pol)
        else:
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
    def gcps_dest(self):
        """

        Returns
        -------
        dst : np.ndarray
            destination coordinates of ground control point. z-coordinates are parsed from z_0 if necessary
        """
        if hasattr(self, "gcps"):
            if "dst" in self.gcps:
                return np.array(self.gcps["dst"] if len(self.gcps["dst"][0]) == 3 else np.c_[self.gcps["dst"], np.ones(4)*self.gcps["z_0"]])
        # if conditions are not yet met, then return None
        return None

    @property
    def gcps_dest_bbox(self):
        """

        Returns
        -------
        dst : np.ndarray
            Destination coordinates measured as column, row in the intended bounding box with the intended resolution
        """
        return np.array(cv.transform_to_bbox(self.gcps_dest, self.bbox, self.resolution))


    @property
    def gcps_bbox_reduced(self):
        """

        Returns
        -------
        dst : np.ndarray
            Destination coordinates in col, row in the intended bounding box, reduced with their mean coordinate

        """
        return self.gcps_dest_bbox - self.gcps_dest_bbox.mean(axis=0)
    @property
    def gcps_reduced(self):
        """
        Get the location of gcp destination points, reduced with their mean for better local projection

        Returns
        -------
        dst : np.ndarray
            Reduced coordinate (x, y) or (x, y, z) of gcp destination points
        """
        return np.array(self.gcps_dest - self.gcps_mean)

    @property
    def gcps_mean(self):
        """
        Get the mean location of gcp destination points

        Returns
        -------
        dst_mean : np.ndarray
            mean coordinate (x, y) or (x, y, z) of gcp destination points
        """
        return np.array(self.gcps_dest).mean(axis=0)

    @property
    def gcps_dims(self):
        """

        Returns
        -------
        dims : int
            amount of dimensions of gcps (can be 2 or 3)

        """
        return len(self.gcps["dst"][0]) if hasattr(self, "gcps") else None

    @property
    def is_nadir(self):
        """
        Returns if the camera configuration belongs to nadir video

        Returns
        -------
        is_nadir : bool
            False if not nadir, True if nadir

        """
        return self._is_nadir

    @is_nadir.setter
    def is_nadir(
            self,
            nadir_prop: bool
    ):
        self._is_nadir = nadir_prop

    @property
    def pnp(self):
        return cv.solvepnp(
            self.gcps_reduced,
            self.gcps["src"],
            self.camera_matrix,
            self.dist_coeffs
        )


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
    def stabilize(self):
        """
        Return stabilization polygon (anything outside is used for stabilization

        Returns
        -------
        coords : list of lists
            coordinates in original image frame comprising the polygon for use in stabilization
        """
        return self._stabilize

    @stabilize.setter
    def stabilize(
            self,
            coords: List[List[float]]
    ):
        self._stabilize = coords


    @property
    def rotation(self):
        """
        Return rotation OpenCV code

        Returns
        -------
        code : int
            integer code belonging to rotation, 0 for 90 deg, 1 for 180 deg and 2 for 270 deg
        """
        if hasattr(self, "_rotation"):
            return self._rotation
        else:
            return None

    @rotation.setter
    def rotation(
            self,
            rotation_code: int
    ):
        self._rotation = rotation_code


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
        fn: str,
        chessboard_size: Optional[Tuple] = (9, 6),
        max_imgs: Optional[int] = 30,
        plot: Optional[bool] = True,
        progress_bar: Optional[bool] = True,
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

    def estimate_lens_position(self):
        """estimate lens position from distortion and intrinsec/extrinsic matrix."""
        _, rvec, tvec = self.pnp
        rmat = cv2.Rodrigues(rvec)[0]
        # determine lens position related to center of objective
        lens_pos_centroid = (np.array(-rmat).T @ tvec).flatten()
        lens_pos = np.array(lens_pos_centroid) + self.gcps_mean
        return lens_pos

    def get_bbox(
            self,
            camera: Optional[bool] = False,
            h_a: Optional[float] = None,
            z_a: Optional[float] = None,
            within_image: Optional[bool] = False,
            expand_exterior=True,
    ) -> Polygon:
        """

        Parameters
        ----------
        camera : bool, optional
            If set, the bounding box will be returned as row and column coordinates in the camera perspective.
            In this case ``h_a`` may be set to provide the right water level, to estimate the bounding box for.
        h_a : float, optional
            If set with ``camera=True``, then the bbox coordinates will be transformed to the camera perspective,
            using h_a as a present water level (in local datum). In case a video with higher (lower) water levels is
            used, this will result in a different perspective plane than the control video.
        z_a : float, optional
            similar to setting h_a, but z_a represent the vertical coordinate in the coordinate reference system of
            the camera configuration instead of the local datum.
        within_image : bool, optional (default False)
            Set to True to make an attempt to remove parts of the polygon that lie outside of the image field of view
        expand_exterior : bool, optional
            Set to True to expand the corner points to more points. This is particularly useful for plotting purposes.

        Returns
        -------
        A bounding box, that in the used CRS is perfectly rectangular, and aligned in the up/downstream direction.
        It can (and certainly will) be rotated with respect to a typical bbox with xlim, ylim coordinates.
        If user sets ``camera=True`` then the geographical bounding box will be converted into a camera perspective,
        using the homography belonging to the available ground control points and current water level.

        This can then be used to reconstruct the grid for velocimetry calculations.
        """
        bbox = self.bbox
        coords = np.array(bbox.exterior.coords)
        if within_image:
            # in this case, always more points than just corners are needed, so expand_exterior is forced to True
            expand_exterior = True
        if expand_exterior:
            # make a new set of bbox coordinates with a higher density. This is meant to enable plotting of distortion on
            # image frame, and to plot partial coverage in the real-world coordinates
            coords_expand = np.zeros((0, 2))
            for n in range(0, len(coords)-1):
                new_coords = np.linspace(coords[n], coords[n + 1], 100)
                coords_expand = np.r_[coords_expand, new_coords]
            coords = coords_expand
        if not z_a:
            z_a = self.get_z_a(h_a)
        # add vertical coordinates to the set
        coords = np.c_[coords, np.ones(len(coords))*z_a]
        # project points to pixel image coordinates
        corners = self.project_points(coords, within_image=within_image)
        corners = corners[np.isfinite(corners[:, 0])]
        if not(camera):
            # project back to real-world coordinates after possibly cutting at edges of visibility
            corners = self.unproject_points(np.array(np.array(list(zip(*corners))).T), z_a)
            # corners = self.unproject_points(list(zip(coords[:, 0], coords[:, 1])), z_a)
            # project points (after cutting on image edges) to geographical space
        bbox = Polygon(corners[np.isfinite(corners[:, 0])][:, 0:2])

        return bbox

    def get_camera_coords(
            self,
            points: List[List],
    ):
        """
        Convert real-world coordinates into camera coordinates using
        Parameters
        ----------
        points : array-like list (of lists)
            [x, y, z] real-world coordinates

        Returns
        -------
        cam_points : np.ndarray (of points)
            [x, y, z] camera coordinates
        """
        _, rvec, tvec = self.pnp
        # get rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # convert points into array
        points = np.array(points)
        cam_points = np.einsum('ij,kj->ki', R, np.array(points)) + tvec.flatten()
        return cam_points

    def get_depth(
            self,
            z: List[float],
            h_a: Optional[float] = None
    ) -> List[float]:
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


    def get_dist_shore(
            self,
            x: List[float],
            y: List[float],
            z: List[float],
            h_a: Optional[float] = None
    ) -> List[float]:
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

    def get_dist_wall(
            self,
            x: List[float],
            y: List[float],
            z: List[float],
            h_a: Optional[float] = None
    ) -> List[float]:
        """
        Retrieve distance to wall for measured bathymetry points using the camera configuration and an actual water
        level, measured in local reference (e.g. staff gauge).

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
        distance : list of floats

        """
        depth = self.get_depth(z, h_a=h_a)
        dist_shore = self.get_dist_shore(x, y, z, h_a=h_a)
        dist_wall = (dist_shore**2 + depth**2)**0.5
        return dist_wall

    def z_to_h(
            self,
            z: float
    ) -> float:
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

    def get_M(
            self,
            h_a: Optional[float] = None,
            to_bbox_grid: Optional[bool] = False,
            reverse: Optional[bool] = False
    ) -> np.ndarray:
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
            dst_a = self.gcps_bbox_reduced
        else:
            dst_a = self.gcps_reduced
        # compute the water level in the coordinate system reduced with the mean gcp coordinate
        z_a = self.get_z_a(h_a)
        z_a -= self.gcps_mean[-1]
        # treating 3D homography
        return cv.get_M_3D(
            src=src,
            dst=dst_a,
            camera_matrix=self.camera_matrix,
            dist_coeffs=cv.DIST_COEFFS, # self.dist_coeffs,
            z=z_a,
            reverse=reverse
        )

    def get_z_a(
            self,
            h_a: Optional[float] = None
    ) -> float:
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

    def set_bbox_from_corners(
            self,
            corners: List[List[float]]
    ):
        """
        Establish bbox based on a set of camera perspective corner points Assign corner coordinates to camera
        configuration

        Parameters
        ----------
        corners : list of lists (4)
            [columns, row] coordinates in original camera perspective without any undistortion applied

        """
        assert (np.array(corners).shape == (4,
                                            2)), f"a list of lists of 4 coordinates must be given, resulting in (4, " \
                                                 f"2) shape. Current shape is {corners.shape} "

        # get homography
        corners_xyz = self.unproject_points(corners, np.ones(4)*self.gcps["z_0"])
        bbox = cv.get_aoi(
            corners_xyz,
            resolution=self.resolution
        )
        self.bbox = bbox

    def set_intrinsic(
            self,
            camera_matrix: Optional[List[List]] = None,
            dist_coeffs: Optional[List[List]] = None,
            lens_pars: Optional[Dict[str, float]] = None
    ):
        # first set a default estimate from pose if 3D gcps are available
        self.set_lens_pars()  # default parameters use width of frame
        if hasattr(self, "gcps"):
            if len(self.gcps["src"]) >= 4:
                self.camera_matrix, self.dist_coeffs, err = cv.optimize_intrinsic(
                    self.gcps["src"],
                    self.gcps_dest,
                    # self.gcps["dst"],
                    self.height,
                    self.width,
                    lens_position=self.lens_position
                )
            if lens_pars is not None:
                # override with lens parameter set by user
                self.set_lens_pars(**lens_pars)
            if camera_matrix is not None and dist_coeffs is not None:
                # override with
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs


    def set_lens_pars(
            self,
            k1: Optional[float] = 0.,
            c: Optional[float] = 2.,
            focal_length: Optional[float] = None
    ):
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
        assert (isinstance(c, (int, float))), "c must be a float"
        if focal_length is not None:
            assert (isinstance(focal_length, (int, float, None))), "f must be a float"
        self.dist_coeffs = cv._get_dist_coefs(k1)
        self.camera_matrix = cv._get_cam_mtx(self.height, self.width, c=c, focal_length=focal_length)

    def set_gcps(
            self,
            src: List[List],
            dst: List[List],
            z_0: float,
            h_ref: Optional[float] = None,
            crs: Optional[Any] = None
    ):
        """
        Set ground control points for the given CameraConfig

        Parameters
        ----------
        src : list of lists (2, 4 or 6+)
            [x, y] pairs of columns and rows in the frames of the original video
        dst : list of lists (2, 4 or 6+)
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
            assert (len(src) in [2, 4]), f"2 or 4 source points are expected in src, but {len(src)} were found"
            if len(src) == 4:
                assert (len(dst) == 4), f"4 destination points are expected in dst, but {len(dst)} were found"
            else:
                assert (len(dst) == 2), f"2 destination points are expected in dst, but {len(dst)} were found"
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
        # check if 2 points are available
        if len(src) == 2:
            self.is_nadir = True
            src, dst = cv._get_gcps_2_4(src, dst, self.width, self.height)
        if h_ref is None:
            h_ref = 0.
        self.gcps = {
            "src": src,
            "dst": dst,
            "h_ref": h_ref,
            "z_0": z_0,
        }

    def set_lens_position(
            self,
            x: float,
            y: float,
            z: float,
            crs: Optional[Any] = None
    ):
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

    def project_points(
            self,
            points: List[List],
            within_image=False
    ) -> np.ndarray:
        """
        Project real world x, y, z coordinates into col, row coordinates on image. If
        col, row coordinates are not allowed to go outside of the image frame, then set within_image = True

        Parameters
        ----------
        points : list of lists or array-like
            list of points [x, y, z] in real world coordinates

        Returns
        -------
        points_project : list or array-like
            list of points (equal in length as points) with [col, row] coordinates
        """
        _, rvec, tvec = self.pnp
        # normalize points wrt mean of gcps
        points = np.float32(np.array(points) - self.gcps_mean)
        points_proj, jacobian = cv2.projectPoints(
            points,
            rvec,
            tvec,
            np.array(self.camera_matrix),
            np.array(self.dist_coeffs)
        )
        points_proj = np.array([list(point[0]) for point in points_proj])

        # points_back = cv.unproject_points(src=points_proj, z=points[:, -1], )
        if within_image:
            # back project, the ones that do not get close to the original will be removed
            points_back = cv.unproject_points(
                points_proj,
                z=points[:, -1],
                rvec=rvec,
                tvec=tvec,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs
            )
            # TODO: figure out how to filter out points that bend into the image frame according to the
            # distortion parameters, but are in fact outside. This is not yet working the way it should
            filter = np.all(np.isclose(np.array(points_back), np.array(points), atol=1e-2), axis=1)
            points_proj[~filter] = np.nan
            # also filter points outside edges of image
            points_proj[points_proj[:, 0] < 0, 0] = 0.
            points_proj[points_proj[:, 0] > self.width - 1, 0] = self.width - 1
            points_proj[points_proj[:, 1] < 0, 1] = 0.
            points_proj[points_proj[:, 1] > self.height - 1, 1] = self.height - 1
        return points_proj


    def unproject_points(
            self,
            points: List[List],
            zs: Union[float, List[float]]
    ) -> np.ndarray:
        """
        Reverse projects points in [column, row] space to [x, y, z] real world
        Parameters
        ----------
        points : List of lists or array-like
            Points in [col, row] to unproject
        zs : float or list of floats : z-coordinate on which to unproject points

        Returns
        -------
        points_unproject : List of lists or array-like
            unprojected points as list of [x, y, z] coordinates
        """
        _, rvec, tvec = self.pnp
        # reduce zs by the mean of the gcps
        _zs = zs - self.gcps_mean[-1]
        dst = cv.unproject_points(
            np.array(points),
            _zs,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        dst = np.array(dst) + self.gcps_mean
        return dst


    def plot(
            self,
            figsize: Optional[Tuple] = (13, 8),
            ax: Optional[plt.Axes] = None,
            tiles: Optional[Any] = None,
            buffer: Optional[float] = 0.0005,
            zoom_level: Optional[int] = 19,
            camera: Optional[bool] = False,
            tiles_kwargs: Optional[Dict] = {}
    ) -> plt.Axes:
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
            if self.lens_position is not None and not camera:
            #
            # if hasattr(self, "lens_position") and not camera:
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
            self.plot_bbox(ax=ax, camera=camera, transformer=transformer, within_image=True, **patch_kwargs)
        if camera:
            # make sure that zero is on the top
            ax.set_aspect("equal")
            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        ax.legend()
        return ax

    def plot_bbox(
            self,
            ax: Optional[plt.Axes] = None,
            camera: Optional[bool] = False,
            transformer: Optional[Any] = None,
            h_a: Optional[float] = None,
            within_image: Optional[bool] = True,
            **kwargs
    ):
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
        bbox = self.get_bbox(camera=camera, h_a=h_a, within_image=within_image)
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


    def to_dict(self) -> Dict:
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

    def to_dict_str(self) -> Dict:
        d = self.to_dict()
        # convert anything that is not string in string
        dict_str = {k: v if not(isinstance(v, Polygon)) else v.__str__() for k, v in d.items()}
        return dict_str

    def to_file(
            self,
            fn: str
    ):
        """Write the CameraConfig object to json structure

        Parameters
        ----------
        fn : str
            Path to file to write camera config to

        """

        with open(fn, "w") as f:
            f.write(self.to_json())

    def to_json(self) -> str:
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


def get_camera_config(
        s: str
) -> CameraConfig:
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



def load_camera_config(
        fn: str
) -> CameraConfig:
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
