"""Camera configuration for pyorc."""

import copy
import json
import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from rasterio.features import rasterize
from shapely import ops, wkt
from shapely.geometry import LineString, Point, Polygon

from pyorc import cv, helpers, plot_helpers

MODES = Literal["camera", "geographical", "3d"]


class CameraConfig:
    """Camera configuration containing information about the perspective of the camera.

    The camera configuration contains information and functionalities to reconstruct perspective information
    relating 2D image coordinates to 3D real world coordinates.
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
        calibration_video: Optional[str] = None,
        is_nadir: Optional[bool] = False,
        stabilize: Optional[List[List]] = None,
        rotation: Optional[int] = None,
        rvec: Optional[List[float]] = None,
        tvec: Optional[List[float]] = None,
    ):
        """Initialize a camera configuration instance.

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
        camera_matrix : List[List[float]], optional
            pre-defined camera matrix (if e.g. known from a separate calibration)
        dist_coeffs : List[List[float]], optional
            pre-defined distortion parameters (if e.g. known from a separate calibration)
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
        rvec : list of floats (3), optional
            OpenCV compatible rotation vector, if known. If None, the rotation will be computed from pnp solving if
            gcps are available.
        tvec : list of floats (3), optional
            OpenCV compatible translation vector, if known. If None, the rotation will be computed from pnp solving if
            gcps are available.

        """
        assert isinstance(height, int), 'height must be provided as type "int"'
        assert isinstance(width, int), 'width must be provided as type "int"'
        assert isinstance(window_size, int), 'window_size must be of type "int"'
        self.height = height
        self.width = width
        self.is_nadir = is_nadir
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = rvec
        self.tvec = tvec
        if crs is not None:
            try:
                crs = CRS.from_user_input(crs)
            except CRSError:
                raise CRSError(f'crs "{crs}" is not a valid Coordinate Reference System')
            assert crs.is_geographic == 0, "Prodstvided crs must be projected with units like [m]"
            self.crs = crs.to_wkt()
        if resolution is not None:
            self.resolution = resolution
        if lens_position is not None:
            self.set_lens_position(*lens_position)
        else:
            self.lens_position = None
        if gcps is not None:
            self.set_gcps(**gcps)
        if self.is_nadir:
            # with nadir, no perspective can be constructed, hence, camera matrix and dist coeffs will be set
            # to default values
            self.camera_matrix = cv.get_cam_mtx(self.height, self.width)
            self.dist_coeffs = cv.DIST_COEFFS
        # camera pars are incomplete and need to be derived
        else:
            self.calibrate()
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
        """Give geographical bbox fitting around corners points of area of interest in camera perspective.

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
        """Get camera matrix."""
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, camera_matrix):
        self._camera_matrix = camera_matrix.tolist() if isinstance(camera_matrix, np.ndarray) else camera_matrix

    @property
    def dist_coeffs(self):
        """Get distortion coefficients."""
        return self._dist_coeffs

    @dist_coeffs.setter
    def dist_coeffs(self, dist_coeffs):
        self._dist_coeffs = dist_coeffs.tolist() if isinstance(dist_coeffs, np.ndarray) else dist_coeffs

    @property
    def focal_length(self):
        """Get focal length."""
        if not self.camera_matrix:
            return None
        return self.camera_matrix[0][0]

    @property
    def k1(self):
        """Get first distortion coefficient."""
        if not self.dist_coeffs:
            return None
        return self.dist_coeffs[0]

    @property
    def k2(self):
        """Get second distortion coefficient."""
        if not self.dist_coeffs:
            return None
        return self.dist_coeffs[1]

    @property
    def gcps_dest(self):
        """Get destination coordinates of GCPs.

        Returns
        -------
        dst : np.ndarray
            destination coordinates of ground control point. z-coordinates are parsed from z_0 if necessary

        """
        if hasattr(self, "gcps"):
            if "dst" in self.gcps:
                return np.array(
                    self.gcps["dst"]
                    if len(self.gcps["dst"][0]) == 3
                    else np.c_[self.gcps["dst"], np.ones(4) * self.gcps["z_0"]],
                    dtype=np.float64,
                )
        # if conditions are not yet met, then return None
        return None

    @property
    def gcps_dest_bbox(self):
        """Give destination coordinates as row, col in intended bounding box.

        Returns
        -------
        dst : np.ndarray
            Destination coordinates measured as column, row in the intended bounding box with the intended resolution

        """
        return np.array(cv.transform_to_bbox(self.gcps_dest, self.bbox, self.resolution))

    @property
    def gcps_bbox_reduced(self):
        """Give col, row coordinates of gcps within intended bounding box, reduced by mean coordinate.

        Returns
        -------
        dst : np.ndarray
            Destination coordinates in col, row in the intended bounding box, reduced with their mean coordinate

        """
        return self.gcps_dest_bbox - self.gcps_dest_bbox.mean(axis=0)

    @property
    def gcps_reduced(self):
        """Get location of gcp destination points, reduced with their mean for better local projection.

        Returns
        -------
        dst : np.ndarray
            Reduced coordinate (x, y) or (x, y, z) of gcp destination points

        """
        return np.array(self.gcps_dest - self.gcps_mean)

    @property
    def gcps_mean(self):
        """Get mean location of gcp destination points.

        Returns
        -------
        dst_mean : np.ndarray
            mean coordinate (x, y) or (x, y, z) of gcp destination points

        """
        return np.array([0.0, 0.0, 0.0]) if self.gcps_dest is None else np.array(self.gcps_dest).mean(axis=0)

    @property
    def gcps_dims(self):
        """Return amount of dimensions of GCPs provided (2 or 3).

        Returns
        -------
        dims : int
            amount of dimensions of gcps (can be 2 or 3)

        """
        return len(self.gcps["dst"][0]) if hasattr(self, "gcps") else None

    @property
    def is_nadir(self):
        """Return if the camera configuration belongs to nadir video.

        Returns
        -------
        is_nadir : bool
            False if not nadir, True if nadir

        """
        return self._is_nadir

    @is_nadir.setter
    def is_nadir(self, nadir_prop: bool):
        self._is_nadir = nadir_prop

    @property
    def pnp(self):
        """Return Precise N point solution from ground control points, intrinsics and distortion."""
        # solve rvec and tvec with reduced coordinates, this ensure that the solvepnp solution is stable.
        _, rvec, tvec = cv.solvepnp(self.gcps_reduced, self.gcps["src"], self.camera_matrix, self.dist_coeffs)
        # ensure that rvec and tvec are corrected for the fact that mean gcp location was subtracted
        rvec_cam, tvec_cam = cv.pose_world_to_camera(rvec, tvec)
        tvec_cam += self.gcps_mean
        # transform back to world
        rvec, tvec = cv.pose_world_to_camera(rvec_cam, tvec_cam)
        return rvec, tvec

    @property
    def rvec(self):
        """Return rvec from precise N point solution."""
        return self.pnp[0].tolist() if self._rvec is None else self._rvec

    @rvec.setter
    def rvec(self, _rvec):
        self._rvec = _rvec.tolist() if isinstance(_rvec, np.ndarray) else _rvec

    @property
    def shape(self):
        """Return rows and columns in projected frames from `Frames.project`.

        Returns
        -------
        rows : int
            Amount of rows in projected frame
        cols : int
            Amount of columns in projected frame

        """
        cols, rows = cv._get_shape(self.bbox, resolution=self.resolution, round=1)
        return rows, cols

    @property
    def stabilize(self):
        """Return stabilization polygon (anything outside is used for stabilization.

        Returns
        -------
        coords : list of lists
            coordinates in original image frame comprising the polygon for use in stabilization

        """
        return self._stabilize

    @stabilize.setter
    def stabilize(self, coords: List[List[float]]):
        self._stabilize = coords

    @property
    def rotation(self):
        """Return rotation OpenCV code.

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
    def rotation(self, rotation_code: int):
        self._rotation = rotation_code

    @property
    def transform(self):
        """Returns Affine transform of projected frames from `Frames.project`.

        Returns
        -------
        transform : rasterio.transform.Affine object

        """
        return cv._get_transform(self.bbox, resolution=self.resolution)

    @property
    def tvec(self):
        """Return tvec from precise N point solution."""
        return self.pnp[1].tolist() if self._tvec is None else self._tvec

    @tvec.setter
    def tvec(self, _tvec):
        self._tvec = _tvec.tolist() if isinstance(_tvec, np.ndarray) else _tvec

    def set_lens_calibration(
        self,
        fn: str,
        chessboard_size: Optional[Tuple] = (9, 6),
        max_imgs: Optional[int] = 30,
        plot: Optional[bool] = True,
        progress_bar: Optional[bool] = True,
        **kwargs,
    ):
        """Calibrate and set the properties `camera_matrix` and `dist_coeffs` using a video of a chessboard pattern.

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
        **kwargs : dict
            keyword arguments to pass to underlying methods

        """
        assert os.path.isfile(fn), f"Video calibration file {fn} not found"
        camera_matrix, dist_coeffs = cv.calibrate_camera(fn, chessboard_size, max_imgs, plot, progress_bar, **kwargs)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def estimate_lens_position(self):
        """Estimate lens position from distortion and intrinsec/extrinsic matrix."""
        rvec, tvec = np.array(self.rvec), np.array(self.tvec)
        rmat = cv2.Rodrigues(rvec)[0]
        # determine lens position related to center of objective
        lens_pos = (np.array(-rmat).T @ tvec).flatten()
        return lens_pos

    def get_bbox(
        self,
        camera: Optional[bool] = False,
        mode: Optional[MODES] = "geographical",
        h_a: Optional[float] = None,
        z_a: Optional[float] = None,
        within_image: Optional[bool] = False,
        expand_exterior=True,
    ) -> Polygon:
        """Get bounding box.

        Can be retrieved in geographical or camera perpective.

        Parameters
        ----------
        camera : bool, optional
            If set, the bounding box will be returned as row and column coordinates in the camera perspective.
            In this case ``h_a`` may be set to provide the right water level, to estimate the bounding box for.
            This option is deprecated, instead use mode="camera".
        mode : Literal["geographical", "camera", "3d"], optional
            Determines the type of bounding box to return. If set to "geographical" (default), the bounding box
            is returned in the geographical coordinates. If set to "camera", the bounding box is returned in the
            camera perspective. If set to "3d", the bounding box is returned as a 3D polygon in the CRS
        h_a : float, optional
            If set with `mode="camera"`, then the bbox coordinates will be transformed to the camera perspective,
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
        If user sets ``mode="camera"`` then the geographical bounding box will be converted into a camera perspective,
        using the homography belonging to the available ground control points and current water level.

        This can then be used to reconstruct the grid for velocimetry calculations.

        """
        if camera:
            import warnings

            warnings.warn(
                "The camera=True option is deprecated, use mode='camera' instead. This option will be removed in "
                "a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "camera"
        bbox = self.bbox
        coords = np.array(bbox.exterior.coords)
        if within_image:
            # in this case, always more points than just corners are needed, so expand_exterior is forced to True
            expand_exterior = True
        if expand_exterior:
            # make a new set of bbox coordinates with a higher density. This is meant to enable plotting of
            # distortion on image frame, and to plot partial coverage in the real-world coordinates
            coords_expand = np.zeros((0, 2))
            for n in range(0, len(coords) - 1):
                new_coords = np.linspace(coords[n], coords[n + 1], 100)
                coords_expand = np.r_[coords_expand, new_coords]
            coords = coords_expand
        if not z_a:
            z_a = self.get_z_a(h_a)
        # add vertical coordinates to the set
        coords = np.c_[coords, np.ones(len(coords)) * z_a]
        # project points to pixel image coordinates
        corners = self.project_points(coords, within_image=within_image)
        corners = corners[np.isfinite(corners[:, 0])]
        if not mode == "camera":
            # project back to real-world coordinates after possibly cutting at edges of visibility
            corners = self.unproject_points(np.array(np.array(list(zip(*corners))).T), z_a)
        if mode == "3d":
            return Polygon(corners[np.isfinite(corners[:, 0])])
        return Polygon(corners[np.isfinite(corners[:, 0])][:, 0:2])

    def get_depth(self, z: List[float], h_a: Optional[float] = None) -> List[float]:
        """Retrieve depth for measured bathymetry points.

        This is done using the camera configuration and an actual water level,
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
        self, x: List[float], y: List[float], z: List[float], h_a: Optional[float] = None
    ) -> List[float]:
        """Retrieve depth for measured bathymetry points.

        This is done using the camera configuration and an actual water level, measured
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
            assert self.gcps["h_ref"] is None, (
                "No actual water level is provided, but a reference water level is " "provided "
            )
            # h_a = 0.
            # h_ref = 0.
        # else:
        # h_ref = self.gcps["h_ref"]
        z_dry = depth <= 0
        z_dry[[0, -1]] = True
        # compute distance to nearest dry points with Pythagoras
        dist_shore = np.array([(((x[z_dry] - _x) ** 2 + (y[z_dry] - _y) ** 2) ** 0.5).min() for _x, _y in zip(x, y)])
        return dist_shore

    def get_dist_wall(self, x: List[float], y: List[float], z: List[float], h_a: Optional[float] = None) -> List[float]:
        """Retrieve distance to wall for measured bathymetry points.

        Done by using the camera configuration and an actual water level, measured in local reference (e.g. staff
        gauge).

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
        dist_wall = (dist_shore**2 + depth**2) ** 0.5
        return dist_wall

    def get_extrinsic(self):
        """Return rotation and translation vector based on control points and intrinsic parameters."""
        # solve rvec and tvec with reduced coordinates, this ensure that the solvepnp solution is stable.
        _, rvec, tvec = cv.solvepnp(self.gcps_reduced, self.gcps["src"], self.camera_matrix, self.dist_coeffs)
        # ensure that rvec and tvec are corrected for the fact that mean gcp location was subtracted
        rvec_cam, tvec_cam = cv.pose_world_to_camera(rvec, tvec)
        tvec_cam += self.gcps_mean
        # transform back to world
        rvec, tvec = cv.pose_world_to_camera(rvec_cam, tvec_cam)
        return rvec, tvec

    def z_to_h(self, z: float) -> float:
        """Convert z coordinates of bathymetry to height coordinates in local reference (e.g. staff gauge).

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

    def h_to_z(self, h_a: float) -> float:
        """Convert z coordinates of bathymetry to height coordinates in local reference (e.g. staff gauge).

        Parameters
        ----------
        h_a : float
            measured level in local datum

        Returns
        -------
        z : float
            level in global datum

        """
        h_ref = 0 if self.gcps["h_ref"] is None else self.gcps["h_ref"]
        return h_a - h_ref + self.gcps["z_0"]

    def get_M(
        self, h_a: Optional[float] = None, to_bbox_grid: Optional[bool] = False, reverse: Optional[bool] = False
    ) -> np.ndarray:
        """Establish a transformation matrix for a certain actual water level `h_a`.

        This is done by mapping where the ground control points, measured at `h_ref` will end up with new water level
        `h_a`, given the lens position.

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
            dist_coeffs=cv.DIST_COEFFS,  # self.dist_coeffs,
            z=z_a,
            reverse=reverse,
        )

    def get_z_a(self, h_a: Optional[float] = None) -> float:
        """Get actual water level measured in global vertical datum (+z_0) from water level in local datum (+h_ref).

        Parameters
        ----------
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

    def map_idx_img_ortho(self, x, y, z):
        """Map pixel indices from the image to an orthographic projection for nearest resampling.

        Can be used independently, or (when mean resampling is used) with `map_idx_img_ortho_mean`.

        Parameters
        ----------
        x : np.ndarray
            1D array representing the x-coordinates of the target orthographic grid.
        y : np.ndarray
            1D array representing the y-coordinates of the target orthographic grid.
        z : float
            The z-coordinate (elevation) value to compute the transformation.

        Returns
        -------
        idx_img : np.ndarray
            1D array of flattened indices of the source image pixels that correspond
            to selected orthographic grid pixels.
        idx_ortho : np.ndarray
            1D array of flattened indices of the destination ortho image pixels that correspond
            to the source pixels in idx_img.

        """
        # make a large flattened list of coordinates of target grid.
        cols, rows = np.meshgrid(np.arange(len(x)), np.arange(len(y)))
        xs, ys = helpers.pixel_to_map(cols.flatten(), rows.flatten(), self.transform)
        # back-project real-world coordinates to camera coordinates
        points_cam = self.project_points(list(zip(xs, ys, np.ones(len(xs)) * z)))
        # round cam coordinates to pixels
        points_cam = np.int64(np.round(points_cam))
        # find locations that lie within the camera objective, rest should remain missing value
        idx_ortho = np.all(
            [
                points_cam[:, 0] > 0,
                points_cam[:, 0] < self.width,
                points_cam[:, 1] > 0,
                points_cam[:, 1] < self.height,
            ],
            axis=0,
        )
        # check if there are values inside the objective. If not raise a warning
        if idx_ortho.sum() == 0:
            warnings.warn(
                f"The water level is either very low or high compared to the reference water level. "
                f"As a result, there are no pixels in the objective that fit in the area of interest. "
                f"Difference in water level and reference water level is {z - self.gcps['z_0']}. You will get "
                f"missing values only.",
                stacklevel=2,
            )
        # coerce 2D idxs to 1D idxs
        idx_img = np.array(points_cam[idx_ortho, 1]) * self.width + np.array(points_cam[idx_ortho, 0])
        return idx_img, idx_ortho

    def map_mean_idx_img_ortho(self, x, y, z):
        """Map pixel indices from the image to an orthographic projection for mean resampling.

        This function performs a mapping of pixel indices from one coordinate
        system (e.g., camera coordinate system) to another orthographic grid,
        with additional filtering based on specified constraints. It calculates
        the target pixel locations on the orthographic grid, checks which pixels
        reside inside the area of interest, and returns indexes and index groups
        for use in a mean resampling. undersampled areas will not be fully covered,
        hence this method should be used in conjunction with `map_idx_img_ortho`.

        Parameters
        ----------
        x : np.ndarray
            1D array representing the x-coordinates of the target orthographic grid.
        y : np.ndarray
            1D array representing the y-coordinates of the target orthographic grid.
        z : float
            The z-coordinate (elevation) value to compute the transformation.

        Returns
        -------
        src_idx : np.ndarray
            1D array of flattened indices of the source image pixels that correspond
            to selected orthographic grid pixels.
        uidx : np.ndarray
            Sorted unique indices of the filtered orthographic grid pixels.
        norm_idx : np.ndarray
            Normalized indices corresponding to their positions in the unique
            filtered orthographic grid pixels.

        """
        # mapping of pixels from image to ortho
        coli, rowi = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # first exclude pixels not within the area of interest, can save a lot of time
        poly = self.get_bbox(mode="camera", z_a=z)
        mask = rasterize([poly], out_shape=(self.height, self.width)) == 1
        src_pix = list(zip(coli[mask], rowi[mask]))

        # orthoproject pixels
        dst_pix = self.unproject_points(src_pix, z)
        x_pix, y_pix, z_pix = dst_pix.T
        idx_y, idx_x = helpers.map_to_pixel(x_pix, y_pix, self.transform)
        # ensure no pixels outside of target grid (can be in case of edges)
        # idx_inside contains False/True which pixels are within AOI from selected
        idx_inside = np.all([idx_y >= 0, idx_y < len(y), idx_x >= 0, idx_x < len(x)], axis=0)
        idx_x = idx_x[idx_inside]
        idx_y = idx_y[idx_inside]
        # get 1D flat array indexes
        idx = np.array(idx_y) * len(x) + np.array(idx_x)
        src_pix_sel = np.array(src_pix)[idx_inside]

        uidx, counts = np.unique(idx, return_counts=True)

        # Filter out indices that occur only once (covered by nearest-neighbour)
        valid_idx = uidx[counts > 1]
        mask = np.isin(idx, valid_idx)  # Mask to filter data and indices
        src_pix_sel = src_pix_sel[mask]

        # flattened indexes of source img to read
        src_idx = src_pix_sel[:, 1] * self.width + src_pix_sel[:, 0]

        # Apply the mask to both data and indices
        # filtered_data = data[mask]
        filtered_idx = idx[mask]
        # convert into a ascending lists of indexes, which are easier to process in numba
        uidx, norm_idx = np.unique(filtered_idx, return_inverse=True)
        return src_idx, uidx, norm_idx

    def set_bbox_from_corners(self, corners: List[List[float]]):
        """Establish bbox based on a set of camera perspective corner points.

        Parameters
        ----------
        corners : list of lists (4)
            [columns, row] coordinates in original camera perspective without any undistortion applied

        """
        assert np.array(corners).shape == (4, 2), (
            f"a list of lists of 4 coordinates must be given, resulting in (4, "
            f"2) shape. Current shape is {corners.shape} "
        )
        assert self.gcps["z_0"] is not None, "The water level must be set before the bounding box can be established."

        # get homography
        corners_xyz = self.unproject_points(corners, np.ones(4) * self.gcps["z_0"])
        bbox = cv.get_aoi(corners_xyz, resolution=self.resolution)
        self.bbox = bbox

    def set_bbox_from_width_length(self, points: List[List[float]]):
        """Establish bbox based on three provided points.

        The points are provided in the original camera perspective as [col, row] and require that a water level
        has already been set in order to project them in a feasible way.

        first point : left bank (seen in downstream direction)
        second point : right bank
        third point : selected upstream or downstream of the two points.

        The last point defines how large the bounding box is in up-and-downstream direction. A user should attempt to
        choose the first two points roughly in the middle of the intended bounding box. The last point is then
        used to estimate the length perpendicular to the line between the first two points. The bounding box is
        extended in both directions with the same length.

        Parameters
        ----------
        points : list of lists (3)
            [columns, row] coordinates in original camera perspective without any undistortion applied

        """
        assert np.array(points).shape == (3, 2), (
            f"a list of lists of 3 coordinates must be given, resulting in (3, "
            f"2) shape. Current shape is {np.array(points).shape} "
        )
        assert self.gcps["z_0"] is not None, "The water level must be set before the bounding box can be established."
        # get homography
        points_xyz = self.unproject_points(points, np.ones(3) * self.gcps["z_0"])
        bbox = cv.get_aoi(points_xyz, resolution=self.resolution, method="width_length")
        self.bbox = bbox

    def rotate_translate_bbox(self, angle: float = None, xoff: float = None, yoff: float = None):
        """Rotate and translate the bounding box.

        Parameters
        ----------
        angle : float, optional
            Rotation angle in radians (anti-clockwise) around the center of the bounding box
        xoff : float, optional
            Translation distance in x direction in CRS units
        yoff : float, optional
            Translation distance in y direction in CRS units

        Returns
        -------
        CameraConfig
            New CameraConfig instance with rotated and translated bounding box

        """
        # Make a deep copy of current config
        new_config = copy.deepcopy(self)

        # Get the current bbox
        bbox = new_config.bbox
        if bbox is None:
            return new_config

        # Apply rotation if specified
        if angle is not None:
            print(angle)
            # # Convert to radians
            # angle = np.radians(rotation)
            # Get centroid as origin
            centroid = bbox.centroid
            # Apply rotation around centroid
            bbox = shapely.affinity.rotate(
                bbox,
                angle,
                origin=centroid,
                use_radians=True,
            )

        # Now perform translation. Get coordinates of corners
        coords = list(bbox.exterior.coords)

        # Get unit vectors of x and y directions
        p1 = np.array(coords[0])
        p2 = np.array(coords[1])  # second point
        p3 = np.array(coords[2])  # third point

        x_vec = p2 - p1
        y_vec = p3 - p2

        x_vec = x_vec / np.linalg.norm(x_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        # Project translations onto these vectors
        dx = 0 if xoff is None else xoff * x_vec[0]
        dy = 0 if xoff is None else xoff * x_vec[1]

        dx -= 0 if yoff is None else yoff * y_vec[0]
        dy -= 0 if yoff is None else yoff * y_vec[1]

        # Apply translation
        bbox = shapely.affinity.translate(bbox, xoff=dx, yoff=dy)
        new_config.bbox = bbox
        return new_config

    def calibrate(
        self,
    ):
        """Calibrate camera parameters using ground control.

        If nothing provided, they are derived by optimizing pnp fitting together with optimizing the focal length
        and two radial distortion coefficients (k1, k2).

        You may also provide camera matrix or distortion coefficients, which will only optimize
        the remainder parameters.

        As a result, the following is updated on the CameraConfig instance:
        - camera_matrix: the 3x3 camera matrix
        - dist_coeffs: the 5x1 distortion coefficients
        - rvec: the 3x1 rotation vector
        - tvec: the 3x1 translation vector
        """
        if hasattr(self, "gcps") and (self.camera_matrix is None or self.dist_coeffs is None):
            # some calibration is needed, and there are GCPs available for it
            if len(self.gcps["src"]) >= 4:
                self.camera_matrix, self.dist_coeffs, err = cv.optimize_intrinsic(
                    self.gcps["src"],
                    self.gcps_dest,
                    # self.gcps["dst"],
                    self.height,
                    self.width,
                    lens_position=self.lens_position,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                )
        # finally, also derive the rvec and tvec if camera matrix and distortion coefficients are known
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            rvec, tvec = self.get_extrinsic()
            self.rvec = rvec
            self.tvec = tvec

    def set_gcps(
        self, src: List[List], dst: List[List], z_0: float, h_ref: Optional[float] = None, crs: Optional[Any] = None
    ):
        """Set ground control points for the given CameraConfig.

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
        assert isinstance(src, list), "src must be a list of (x, y) or (x, y, z) coordinates"
        assert isinstance(dst, list), "dst must be a list of (x, y) or (x, y, z) coordinates"
        if np.array(dst).shape[1] == 2:
            assert len(src) in [2, 4], f"2 or 4 source points are expected in src, but {len(src)} were found"
            if len(src) == 4:
                assert len(dst) == 4, f"4 destination points are expected in dst, but {len(dst)} were found"
            else:
                assert len(dst) == 2, f"2 destination points are expected in dst, but {len(dst)} were found"
        else:
            assert len(src) == len(
                dst
            ), f"Amount of (x, y, z) coordinates in src ({len(src)}) and dst ({len(dst)} must be equal"
            assert len(src) >= 6, f"for (x, y, z) points, at least 6 pairs must be available, only {len(src)} provided"
        if h_ref is not None:
            assert isinstance(h_ref, (float, int)), "h_ref must contain a float number"
        if z_0 is not None:
            assert isinstance(z_0, (float, int)), "z_0 must be provided as type float"
        assert all(isinstance(x, (float, int)) for p in src for x in p), "src contains non-int parts"
        assert all(isinstance(x, (float, int)) for p in dst for x in p), "dst contains non-float parts"
        if crs is not None:
            if not (hasattr(self, "crs")):
                raise ValueError(
                    "CameraConfig does not contain a crs, so gcps also cannot contain a crs. Ensure that the provided "
                    "destination coordinates are in a locally defined coordinate reference system, e.g. established "
                    "with a spirit level."
                )
            dst = helpers.xyz_transform(dst, crs, CRS.from_wkt(self.crs))
        # if there is no h_ref, then no local gauge system, so set h_ref to zero
        # check if 2 points are available
        if len(src) == 2:
            self.is_nadir = True
            src, dst = cv._get_gcps_2_4(src, dst, self.width, self.height)
        if h_ref is None:
            h_ref = 0.0
        self.gcps = {
            "src": src,
            "dst": dst,
            "h_ref": h_ref,
            "z_0": z_0,
        }

    def set_lens_position(self, x: float, y: float, z: float, crs: Optional[Any] = None):
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

    def project_points(self, points: List[List], within_image=False, swap_y_coords=False) -> np.ndarray:
        """Project real world x, y, z coordinates into col, row coordinates on image.

        If col, row coordinates are not allowed to go outside of the image frame, then set `within_image = True`.
        Method uses the intrinsics and extrinsics and distortion parameters to perform the projection.

        Parameters
        ----------
        points : list of lists or array-like
            list of points [x, y, z] in real world coordinates
        within_image : bool, optional
            Set coordinates to NaN if these fall outside of the image.
        swap_y_coords : bool, optional
            If set to True (default: False), y-coordinates will be swapped, in order to match plotting defaults
            which return row counting from top to bottom instead of bottom to top.

        Returns
        -------
        points_project : list or array-like
            list of points (equal in length as points) with [col, row] coordinates

        """
        rvec, tvec = np.array(self.rvec), np.array(self.tvec)
        # normalize points wrt mean of gcps
        points = np.array(points, dtype=np.float64)
        points_proj, jacobian = cv2.projectPoints(
            points, rvec, tvec, np.array(self.camera_matrix), np.array(self.dist_coeffs)
        )
        points_proj = np.array([list(point[0]) for point in points_proj])

        # points_back = cv.unproject_points(src=points_proj, z=points[:, -1], )
        if within_image:
            # also filter points outside edges of image
            points_proj[points_proj[:, 0] < 0, 0] = -1.0
            points_proj[points_proj[:, 0] > self.width - 1, 0] = self.width
            points_proj[points_proj[:, 1] < 0, 1] = -1.0
            points_proj[points_proj[:, 1] > self.height - 1, 1] = self.height
            # points_proj[points_proj[:, 0] < 0, 0] = np.nan
            # points_proj[points_proj[:, 0] > self.width - 1, 0] = np.nan  # self.width
            # points_proj[points_proj[:, 1] < 0, 1] = np.nan  # -1.0
            # points_proj[points_proj[:, 1] > self.height - 1, 1] = np.nan  # self.height

            # check which points lie behind the camera
            R, _ = cv2.Rodrigues(rvec)
            points_camera = cv.world_to_camera(points, rvec, tvec)
            behind_camera = points_camera[:, 2] <= 0.0
            # set points behind camera to nan
            points_proj[behind_camera, :] = np.nan
        # swap y coords if set
        if swap_y_coords:
            points_proj[:, 1] = self.height - points_proj[:, 1]
        return points_proj

    def project_grid(self, xs, ys, zs, swap_y_coords=False):
        """Project gridded coordinates to col, row coordinates on image.

        Method uses the intrinsics and extrinsics and distortion parameters to perform the projection.

        Parameters
        ----------
        xs : np.ndarray
            2d array of real-world x-coordinates
        ys : np.ndarray
            2d array of real-world y-coordinates
        zs : np.ndarray
            2d array of real-world z-coordinates
        swap_y_coords : bool, optional
            If set to True (default: False), y-coordinates will be swapped, in order to match plotting defaults
            which return row counting from top to bottom instead of bottom to top.

        Returns
        -------
        xp : np.ndarray
            list of col coordinates of image objective
        yp : np.ndarray
            list of row coordinates of image objective

        """
        points = list(zip(xs.flatten(), ys.flatten(), zs.flatten()))
        points_proj = np.array(self.project_points(points, swap_y_coords=swap_y_coords))
        xp, yp = points_proj[:, 0], points_proj[:, 1]
        # reshape back
        xp = np.reshape(xp, (len(xs), -1))
        yp = np.reshape(yp, (len(xs), -1))
        return xp, yp

    def unproject_points(self, points: List[List], zs: Union[float, List[float]]) -> np.ndarray:
        """Reverse projects points in [column, row] space to [x, y, z] real world.

        Parameters
        ----------
        points : List of lists or array-like
            Points in [col, row] to unproject
        zs : float or list of floats
            z-coordinates on which to unproject points

        Returns
        -------
        points_unproject : List of lists or array-like
            unprojected points as list of [x, y, z] coordinates

        """
        rvec, tvec = np.array(self.rvec), np.array(self.tvec)
        # reduce zs by the mean of the gcps
        dst = cv.unproject_points(
            np.array(points, dtype=np.float64),
            zs,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )
        dst = np.array(dst, dtype=np.float64)
        return dst

    def plot(
        self,
        figsize: Optional[Tuple] = (13, 8),
        ax: Optional[plt.Axes] = None,
        tiles: Optional[Any] = None,
        buffer: Optional[float] = 0.0005,
        zoom_level: Optional[int] = 19,
        camera: Optional[bool] = False,
        mode: Optional[MODES] = "geographical",
        pose_length: float = 1.0,
        tiles_kwargs: Optional[Dict] = None,
    ) -> plt.Axes:
        """Plot geographical situation of the CameraConfig.

        This is very useful to check if the CameraConfig seems to be in the right location. Requires `cartopy`
        to be installed.

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
            This option is deprecated, instead use mode="camera".
        mode : Literal["geographical", "camera", "3d"], optional
            Determines the type of bounding box to return. If set to "geographical" (default), the bounding box
            is returned in the geographical coordinates. If set to "camera", the bounding box is returned in the
            camera perspective. If set to "3d", the bounding box is returned as a 3D polygon in the CRS
        pose_length: float, optional
            length of pose axes to draw (only used in mode="3d").
        tiles_kwargs : dict
            additional keyword arguments to pass to ax.add_image when tiles are added

        Returns
        -------
        ax : plt.axes

        """
        if camera:
            import warnings

            warnings.warn(
                "The camera=True option is deprecated, use mode='camera' instead. This option will be removed in "
                "a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "camera"
        if not tiles_kwargs:
            tiles_kwargs = {}
        # initiate transform
        transformer = None
        # if there is an axes, get the extent
        xlim = ax.get_xlim() if ax is not None else None
        ylim = ax.get_ylim() if ax is not None else None

        # prepare points for plotting
        if mode == "camera":
            points = [Point(x, y) for x, y in self.gcps["src"]]
        elif mode == "geographical":
            points = [Point(p[0], p[1]) for p in self.gcps["dst"]]
        else:
            # 3d points are needed
            if hasattr(self, "gcps"):
                if len(self.gcps["dst"]) == 3:
                    points = [Point(*p) for p in self.gcps["dst"]]
                else:
                    points = [Point(p[0], p[1], self.gcps["z_0"]) for p in self.gcps["dst"]]
        if mode != "camera":
            if self.lens_position is not None:
                lens_position = self.lens_position
            else:
                lens_position = self.estimate_lens_position()
            if mode == "3d":
                points.append(Point(*lens_position))
            else:
                points.append(Point(lens_position[0], lens_position[1]))
            # transform points in case a crs is provided and we want a geographical plot
            if mode == "geographical" and hasattr(self, "crs"):
                # make a transformer to lat lon
                transformer = Transformer.from_crs(
                    CRS.from_user_input(self.crs), CRS.from_epsg(4326), always_xy=True
                ).transform
                points = [ops.transform(transformer, p) for p in points]
            if mode == "geographical":
                xmin, ymin, xmax, ymax = list(np.array(LineString(points).bounds))
                extent = [xmin - buffer, xmax + buffer, ymin - buffer, ymax + buffer]
        x = [p.x for p in points]
        y = [p.y for p in points]
        if mode == "3d":
            z = [p.z for p in points]
        if ax is None:
            plt.figure(figsize=figsize)
            if hasattr(self, "crs") and mode == "geographical":
                ax = helpers.get_geo_axes(tiles=tiles, extent=extent, zoom_level=zoom_level, **tiles_kwargs)
            else:
                if mode == "3d":
                    ax = plt.axes(projection="3d")
                else:
                    ax = plt.axes()
        if hasattr(ax, "add_geometries"):
            import cartopy.crs as ccrs

            plot_kwargs = dict(transform=ccrs.PlateCarree())
        else:
            plot_kwargs = {}
        if hasattr(self, "gcps"):
            if mode == "3d":
                ax.plot(
                    x[0 : len(self.gcps["dst"])],
                    y[0 : len(self.gcps["dst"])],
                    z[0 : len(self.gcps["dst"])],
                    "o",
                    label="Control points",
                    markersize=12,
                    markeredgecolor="w",
                    zorder=2,
                    **plot_kwargs,
                )
            else:
                ax.plot(
                    x[0 : len(self.gcps["dst"])],
                    y[0 : len(self.gcps["dst"])],
                    ".",
                    label="Control points",
                    markersize=12,
                    markeredgecolor="w",
                    zorder=2,
                    **plot_kwargs,
                )
        # if len(x) > len(self.gcps["dst"]):
        if mode == "3d":
            ax.plot(
                x[-1],
                y[-1],
                z[-1],
                "o",
                label="Lens position",
                markersize=12,
                zorder=2,
                markeredgecolor="w",
                **plot_kwargs,
            )
            # add pose
            _ = self.plot_3d_pose(ax=ax, length=pose_length)
            if hasattr(self, "bbox"):
                # also plot dashed lines from cam to bbox
                for xy in self.bbox.exterior.coords:
                    ax.plot([x[-1], xy[0]], [y[-1], xy[1]], [z[-1], self.gcps["z_0"]], linestyle="--", color="gray")
                # plot bbox exterior
                ax.plot(*self.bbox.exterior.xy, [self.gcps["z_0"]] * 5, color="k", label="bbox exterior")
        else:
            ax.plot(
                x[-1],
                y[-1],
                ".",
                label="Lens position",
                markersize=12,
                zorder=2,
                markeredgecolor="w",
                **plot_kwargs,
            )
        patch_kwargs = {
            **plot_kwargs,
            "alpha": 0.5,
            "zorder": 2,
            "edgecolor": "w",
            "label": "bbox visible",
            **plot_kwargs,
        }
        if hasattr(self, "bbox"):
            self.plot_bbox(ax=ax, mode=mode, transformer=transformer, within_image=True, **patch_kwargs)
        if mode == "camera":
            # make sure that zero is on the top
            ax.set_aspect("equal")
            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            ax.set_xlabel("column [-]")
            ax.set_ylabel("row [-]")
        elif mode == "3d":
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
        ax.legend()
        return ax

    def plot_bbox(
        self,
        ax: Optional[plt.Axes] = None,
        camera: Optional[bool] = False,
        mode: Optional[MODES] = "geographical",
        transformer: Optional[Any] = None,
        h_a: Optional[float] = None,
        within_image: Optional[bool] = True,
        **kwargs,
    ):
        """Plot bounding box.

        This can be done for orthorectification in a geographical projection (`camera=False`) or the camera
        Field Of View (`mode="camera"`).

        Parameters
        ----------
        ax : plt.axes, optional
            if not provided, axes is setup (Default: None)
        camera : bool, optional
            If set to True, all camera config information will be back projected to the original camera objective.
            This option is deprecated, instead use mode="camera".
        mode : Literal["geographical", "camera", "3d"], optional
            Determines the type of bounding box to return. If set to "geographical" (default), the bounding box
            is returned in the geographical coordinates. If set to "camera", the bounding box is returned in the
            camera perspective. If set to "3d", the bounding box is returned as a 3D polygon in the CRS
        transformer : pyproj transformer transformation function, optional
            used to reproject bbox to axes object projection (e.g. lat lon)
        h_a : float, optional
            If set with `mode="camera"`, then the bbox coordinates will be transformed to the camera perspective,
            using h_a as a present water level. In case a video with higher (lower) water levels is used, this
            will result in a different perspective plane than the control video.
        within_image : bool, optional
            If set (default), points outside the camera objective are removed.
        **kwargs : dict
            additional keyword arguments used for plotting the bbox polygon with `matplotlib.patches.Polygon`

        Returns
        -------
        p : matplotlib.patch mappable

        """
        # collect information to plot
        if camera:
            import warnings

            warnings.warn(
                "The camera=True option is deprecated, use mode='camera' instead. This option will be removed in "
                "a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "camera"
        bbox = self.get_bbox(mode=mode, h_a=h_a, within_image=within_image)
        if mode == "geographical" and transformer is not None:
            # geographical projection is needed
            bbox = ops.transform(transformer, bbox)
        if mode == "3d":
            return plot_helpers.plot_3d_polygon(bbox, ax=ax, **kwargs)
        return plot_helpers.plot_polygon(bbox, ax=ax, **kwargs)

    def plot_3d_pose(self, ax=None, length=1):
        """Plot 3D pose of a camera using its rotation and translation vectors.

        Parameters
        ----------
        ax : axes, optional
            3d axes to plot on, if not set, a new axes will be established.
        length : float
            length of the axes drawn in meters

        Returns
        -------
        list[handles]
            list of handles to the plotted pose axes

        """
        rvec = np.array(self.rvec)
        tvec = np.array(self.tvec)
        # rvec, tvec = cv.pose_world_to_camera(rvec, tvec)
        # Convert the rotation vector to a 3x3 rotation matrix
        R, _ = cv2.Rodrigues(rvec.flatten())

        # Define the camera's axis directions in its local coordinate system
        camera_axes = (
            np.array(
                [
                    [0, 0, 0],  # lens center
                    [1, 0, 0],  # X-axis (red - right looking)
                    [0, 1, 0],  # Y-axis (green - down looking)
                    [0, 0, 1],  # Z-axis (blue - forward looking)
                ]
            )
            * length
        )
        pts_trans = camera_axes - tvec
        world_axes_translated = (R.T @ pts_trans.T).T
        ax = plt.axes(projection="3d") if ax is None else ax
        # Plot the origin of the camera
        ps = []
        # Plot the camera axes
        for i, (color, label) in enumerate(zip(["r", "g", "b"], ["right-pose", "down-pose", "forward-pose"])):
            # if i == 2:
            xx = [world_axes_translated[0, 0], world_axes_translated[i + 1, 0]]
            yy = [world_axes_translated[0, 1], world_axes_translated[i + 1, 1]]
            zz = [world_axes_translated[0, 2], world_axes_translated[i + 1, 2]]
            ps.append(ax.plot(xx, yy, zz, color=color, label=label, linewidth=3))
        return ps

    def to_dict(self) -> Dict:
        """Return the CameraConfig object as dictionary.

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
        """Convert the current instance to a dictionary with all values converted to strings.

        Returns
        -------
        dict
            A dictionary representation of the instance where all values are strings. If an attribute is an instance
            of `Polygon`, it is converted to its string representation.

        """
        d = self.to_dict()
        # convert anything that is not string in string
        dict_str = {k: v if not (isinstance(v, Polygon)) else v.__str__() for k, v in d.items()}
        return dict_str

    def to_file(self, fn: str):
        """Write the CameraConfig object to json structure.

        Parameters
        ----------
        fn : str
            Path to file to write camera config to

        """
        with open(fn, "w") as f:
            f.write(self.to_json())

    def to_json(self) -> str:
        """Convert CameraConfig object to string.

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


def get_camera_config(s: str) -> CameraConfig:
    """Read camera config from string.

    Parameters
    ----------
    s : str
        json string containing camera config

    Returns
    -------
    cam_config : CameraConfig

    """
    d = json.loads(s)
    if "height" not in d or "width" not in d:
        raise IOError(depr_warning_height_width)
    # ensure the bbox is a Polygon object
    if "bbox" in d:
        if isinstance(d["bbox"], str):
            d["bbox"] = wkt.loads(d["bbox"])
    return CameraConfig(**d)


def load_camera_config(fn: str) -> CameraConfig:
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
