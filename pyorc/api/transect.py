"""Transect module for pyorc."""

import numpy as np
import xarray as xr
from xarray.core import utils

from pyorc import helpers

from .cross_section import CrossSection
from .orcbase import ORCBase
from .plot import _Transect_PlotMethods


@xr.register_dataset_accessor("transect")
class Transect(ORCBase):
    """Transect functionalities that can be applied on ``xarray.Dataset``."""

    def __init__(self, xarray_obj):
        """Initialize a transect ``xarray.Dataset`` containing cross-sectional velocities in [time, points] dimensions.

        Parameters
        ----------
        xarray_obj: xr.Dataset
            transect data fields (from ``pyorc.Velocimetry.get_transect``)

        """
        super(Transect, self).__init__(xarray_obj)

    @property
    def cross_section(self):
        """Return cross-sectional coordinates as `CrossSection` object."""
        if not hasattr(self._obj, "zcoords"):
            return None
        coords = [[_x, _y, _z] for _x, _y, _z in zip(self._obj.xcoords, self._obj.ycoords, self._obj.zcoords)]
        return CrossSection(camera_config=self.camera_config, cross_section=coords)

    def vector_to_scalar(self, v_x="v_x", v_y="v_y"):
        """Set "v_eff" and "v_dir" variables as effective velocities over cross-section, and its angle.

        Parameters
        ----------
        v_x : str, optional
            name of variable containing x-directional velocities. (default: "v_x")
        v_y :
            name of variable containing y-directional velocities. (default: "v_y")

        Returns
        -------
        da : xr.DataArray
            velocities perpendicular to cross section [time, points]

        """
        # compute per velocity vector in the dataset, what its angle is
        v_angle = np.arctan2(self._obj[v_x], self._obj[v_y])
        # compute the scalar value of velocity
        v_scalar = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5

        # compute difference in angle between velocity and perpendicular of cross section
        flow_dir = self._obj["v_dir"]
        angle_diff = v_angle - flow_dir
        # compute effective velocity in the flow direction (i.e. perpendicular to cross section)
        v_eff = np.cos(angle_diff) * v_scalar
        v_eff.attrs = {
            "standard_name": "velocity",
            "long_name": "velocity in perpendicular direction of cross section, measured by angle in radians, "
            "measured from up-direction",
            "units": "m s-1",
        }
        # set name
        v_eff.name = "v_eff_nofill"  # there still may be gaps in this series
        self._obj["v_eff_nofill"] = v_eff

    def get_bottom_surface_z_perspective(self, h, sample_size=1000, interval=None):
        """Return densified bottom and surface points, warped to image perspective."""
        # get bottom coordinates
        bottom_points = self.get_transect_perspective(within_image=True)

        # get surface coordinates
        surface_points = self.get_transect_perspective(h=h, within_image=True)

        # densify points, to ensure zero water level crossings are captured
        bottom_points = helpers.densify_points(bottom_points, sample_size=sample_size)
        surface_points = helpers.densify_points(surface_points, sample_size=sample_size)

        # also densify to the same amount with zcoords
        z_points = helpers.densify_points(self._obj.zcoords, sample_size=sample_size)

        if interval is not None:
            # only sample every interval-th point
            bottom_points = bottom_points[::interval]
            surface_points = surface_points[::interval]
            z_points = z_points[::interval]
        # understand which points are below surface
        z_surface = h - self.camera_config.gcps["h_ref"] + self.camera_config.gcps["z_0"]
        mask = z_points < z_surface

        # filter bottom and surface
        bottom_points = np.array(bottom_points)[mask]
        surface_points = np.array(surface_points)[mask]
        return bottom_points, surface_points

    def get_transect_perspective(self, h=None, within_image=True):
        """Get row, col locations of the transect coordinates.

        Parameters
        ----------
        h : float, optional
            Water level (measured locally) used to calculate surface coordinates. If not provided, bottom coordinates.
            are used.
        within_image : bool, optional
            If True (default), removes points outside the camera objective.

        Returns
        -------
        numpy.ndarray
            array of projected transect points based on the camera configuration and at given water level (if provided).

        """
        x = self._obj.xcoords
        y = self._obj.ycoords
        if h is not None:
            z_surface = h - self.camera_config.gcps["h_ref"] + self.camera_config.gcps["z_0"]
            z = np.ones(len(x)) * z_surface
            # retrieve coordinates at the surface
        else:
            z = self._obj.zcoords  # retrieve the bottom coordinates
        points = [[_x, _y, _z] for _x, _y, _z in zip(x, y, z)]
        # ensure that y coordinates are in the right direction
        points_proj = self.camera_config.project_points(points, within_image=within_image, swap_y_coords=True)
        return points_proj

    def get_wetted_perspective(self, h, sample_size=1000):
        """Get wetted polygon in camera perspective.

        Parameters
        ----------
        h : float
            The water level value to calculate the surface perspective.
        sample_size : int, optional
            The number of points to densify the transect with, by default 1000

        Returns
        -------
        ndarray
            A numpy array containing the points forming the wetted polygon perspective.

        """
        bottom_points, surface_points = self.get_bottom_surface_z_perspective(h=h, sample_size=sample_size)
        # concatenate points reversing one set for preps of a polygon
        pol_points = np.concatenate([bottom_points, np.flipud(surface_points)], axis=0)

        # add the first point at the end to close the polygon
        pol_points = np.concatenate([pol_points, pol_points[0:1]], axis=0)
        return pol_points

    def get_depth_perspective(self, h, sample_size=1000, interval=25):
        """Get line (x, y) pairs that show the depth over several intervals in the wetted part of the cross section.

        Parameters
        ----------
        h : float
            The water level with which the depth perspective needs to be calculated.
        sample_size : int, optional
            The number of samples to create by interpolating the cross section (default is 1000).
        interval : int, optional
            The interval between extended samples (default is 25).

        Returns
        -------
        List of (x, y) tuple pairs.
            Each tuple pair defines one perspective depth line.

        """
        bottom_points, surface_points = self.get_bottom_surface_z_perspective(
            h=h, sample_size=sample_size, interval=interval
        )
        # make line pairs
        return list(zip(bottom_points, surface_points))

    def get_xyz_perspective(self, trans_mat=None, xs=None, ys=None, mask_outside=True):
        """Get camera-perspective column, row coordinates from cross-section locations.

        Parameters
        ----------
        trans_mat : np.ndarray, optional
             perspective transform matrix (Default value = None)
        xs : np.array, optional
            x-coordinates to transform, derived from self.x if not provided (Default value = None)
        ys :
            y-coordinates to transform, derived from self.y if not provided (Default value = None)
        mask_outside :
            values not fitting in the original camera frame are set to NaN (Default value = True)

        Returns
        -------
        cols : list of ints
            columns of locations in original camera perspective
        rows : list of ints
            rows of locations in original camera perspective


        """
        if xs is None:
            xs = self._obj.x.values
        if ys is None:
            ys = self._obj.y.values
        # compute bathymetry as measured in local height reference (such as staff gauge)
        # if self.camera_config.gcps["h_ref"] is None:
        #     h_ref = 0.0
        # else:
        #     h_ref = self.camera_config.gcps["h_ref"]
        hs = self.camera_config.z_to_h(self._obj.zcoords).values
        # zs = (self._obj.zcoords - self.camera_config.gcps["z_0"] + h_ref).values
        if trans_mat is None:
            ms = [self.camera_config.get_M(h, reverse=True, to_bbox_grid=True) for h in hs]
        else:
            # use user defined M instead
            ms = [trans_mat for _ in hs]
        # compute row and column position of vectors in original reprojected background image col/row coordinates
        cols, rows = zip(
            *[
                helpers.xy_to_perspective(
                    x, y, self.camera_config.resolution, trans_mat, reverse_y=self.camera_config.shape[0]
                )
                for x, y, trans_mat in zip(xs, ys, ms)
            ],
        )
        # ensure y coordinates start at the top in the right orientation
        shape_y, shape_x = self.camera_shape
        rows = shape_y - np.array(rows)
        cols = np.array(cols)
        if mask_outside:
            # remove values that do not fit in the frames
            cols[np.any([cols < 0, cols > self.camera_shape[1]], axis=0)] = np.nan
            rows[np.any([rows < 0, rows > self.camera_shape[0]], axis=0)] = np.nan

        return cols, rows

    def get_river_flow(self, q_name="q", discharge_name="river_flow"):
        """Integrate time series of depth averaged velocities [m2 s-1] into cross-section integrated flow [m3 s-1].

        Depth average velocities must first have been estimated using get_q. A variable "Q" will be added to Dataset
        with only "quantiles" as dimension.

        Parameters
        ----------
        q_name : str, optional
             name of variable where depth integrated velocities [m2 s-1] are stored (Default value = "q")
        discharge_name : str, optional
             name of variable where resulting width integrated river flow estimates must be stored
             (Default value = "river_flow")

        """
        if "q" not in self._obj:
            raise ValueError(
                f'Dataset must contain variable "{q_name}", which is the depth-integrated velocity [m2 s-1], '
                "perpendicular to cross-section. Create this with ds.transect.get_q"
            )
        # integrate over the distance coordinates (s-coord)
        discharge = self._obj[q_name].fillna(0.0).integrate(coord="scoords")
        discharge.attrs = {
            "standard_name": "river_discharge",
            "long_name": "River Flow",
            "units": "m3 s-1",
        }
        # set name
        discharge.name = "Q"
        self._obj[discharge_name] = discharge

    def get_q(self, v_corr=0.9, fill_method="zeros"):
        """Integrated velocity over depth for quantiles of time series.

        A correction `v_corr` between surface velocity and depth-average velocity is used.

        Parameters
        ----------
        v_corr : float, optional
            correction factor (default: 0.9)
        fill_method : str, optional
            method to fill missing values. "zeros" fills NaNS with zeros, "interpolate" interpolates values
            from nearest neighbour, "log_interp" interpolates values linearly with velocities scaled by the log of
            depth over a roughness length, "log_fit" fits a 4-parameter logarithmic profile with depth and with
            changing velocities towards banks on known velocities, and fills missing with the fitted relationship
            (experimental) (Default value = "zeros").

        Returns
        -------
        ds : xr.Dataset
            Transect for selected quantiles in time, with "q_nofill" containing the integrated values, and "q" the
            integrated values, filled with chosen fill method.

        """
        # aggregate to a limited set of quantiles
        assert fill_method in [
            "zeros",
            "log_fit",
            "log_interp",
            "interpolate",
        ], f'fill_method must be "zeros", "log_fit", "log_interp", or "interpolate", instead "{fill_method}" given'
        ds = self._obj
        x = ds["xcoords"].values
        y = ds["ycoords"].values
        z = ds["zcoords"].values
        # add filled surface velocities with a logarithmic profile curve fit
        depth = self.camera_config.get_depth(z, self.h_a)
        # make velocities zero where depth is zero
        ds["v_eff_nofill"][:, depth <= 0] = 0.0
        if fill_method == "zeros":
            ds["v_eff"] = ds["v_eff_nofill"].fillna(0.0)
        elif fill_method == "log_fit":
            dist_shore = self.camera_config.get_dist_shore(x, y, z, self.h_a)
            ds["v_eff"] = helpers.velocity_log_fit(ds["v_eff_nofill"], depth, dist_shore, dim="quantile")
        elif fill_method == "log_interp":
            dist_wall = self.camera_config.get_dist_wall(x, y, z, self.h_a)
            ds["v_eff"] = helpers.velocity_log_interp(ds["v_eff_nofill"], dist_wall, dim="quantile")
        elif fill_method == "interpolate":
            depth = ds.zcoords * 0 + self.camera_config.get_depth(ds.zcoords, self.h_a)
            # interpolate gaps in between known values
            ds["v_eff"] = ds["v_eff_nofill"].interpolate_na(dim="points")
            # anywhere where depth == 0, remove values
            ds["v_eff"] = ds["v_eff"].where(depth > 0)
            # fill NAs with zeros
            ds["v_eff"] = ds["v_eff"].fillna(0.0)
        # compute q for both non-filled and filled velocities
        ds["q_nofill"] = helpers.depth_integrate(depth, ds["v_eff_nofill"], v_corr=v_corr, name="q_nofill")
        ds["q"] = helpers.depth_integrate(depth, ds["v_eff"], v_corr=v_corr, name="q")
        return ds

    plot = utils.UncachedAccessor(_Transect_PlotMethods)
