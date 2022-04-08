import copy
from matplotlib.colors import Normalize
import numpy as np
import rasterio
import xarray as xr
import warnings

from pyproj import CRS
from scipy.interpolate import interp1d
from .transect import Transect
from .orcbase import ORCBase
from .. import helpers, const
import pyorc.plot as plot_orc


@xr.register_dataset_accessor("velocimetry")
class Velocimetry(ORCBase):
    def __init__(self, xarray_obj):
        super(Velocimetry, self).__init__(xarray_obj)

    def filter_temporal(
            self,
            v_x="v_x",
            v_y="v_y",
            filter_std=True,
            filter_angle=True,
            filter_velocity=True,
            filter_corr=True,
            filter_neighbour=True,
            kwargs_corr={},
            kwargs_std={},
            kwargs_angle={},
            kwargs_velocity={},
            kwargs_neighbour={},
            inplace=False
    ):
        """
        Masks values using several filters that use temporal variations or comparison as basis.

        :param ds: xr.Dataset, or file containing, with velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param filter_std: boolean, if True (default, filtering on variance is applied)
        :param filter_std: boolean, if True (default, filtering on variance is applied)
        :param filter_angle: boolean, if True (default, filtering on angles is applied)
        :param filter_velocity: boolean, if True (default, filtering on velocity is applied)
        :param filter_corr: boolean, if True (default, filtering on correlation is applied)
        :param kwargs_std: dict, set of key-word arguments to pass on to filter_temporal_std
        :param kwargs_angle: dict, set of key-word arguments to pass on to filter_temporal_angle
        :param kwargs_velocity: dict, set of key-word arguments to pass on to filter_temporal_velocity
        :param kwargs_corr: dict, set of key-word arguments to pass on to filter_temporal_corr
        :return: xr.Dataset, containing temporally filtered velocity vectors as [time, y, x]
        """
        # load dataset in memory and update self
        ds = copy.deepcopy(self._obj.load())
        # start with entirely independent filters
        if filter_corr:
            ds.velocimetry.filter_temporal_corr(v_x=v_x, v_y=v_y, **kwargs_corr)
        if filter_velocity:
            ds.velocimetry.filter_temporal_velocity(v_x=v_x, v_y=v_y, **kwargs_velocity)
        if filter_neighbour:
            ds.velocimetry.filter_temporal_neighbour(v_x=v_x, v_y=v_y, **kwargs_neighbour)
        # finalize with temporally dependent filters
        if filter_std:
            ds.velocimetry.filter_temporal_std(v_x=v_x, v_y=v_y, **kwargs_std)
        if filter_angle:
            ds.velocimetry.filter_temporal_angle(v_x=v_x, v_y=v_y, **kwargs_angle)
        ds.attrs = self._obj.attrs
        if inplace:
            self._obj.update(ds)
        else:
            return ds

    def filter_temporal_angle(
            self,
            v_x="v_x",
            v_y="v_y",
            angle_expected=0.5 * np.pi,
            angle_tolerance=0.25 * np.pi,
            filter_per_timestep=True,
    ):
        """
        filters on the expected angle. The function filters points entirely where the mean angle over time
        deviates more than input parameter angle_bounds (in radians). The function also filters individual
        estimates in time, in case the user wants this (filter_per_timestep=True), in case the angle on
        a specific time step deviates more than the defined amount from the average.
        note: this function does not work appropriately, if the expected angle (+/- anglebounds) are within
        range of zero, as zero is the same as 2*pi. This exception may be resolved in the future if necessary.

        :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param angle_expected float, angle (0-2*pi), measured clock-wise from vertical upwards direction, expected
            in the velocites, default: 0.5*np.pi (meaning from left to right)
        :param angle_tolerance: float (0-2*pi) maximum deviation from expected angle allowed.
        :param filter_per_timestep: if set to True, tolerances are also checked per individual time step
        :return: xr.Dataset, containing angle filtered velocity vectors as [time, y, x], default: True
        """
        # TODO: make function working appropriately, if angles are close to zero (2*pi)
        # first filter on the temporal mean. This is to ensure that widely varying results in angle are deemed not
        # to be trusted.
        v_x_mean = self._obj[v_x].mean(dim="time")
        v_y_mean = self._obj[v_y].mean(dim="time")
        angle_mean = np.arctan2(v_x_mean, v_y_mean)
        # angle_mean = angle.mean(dim="time")
        self._obj[v_x] = self._obj[v_x].where(np.abs(angle_mean - angle_expected) < angle_tolerance)
        self._obj[v_y] = self._obj[v_y].where(np.abs(angle_mean - angle_expected) < angle_tolerance)
        # refine locally if user wishes so
        if filter_per_timestep:
            angle = np.arctan2(self._obj[v_x], self._obj[v_y])
            self._obj[v_x] = self._obj[v_x].where(np.abs(angle - angle_expected) < angle_tolerance)
            self._obj[v_y] = self._obj[v_y].where(np.abs(angle - angle_expected) < angle_tolerance)

    def filter_temporal_neighbour(self, v_x="v_x", v_y="v_y", roll=5, tolerance=0.5):
        """
        Masks values if neighbours over a certain rolling length before and after, have a
        significantly higher velocity than value under consideration, measured by tolerance.

        :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param roll: int, amount of time steps in rolling window (centred)
        :param tolerance: float (0-1), Relative acceptable velocity of maximum found within rolling window
        :return: xr.Dataset, containing time-neighbour filtered velocity vectors as [time, y, x]
        """
        s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        s_roll = s.fillna(0.).rolling(time=roll, center=True).max()
        self._obj[v_x] = self._obj[v_x].where(s > tolerance * s_roll)
        self._obj[v_y] = self._obj[v_y].where(s > tolerance * s_roll)
        # return ds

    def filter_temporal_std(
            self, v_x="v_x", v_y="v_y", tolerance_sample=1.0, tolerance_var=5., mode="or"):
        """
        Masks values if they deviate more than x standard deviations from the mean.

        :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param tolerance: float, representing amount of standard deviations
        :return: xr.Dataset, containing standard deviation filtered velocity vectors as [time, y, x]
        """

        # s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # s_std = s.std(dim="time")
            x_std = self._obj[v_x].std(dim="time")
            y_std = self._obj[v_y].std(dim="time")
        x_mean = self._obj[v_x].mean(dim="time")
        y_mean = self._obj[v_y].mean(dim="time")
        x_var = np.abs(x_std / x_mean)
        y_var = np.abs(y_std / y_mean)

        # s_mean = s.mean(dim="time")
        # s_var = s_std / s_mean
        x_condition = np.abs((self._obj[v_x] - x_mean) / x_std) < tolerance_sample
        y_condition = np.abs((self._obj[v_y] - y_mean) / y_std) < tolerance_sample
        if mode == "or":
            condition = np.any([x_condition, y_condition], axis=0)
        else:
            condition = np.all([x_condition, y_condition], axis=0)
        self._obj[v_x] = self._obj[v_x].where(condition)
        self._obj[v_y] = self._obj[v_y].where(condition)
        # also remove places with a high variance in either (or both) directions
        x_condition = x_var < tolerance_var
        y_condition = y_var < tolerance_var
        if mode == "or":
            condition = np.any([x_condition, y_condition], axis=0)
        else:
            condition = np.all([x_condition, y_condition], axis=0)
        self._obj[v_x] = self._obj[v_x].where(condition)
        self._obj[v_y] = self._obj[v_y].where(condition)

        #
        # self._obj[v_x] = self._obj[v_x].where((s - s_mean) / s_std < tolerance)
        # self._obj[v_y] = self._obj[v_y].where((s - s_mean) / s_std < tolerance)
        # return ds

    def filter_temporal_velocity(self, v_x="v_x", v_y="v_y", s_min=0.1, s_max=5.0):
        """
        Masks values if the velocity scalar lies outside a user-defined valid range.

        :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param s_min: float, minimum scalar velocity [m s-1]
        :param s_max: float, maximum scalar velocity [m s-1]
        :return: xr.Dataset, containing velocity-range filtered velocity vectors as [time, y, x]
        """
        s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        self._obj[v_x] = self._obj[v_x].where(s > s_min)
        self._obj[v_x] = self._obj[v_x].where(s < s_max)
        self._obj[v_y] = self._obj[v_y].where(s > s_min)
        self._obj[v_y] = self._obj[v_y].where(s < s_max)
        s_mean = s.mean(dim="time")
        self._obj[v_x] = self._obj[v_x].where(s_mean > s_min)
        self._obj[v_x] = self._obj[v_x].where(s_mean < s_max)
        self._obj[v_y] = self._obj[v_y].where(s_mean > s_min)
        self._obj[v_y] = self._obj[v_y].where(s_mean < s_max)

        # return ds

    def filter_temporal_corr(self, v_x="v_x", v_y="v_y", corr="corr", tolerance=0.1):
        """
        Masks values with a too low correlation.

        :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param corr: str, name of correlation variable
        :param tolerance: float (0-1), tolerance for correlation value. If correlation is lower than tolerance, it is masked
        :return: xr.Dataset, containing correlation filtered velocity vectors as [time, y, x]
        """
        self._obj[v_x] = self._obj[v_x].where(self._obj[corr] > tolerance)
        self._obj[v_y] = self._obj[v_y].where(self._obj[corr] > tolerance)
        # return ds

    def filter_spatial(
            self,
            v_x="v_x",
            v_y="v_y",
            filter_nan=True,
            filter_median=True,
            kwargs_nan={},
            kwargs_median={},
            inplace=False
    ):
        """
        Masks velocity values on a number of spatial filters.

        :param self: xr.Dataset, or file containing, with velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param kwargs_nan: dict, keyword arguments to pass to filter_spatial_nan
        :param kwargs_median: dict, keyword arguments to pass to filter_spatial_median
        :return: xr.Dataset, containing spatially filtered velocity vectors as [time, y, x]
        """
        # work on v_x and v_y only
        ds_temp = self._obj[[v_x, v_y]].copy(deep=True).load()
        if filter_nan:
            ds_temp.velocimetry.filter_spatial_nan(v_x=v_x, v_y=v_y, **kwargs_nan)
        if filter_median:
            ds_temp.velocimetry.filter_spatial_median(v_x=v_x, v_y=v_y, **kwargs_median)
        # merge the temporary set with the original
        ds = xr.merge([self._obj.drop_vars([v_x, v_y]), ds_temp])
        ds.attrs = self._obj.attrs
        if inplace:
            self._obj.update(ds)
        else:
            return ds

    def filter_spatial_nan(self, v_x="v_x", v_y="v_y", **kwargs):
        """
        Masks values if their surrounding neighbours (inc. value itself) contain too many NaN. Meant to remove isolated
        velocity estimates.

        :param self: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param tolerance: float, amount of NaNs in search window measured as a fraction of total amount of values [0-1]
        :param stride: int, stride used to determine relevant neighbours
        :param missing: float, a temporary missing value, used to be able to convolve NaNs
        :return: xr.Dataset, containing NaN filtered velocity vectors as [time, y, x]
        """
        def _filter_nan(ds_slice, v_x="v_x", v_y="v_y", tolerance=0.3, stride=1, missing=-9999.):
            # u, v = ds[v_x], ds[v_y]
            u, v = ds_slice[v_x].values, ds_slice[v_y].values
            u_move = helpers.neighbour_stack(u.copy(), stride=stride, missing=missing)
            # replace missings by Nan
            nan_frac = np.float64(np.isnan(u_move)).sum(axis=0) / float(len(u_move))
            u[nan_frac > tolerance] = np.nan
            v[nan_frac > tolerance] = np.nan
            ds_slice[v_x][:] = u
            ds_slice[v_y][:] = v
            return ds_slice
        ds_g = self._obj.groupby("time")
        self._obj.update(
            ds_g.apply(_filter_nan, v_x=v_x, v_y=v_y, **kwargs)
        )

    def filter_spatial_median(self, v_x="v_x", v_y="v_y", **kwargs):
        """
        Masks values when their value deviates more than x standard deviations from the median of its neighbours
        (inc. itself).

        :param self: xr.Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param tolerance: float, amount of standard deviations tolerance
        :param stride: int, stride used to determine relevant neighbours
        :param missing: float, a temporary missing value, used to be able to convolve NaNs
        :return: xr.Dataset, containing std filtered velocity vectors as [time, y, x]
        """
        def _filter_median(ds_slice, v_x="v_x", v_y="v_y", tolerance=0.7, stride=1, missing=-9999.):
            u, v = ds_slice[v_x].values, ds_slice[v_y].values
            s = (u ** 2 + v ** 2) ** 0.5
            s_move = helpers.neighbour_stack(s, stride=stride)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # replace missings by Nan
                s_median = np.nanmedian(s_move, axis=0)
            # now filter points that are very far off from the median
            filter = np.abs(s - s_median) / s_median > tolerance
            u[filter] = np.nan
            v[filter] = np.nan
            ds_slice[v_x][:] = u
            ds_slice[v_y][:] = v
            return ds_slice
        ds_g = self._obj.groupby("time")
        self._obj.update(
            ds_g.apply(_filter_median, v_x=v_x, v_y=v_y, **kwargs)
        )

    def get_uv_camera(self, dt=0.1, v_x="v_x", v_y="v_y"):
        """
        Returns row, column locations in the camera objective, and u (x-directional) and v (y-directional) vectors, scaled
        and transformed to the camera objective (i.e. vectors far away are smaller than closeby, and follow the river direction)
        applied on u and v, so that they plot in a geographical space. This is needed because the raster of PIV results
        is usually rotated geographically, so that water always flows from left to right in the grid. The results can be
        used to plot velocities in the camera perspective, e.g. overlayed on a background image directly from the camera.

        :param dt: float, time difference [s] used to scale the u and v velocities to a very small distance to project with
            default: 0.1, usually not needed to modify this.
        :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
        :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
        :return: 5 outputs: 4 np.ndarrays containing camera perspective column location, row location, transformed u and v
            velocity vectors (no unit) and the scalar velocities (m/s). Rotation is not needed because the transformed
            u and v components are already rotated to match the camera perspective. counter-clockwise rotation in radians.
        """
        # retrieve the backward transformation array
        M = self.camera_config.get_M(self.h_a, reverse=True)
        # get the shape of the original frames
        shape_y, shape_x = self.camera_shape
        xi, yi = np.meshgrid(self._obj.x, self._obj.y)
        # flip the y-coordinates to match the row order used by opencv
        yi = np.flipud(yi)

        x_moved, y_moved = xi + self._obj[v_x] * dt, yi + self._obj[v_y] * dt
        xp_moved, yp_moved = helpers.xy_to_perspective(x_moved.values, y_moved.values, self.camera_config.resolution, M)

        # convert row counts to start at the top of the frame instead of bottom
        yp_moved = shape_y - yp_moved

        # missing values end up at the top-left, replace these with nan
        yp_moved[yp_moved == shape_y] = np.nan  # ds["yp"].values[yp_moved == shape_y]
        xp_moved[xp_moved == 0] = np.nan  # ds["xp"].values[xp_moved == 0]

        u, v = xp_moved - self._obj["xp"], yp_moved - self._obj["yp"]
        s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        s.name = "radial_sea_water_velocity_away_from_instrument"
        return "xp", "yp", u, v, s

    def get_uv_geographical(self, v_x="v_x", v_y="v_y"):
        """
        Returns lon, lat coordinate names and u (x-directional) and v (y-directional) velocities, and a rotation that must be
        applied on u and v, so that they plot in a geographical space. This is needed because the raster of PIV results
        is usually rotated geographically, so that water always flows from left to right in the grid. The results can be
        directly forwarded to a plot function for velocities in a geographical map, e.g. overlayed on a background
        image, projected to lat/lon.

        :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
        :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
        :return: 6 outputs: 5 np.ndarrays containing longitude, latitude coordinates, u and v velocities and scalar velocity;
            and float with counter-clockwise rotation in radians, to be applied on u, v to plot in geographical space.

        """
        # select lon and lat variables as coordinates
        u = self._obj[v_x]
        v = -self._obj[v_y]
        s = (u ** 2 + v ** 2) ** 0.5
        aff = self.camera_config.transform
        theta = np.arctan2(aff.d, aff.a)
        s.name = "radial_sea_water_velocity_away_from_instrument"
        return "lon", "lat", u, v, s, theta

    def get_transect(self, x, y, z=None, crs=None, v_eff=True, xs="xs", ys="ys", distance=None, wdw=1):
        """
        Interpolate all variables to supplied x and y coordinates of a cross section. This function assumes that the grid
        can be rotated and that xs and ys are supplied following the projected coordinates supplied in
        "xs" and "ys" coordinate variables in ds. x-coordinates and y-coordinates that fall outside the
        domain of ds, are still stored in the result for further interpolation or extrapolation.
        Original coordinate values supplied are stored in coordinates "x", "y" and (if supplied) "z".

        :param self: xarray dataset
        :param x: tuple or list-like, x-coordinates on which interpolation should be done
        :param y: tuple or list-like, y-coordinates on which interpolation should be done
        :param z: tuple or list-like, z-coordinates on which interpolation should be done, defaults: None
        :param crs: coordinate reference system (e.g. EPSG code) in which x, y and z are measured, defaults to None,
            assuming crs is the same as crs of ds
        :param v_eff: bool, if True, effective velocity (perpendicular to cross-section) is also computed, default: True
        :param xs: str, name of variable that stores the x coordinates in the projection in which "x" is supplied,
            default: "xs"
        :param ys: str, name of variable that stores the y coordinates in the projection in which "y" is supplied
            default: "ys"
        :return: ds_points: xarray dataset, containing interpolated data at the supplied x and y coordinates
        """
        transform = helpers.affine_from_grid(self._obj[xs].values, self._obj[ys].values)
        if crs is not None:
            # transform coordinates of cross section
            x, y = helpers.xy_transform(x, y, crs_from=crs, crs_to=CRS.from_wkt(self.camera_config.crs))
        if distance is None:
            # interpret suitable sampling distance from grid resolution
            distance = np.abs(np.diff(self._obj.x)[0])
            # interpolate to a suitable set of points
        x, y, z, s = helpers.xy_equidistant(x, y, distance=distance, z=z)

        # make a cols and rows temporary variable
        coli, rowi = np.meshgrid(np.arange(len(self._obj["x"])), np.arange(len(self._obj["y"])))
        self._obj["cols"], self._obj["rows"] = (["y", "x"], coli), (["y", "x"], rowi)
        # compute rows and cols locations of coordinates (x, y)
        rows, cols = rasterio.transform.rowcol(
            transform,
            list(x),
            list(y),
            op=float
        )  # ensure we get rows and columns in fractions instead of whole numbers
        rows, cols = np.array(rows), np.array(cols)

        # select x and y coordinates from axes
        idx = np.all(
            np.array([cols >= 0, cols < len(self._obj["x"]), rows >= 0, rows < len(self._obj["y"])]),
            axis=0,
        )
        # compute transect coordinates in the local grid coordinate system (can be outside the grid)
        f_x = interp1d(np.arange(0, len(self._obj["x"])), self._obj["x"], fill_value="extrapolate")
        f_y = interp1d(np.arange(0, len(self._obj["y"])), self._obj["y"], fill_value="extrapolate")
        _x = f_x(cols)
        _y = f_y(rows)

        # covert local coordinates to DataArray
        _x = xr.DataArray(list(_x), dims="points")
        _y = xr.DataArray(list(_y), dims="points")

        # interpolate velocities over points
        if wdw == 0:
            ds_points = self._obj.interp(x=_x, y=_y)
        else:
            # collect points within a stride, collate and analyze for outliers
            ds_wdw = xr.concat([self._obj.shift(x=x_stride, y=y_stride) for x_stride in range(-wdw, wdw + 1) for y_stride in
                                range(-wdw, wdw + 1)], dim="stride")
            # use the median to prevent a large influence of serious outliers
            ds_effective = ds_wdw.median(dim="stride", keep_attrs=True)
            ds_points = ds_effective.interp(x=_x, y=_y)
        if np.isnan(ds_points["v_x"].mean(dim="time")).all():
            raise ValueError(
                "No valid velocimetry points found over bathymetry. Check if the bethymetry is within the camera objective")
        # add the xcoords and ycoords (and zcoords if available) originally assigned so that even points outside the grid covered by ds can be
        # found back from this dataset
        ds_points = ds_points.assign_coords(xcoords=("points", list(x)))
        ds_points = ds_points.assign_coords(ycoords=("points", list(y)))
        ds_points = ds_points.assign_coords(scoords=("points", list(s)))
        if z is not None:
            ds_points = ds_points.assign_coords(zcoords=("points", list(z)))
        # convert to a Transect object
        ds_points = xr.Dataset(ds_points, attrs=ds_points.attrs)
        if v_eff:
            # add the effective velocity, perpendicular to cross section direction
            ds_points.transect.vector_to_scalar()
        return ds_points

    def plot(
            self,
            ax=None,
            scalar=True,
            quiver=True,
            mode="local",
            scalar_kwargs={},
            quiver_kwargs={},
            v_x="v_x",
            v_y="v_y",
            cbar=True,
            cbar_fontsize=15
    ):
        """
        plot velocimetry results. These can be plotted as scalar values (i.e. a mesh) or as quivers, or both by setting
        the inputs 'scalar' and 'quiver' to True or False. Plotting can be done in three modes:
        - "local": a simple planar view plot, with a local coordinate system in meters, with the top-left coordinate
          being the 0, 0 point, and ascending coordinates towards the right and bottom.
        - "geographical": a geographical plot, requiring the package `cartopy`, the results are plotted on a
          geographical axes, so that combinations with tile layers such as OpenStreetMap, or shapefiles can be made.
        - "camera": i.e. seen from the camera perspective. This is the most intuitive view for end users.

        :param ax: pre-defined axes object. If not set, a new axes will be prepared. In case `mode=="geographical"`, a
            cartopy GeoAxes needs to be provided, or will be made in case ax is not set. If an axes with background
            frame is provided (made through frames.plot) then the background must be plotted in the same mode as
            selected here.
        :param scalar: boolean, if set to True, velocities are plotted as scalar values in a mesh (default: True)
        :param quiver: boolean, if set to True, velocities are plotted as quiver (i.e. arrows). In case scalar is also
            True, quivers will be plotted with a single color (defined in `quiver_kwargs`), if not, the scalar values
            are used to color the arrows.
        :param mode: can be "local", "geographical", or "camera". For "geographical" a velocimetry result that contains
            "lon" and "lat" coordinates must be provided (i.e. produced with known CRS for control points).
        :param scalar_kwargs: dict, plotting parameters to be passed to matplotlib.pyplot.pcolormesh, for plotting
            scalar values.
        :param quiver_kwargs: dict, plotting parameters to be passed to matplotlib.pyplot.quiver, for plotting quiver
            arrows.
        :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
        :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
        :param cbar: bool, optional, define if colorbar should be included (default: True)
        :param cbar_fontsize: fontsize to use for the colorbar title (fontsize of tick labels will be made slightly
            smaller).
        :return: ax, axes object resulting from this function.
        """

        if len(self._obj[v_x].shape) > 2:
            raise OverflowError(
                f'Dataset\'s variables should only contain 2 dimensions, this dataset '
                f'contains {len(self._obj[v_x].shape)} dimensions. Reduce this by applying a reducer or selecting a time step. '
                f'Reducing can be done e.g. with ds.mean(dim="time", keep_attrs=True) or slicing with ds.isel(time=0)'
            )
        assert (scalar or quiver), "Either scalar or quiver should be set tot True, nothing to plot"
        assert mode in ["local", "geographical", "camera"], 'Mode must be "local", "geographical" or "camera"'
        if mode == "local":
            x = "x"
            y = "y"
            theta = 0.
            u = self._obj[v_x]
            v = -self._obj[v_y]
            s = (u ** 2 + v ** 2) ** 0.5
        elif mode == "geographical":
            # import some additional packages
            import cartopy.crs as ccrs
            # add transform for GeoAxes
            scalar_kwargs["transform"] = ccrs.PlateCarree()
            quiver_kwargs["transform"] = ccrs.PlateCarree()
            x, y, u, v, s, theta = self.get_uv_geographical()
        else:
            # mode is camera
            x, y, u, v, s = self.get_uv_camera()
            theta = 0.
        # prepare an axis for the provided mode
        ax = plot_orc.prepare_axes(ax=ax, mode=mode)
        f = ax.figure  # handle to figure
        if quiver:
            vmin = None
            vmax = None
            if "vmin" in quiver_kwargs:
                vmin = quiver_kwargs["vmin"]
                del quiver_kwargs["vmin"]
            if "vmax" in quiver_kwargs:
                vmax = quiver_kwargs["vmax"]
                del quiver_kwargs["vmax"]

            if scalar:
                p = plot_orc.quiver(
                    ax,
                    self._obj[x].values,
                    self._obj[y].values,
                    *[v.values for v in helpers.rotate_u_v(u, v, theta)],
                    None,
                    **quiver_kwargs
                )
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
                p = plot_orc.quiver(
                    ax,
                    self._obj[x].values,
                    self._obj[y].values,
                    *[v.values for v in helpers.rotate_u_v(u, v, theta)],
                    s,
                    norm=norm,
                    **quiver_kwargs
                )
        if scalar:
            # plot the scalar velocity value as grid, return mappable
            p = ax.pcolormesh(s[x], s[y], s, zorder=2, **scalar_kwargs)
        if mode == "geographical":
            ax.set_extent(
                [self._obj[x].min() - 0.0001, self._obj[x].max() + 0.0001, self._obj[y].min() - 0.0001, self._obj[y].max() + 0.0001],
                crs=ccrs.PlateCarree())
        # else:
        #     ax.axis('equal')
        if cbar:
            cb = plot_orc.cbar(ax, p, size=cbar_fontsize)
        return ax

    def replace_outliers(self, v_x="v_x", v_y="v_y", stride=1, max_iter=1, inplace=False):
        """
        Replace missing values using neighbourhood operators. Use this with caution as it creates data. If many samples
        in time are available to derive a mean or median velocity from, consider using a reducer on those samples
        instead of a spatial infilling method such as suggested here.

        :param self: xarray Dataset, containing velocity vectors as [time, y, x]
        :param v_x: str, name of x-directional velocity
        :param v_y: str, name of y-directional velocity
        :param stride: int, stride used to determine relevant neighbours
        :param max_iter: number of iterations for replacement
        :return: xr.Dataset, containing filtered velocities
        """
        # TO-DO: make replacement decision dependent on amount of non-NaN values in neighbourhood
        u, v = self._obj[v_x].values, self._obj[v_y].values
        for n in range(max_iter):
            u_move = helpers.neighbour_stack(u, stride=stride)
            v_move = helpers.neighbour_stack(v, stride=stride)
            # compute mean
            u_mean = np.nanmean(u_move, axis=0)
            v_mean = np.nanmean(v_move, axis=0)
            u[np.isnan(u)] = u_mean[np.isnan(u)]
            v[np.isnan(v)] = v_mean[np.isnan(v)]
            # all values with stride distance from edge have to be made NaN
            u[0:stride, :] = np.nan;
            u[-stride:, :] = np.nan;
            u[:, 0:stride] = np.nan;
            u[:, -stride:] = np.nan
            v[0:stride, :] = np.nan;
            v[-stride:, :] = np.nan;
            v[:, 0:stride] = np.nan;
            v[:, -stride:] = np.nan
        if inplace:
            self._obj[v_x][:] = u
            self._obj[v_y][:] = v
        else:
            ds = self._obj.copy(deep=True)
            ds[v_x][:] = u
            ds[v_y][:] = v
            return ds

    def set_encoding(self):
        for k in const.ENCODE_VARS:
            self._obj[k].encoding = const.ENCODING_PARAMS
