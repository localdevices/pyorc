import copy
import numpy as np
import rasterio
import xarray as xr
import warnings

from pyproj import CRS
from scipy.interpolate import interp1d
from .orcbase import ORCBase
from .plot import _Velocimetry_PlotMethods
from .. import helpers, const
from xarray.core import utils

@xr.register_dataset_accessor("velocimetry")
class Velocimetry(ORCBase):
    """Velocimetry functionalities that can be applied on ``xarray.Dataset``"""
    def __init__(self, xarray_obj):
        """Initialize a velocimetry ``xarray.Dataset``

        Parameters
        ----------
        xarray_obj: xr.Dataset
            velocimetry data fields (from ``pyorc.Frames.get_piv``)
        """
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
        """Masks values using several filters that use temporal variations or comparison as basis.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (default: "v_x")
        v_y : str, optional
            name of y-directional velocity (default: "v_y")
        filter_std : boolean, optional
            if True (default), filtering on variance is applied
        filter_std : boolean, optional
            if True (default), filtering on variance is applied
        filter_angle : boolean, optional
            if True (default), filtering on angles is applied
        filter_velocity : boolean
            if True (default, filtering on velocity is applied)
        filter_corr : boolean
            if True (default), filtering on correlation is applied
        filter_neighbour : dict, optional
            if True (default), neighbourhood filtering is applied
        kwargs_std : dict, optional
            set of key-word arguments to pass on to filter_temporal_std
        kwargs_angle : dict, optional
            set of key-word arguments to pass on to filter_temporal_angle
        kwargs_velocity : dict, optional
            set of key-word arguments to pass on to filter_temporal_velocity
        kwargs_corr : dict, optional
            set of key-word arguments to pass on to filter_temporal_corr
        kwargs_neighbour : dict, optional
            set of key-word arguments to pass on to filter_temporal_neighbour
        inplace : boolean
            If set, values are replaced instead of returning a new Dataset

        Returns
        -------
        ds_filter : xr.Dataset
            temporally filtered velocity vectors as [time, y, x]

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
        """filters on the expected angle. The function filters points entirely where the mean angle over time
        deviates more than input parameter angle_bounds (in radians). The function also filters individual
        estimates in time, in case the user wants this (filter_per_timestep=True), in case the angle on
        a specific time step deviates more than the defined amount from the average.
        note: this function does not work appropriately, if the expected angle (+/- anglebounds) are within
        range of zero, as zero is the same as 2*pi. This exception may be resolved in the future if necessary.

        Parameters
        ----------
        v_x : str
            name of x-directional velocity (Default value = "v_x")
        v_y : str
            name of y-directional velocity (Default value = "v_y")
        angle_expected : float
            angle (0-2*pi), measured clock-wise from vertical upwards direction, expected
            in the velocities, default: 0.5*np.pi (meaning from left to right)
        angle_tolerance : float (0-2*pi)
            maximum deviation from expected angle allowed. (Default value = 0.25 * np.pi)
        filter_per_timestep : boolean
            if set to True (default), tolerances are also checked per individual time step

        Returns
        -------
        ds_filter : xr.Dataset
            angle filtered velocity vectors as [time, y, x]

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
        """Masks values if neighbours over a certain rolling length before and after, have a
        significantly higher velocity than value under consideration, measured by tolerance.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (Default: "v_x")
        v_y : str, optional
            name of y-directional velocity (Default: "v_y")
        roll : int, optional
            amount of time steps in rolling window (centred) (default: 5)
        tolerance : float (0-1)
            relative acceptable velocity of maximum found within rolling window (default: .5)

        Returns
        -------
        type
            xr.Dataset, containing time-neighbour filtered velocity vectors as [time, y, x]

        """
        s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        s_roll = s.fillna(0.).rolling(time=roll, center=True).max()
        self._obj[v_x] = self._obj[v_x].where(s > tolerance * s_roll)
        self._obj[v_y] = self._obj[v_y].where(s > tolerance * s_roll)
        # return ds

    def filter_temporal_std(
            self, v_x="v_x", v_y="v_y", tolerance_sample=1.0, tolerance_var=5., mode="or"):
        """Masks values if they deviate more than x standard deviations from the mean.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (Default: "v_x")
        v_y : str, optional
            name of y-directional velocity (Default: "v_y")
        tolerance :  float
            amount of standard deviations tolerance
        tolerance_sample :
             (Default value = 1.0)
        tolerance_var :
             (Default value = 5.)
        mode : str
             can be "and" or "or" (default). If "or" ("and"), then only one (both) of two vector components need(s) to
             be within tolerance.

        Returns
        -------
        ds_filter : xr.Dataset
            standard deviation filtered velocity vectors as [time, y, x]
        """

        # s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_std = self._obj[v_x].std(dim="time")
            y_std = self._obj[v_y].std(dim="time")
            x_mean = self._obj[v_x].mean(dim="time")
            y_mean = self._obj[v_y].mean(dim="time")
            x_var = np.abs(x_std / x_mean)
            y_var = np.abs(y_std / y_mean)
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

        # return ds

    def filter_temporal_velocity(self, v_x="v_x", v_y="v_y", s_min=0.1, s_max=5.0):
        """Masks values if the velocity scalar lies outside a user-defined valid range.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (Default: "v_x")
        v_y : str, optional
            name of y-directional velocity (Default: "v_y")
        s_min : float, optional
            minimum scalar velocity [m s-1] (default: 0.1)
        s_max : float, optional
            maximum scalar velocity [m s-1] (default: 5.)

        Returns
        -------
        ds_filter : xr.Dataset
            velocity-range filtered velocity vectors as [time, y, x]
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
        """Masks values with a too low correlation.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (Default: "v_x")
        v_y : str, optional
            name of y-directional velocity (Default: "v_y")
        corr : str
            name of correlation variable (default: "corr")
        tolerance : float (0-1)
            tolerance for correlation value (default: 0.1). If correlation is lower than tolerance, it is masked

        Returns
        -------
        ds_filter : xr.Dataset
            correlation filtered velocity vectors as [time, y, x]
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
        """Masks velocity values on a number of spatial filters.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (default: "v_x")
        v_y : str, optional
            name of y-directional velocity (default: "v_y")
        filter_nan : boolean, optional
            if True (default), filtering on nans is applied
        filter_median : boolean, optional
            if True (default), filtering on median deviations is applied
        kwargs_nan : dict, optional
            dict, keyword arguments to pass to filter_spatial_nan
        kwargs_median : dict, optional
            keyword arguments to pass to filter_spatial_median
        inplace : boolean, optional
            If True (default: False), values are replaced instead of returning a new Dataset

        Returns
        -------
        ds_filter : xr.Dataset
            spatially filtered velocity vectors as [time, y, x]

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

    def filter_spatial_nan(self, v_x="v_x", v_y="v_y", tolerance=0.3, wdw=1, missing=-9999.):
        """Masks values if their surrounding neighbours (inc. value itself) contain too many NaN. Meant to remove isolated
        velocity estimates.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (default: "v_x")
        v_y : str, optional
            name of y-directional velocity (default: "v_y")
        tolerance : float, optional
            amount of NaNs in search window measured as a fraction of total amount of values [0-1] (default: 0.3)
        wdw : int, optional
            window length used to determine relevant neighbours. 1 means that 1 neighbour on all sides is considered
            (default: 1)
        missing : float, optional
            a temporary missing value, used to be able to convolve NaNs, normally not needed to change (default -9999.)

        Returns
        -------
        ds_filter: xr.Dataset
            NaN filtered velocity vectors as [time, y, x]

        """
        def _filter_nan(ds_slice, v_x="v_x", v_y="v_y", tolerance=0.3, wdw=1, missing=-9999.):
            """
            internal function, see main function
            """
            # u, v = ds[v_x], ds[v_y]
            u, v = ds_slice[v_x].values, ds_slice[v_y].values
            u_move = helpers.neighbour_stack(u.copy(), stride=wdw, missing=missing)
            # replace missings by Nan
            nan_frac = np.float64(np.isnan(u_move)).sum(axis=0) / float(len(u_move))
            u[nan_frac > tolerance] = np.nan
            v[nan_frac > tolerance] = np.nan
            ds_slice[v_x][:] = u
            ds_slice[v_y][:] = v
            return ds_slice
        ds_g = self._obj.groupby("time")
        self._obj.update(
            ds_g.map(
                _filter_nan,
                v_x=v_x,
                v_y=v_y,
                tolerance=tolerance,
                wdw=wdw,
                missing=missing
            )
        )

    def filter_spatial_median(self, v_x="v_x", v_y="v_y", tolerance=0.7, wdw=1, missing=-9999.):
        """Masks values when their value deviates more than x standard deviations from the median of its neighbours
        (inc. itself).

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (default: "v_x")
        v_y : str, optional
            name of y-directional velocity (default: "v_y")
        tolerance : float, optional
            amount of standard deviations tolerance (default: 0.7)
        wdw : int, optional
            window used to determine relevant neighbours
        missing : float, optional
            a temporary missing value, used to be able to convolve NaNs

        Returns
        -------
        ds_filter : xr.Dataset
            std filtered velocity vectors as [time, y, x]

        """
        def _filter_median(ds_slice, v_x="v_x", v_y="v_y", tolerance=0.7, wdw=1, missing=-9999.):
            """
            internal function, see main method
            """
            u, v = ds_slice[v_x].values, ds_slice[v_y].values
            # s = (u ** 2 + v ** 2) ** 0.5
            u_move = helpers.neighbour_stack(copy.deepcopy(u), stride=wdw, missing=missing)
            v_move = helpers.neighbour_stack(copy.deepcopy(v), stride=wdw, missing=missing)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)

                # replace missings by Nan
                u_median = np.nanmedian(u_move, axis=0)
                v_median = np.nanmedian(v_move, axis=0)
                # now filter points that are very far off from the median
                u_filter = np.abs(u - u_median) / u_median > tolerance
                v_filter = np.abs(v - v_median) / v_median > tolerance
            filter = np.any([u_filter, v_filter], axis=0)
            u[filter] = np.nan
            v[filter] = np.nan
            ds_slice[v_x][:] = u
            ds_slice[v_y][:] = v
            return ds_slice
        ds_g = self._obj.groupby("time")
        self._obj.update(
            ds_g.map(_filter_median, v_x=v_x, v_y=v_y, tolerance=tolerance, wdw=wdw, missing=missing)
        )


    def get_transect(
            self, x, y, z=None, s=None,
            crs=None,
            v_eff=True,
            xs="xs",
            ys="ys",
            distance=None,
            wdw=1,
            rolling=None,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ):
        """Interpolate all variables to supplied x and y coordinates of a cross section. This function assumes that the
        grid can be rotated and that xs and ys are supplied following the projected coordinates supplied in
        "xs" and "ys" coordinate variables in ds. x-coordinates and y-coordinates that fall outside the
        domain of ds, are still stored in the result for further interpolation or extrapolation.
        Original coordinate values supplied are stored in coordinates "x", "y" and (if supplied) "z".
        Time series are transformed to set quantiles.

        Parameters
        ----------
        x : tuple or list-like
            x-coordinates on which interpolation should be done
        y : tuple or list-like
            y-coordinates on which interpolation should be done
        z : tuple or list-like
            z-coordinates on which interpolation should be done, defaults: None
        s : tuple or list-like
            distance from bank coordinates on which interpolation should be done, defaults: None
            if set, these distances will be precisely respected, and not interpolated. ``distance`` will be ignored.

        crs : int, dict or str, optional
            coordinate reference system (e.g. EPSG code) in which x, y and z are measured (default: None),
            None assumes crs is the same as crs of xr.Dataset.
        v_eff : boolean, optional
            if True (default), effective velocity (perpendicular to cross-section) is also computed
        xs : str, optional
            name of variable that stores the x coordinates in the projection in which "x" is supplied (default: "xs")
        ys : str, optional
            name of variable that stores the y coordinates in the projection in which "y" is supplied (default: "ys")
        distance : float, optional
            sampling distance over the cross-section in [m]. the bathymetry points will be interpolated to match this
            distance. If not set, the distance will be estimated from the velocimetry grid resolution. (default: None)
        wdw : int, window size to use for sampling the velocity. zero means, only cell itself, 1 means 3x3 window.
            (default: 1)
        rolling : int, optional
            if set other than None (default), a rolling mean over time is applied, before deriving quantile estimates.
        quantiles : list of floats (0-1), optional
            list of quantiles to return (default: [0.05, 0.25, 0.5, 0.75, 0.95]).

        Returns
        -------
        ds_points: xr.Dataset
            interpolated data at the supplied x and y coordinates over quantiles
        """
        transform = helpers.affine_from_grid(self._obj[xs].values, self._obj[ys].values)
        if crs is not None:
            # transform coordinates of cross section
            x, y = zip(*helpers.xyz_transform(
                list(zip(*(x, y))),
                crs_from=crs,
                crs_to=CRS.from_wkt(self.camera_config.crs)
            ))
            x, y = list(x), list(y)
        if s is None:
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
            # use the median (not mean) to prevent a large influence of serious outliers
            ds_effective = ds_wdw.mean(dim="stride", keep_attrs=True)
            ds_points = ds_effective.interp(x=_x, y=_y)
        if np.isnan(ds_points["v_x"].mean(dim="time")).all():
            raise ValueError(
                "No valid velocimetry points found over bathymetry. Check if the bathymetry is within the camera objective")
        # add the xcoords and ycoords (and zcoords if available) originally assigned so that even points outside the grid covered by ds can be
        # found back from this dataset
        ds_points = ds_points.assign_coords(xcoords=("points", list(x)))
        ds_points = ds_points.assign_coords(ycoords=("points", list(y)))
        ds_points = ds_points.assign_coords(scoords=("points", list(s)))
        if z is not None:
            ds_points = ds_points.assign_coords(zcoords=("points", list(z)))
        # add mean angles to dataset
        alpha = helpers.xy_angle(ds_points["x"], ds_points["y"])
        flow_dir = alpha - 0.5 * np.pi
        ds_points["v_dir"] = (("points"), flow_dir)
        ds_points["v_dir"].attrs = {
            "standard_name": "river_flow_angle",
            "long_name": "Angle of river flow in radians from North",
            "units": "rad"
        }
        # convert to a Transect object
        ds_points = xr.Dataset(ds_points, attrs=ds_points.attrs)
        if rolling is not None:
            ds_points = ds_points.rolling(time=rolling, min_periods=1).mean()
        ds_points = ds_points.quantile(quantiles, dim="time", keep_attrs=True)
        if v_eff:
            # add the effective velocity, perpendicular to cross section direction
            ds_points.transect.vector_to_scalar()
        return ds_points

    # add .plot as group of methods, UncachedAccessor ensures that Velocimetry is passed as object
    plot = utils.UncachedAccessor(_Velocimetry_PlotMethods)

    def replace_outliers(self, v_x="v_x", v_y="v_y", wdw=1, max_iter=1, inplace=False):
        """Replace missing values using neighbourhood operators. Use this with caution as it creates data. If many samples
        in time are available to derive a mean or median velocity from, consider using a reducer on those samples
        instead of a spatial infilling method such as suggested here.

        Parameters
        ----------
        v_x : str, optional
            name of x-directional velocity (default: "v_x")
        v_y : str, optional
            name of y-directional velocity (default: "v_y")
        wdw : int, optional
            window size used to determine relevant neighbours (default: 1), 1 means a 3x3 window (1 neighbour)
        max_iter : int, optional
            number of iterations for replacement (default: 1)
        inplace : boolean, optional
            replace values instead of returning new xr.Dataset (default: False)

        Returns
        -------
        ds_replaced : xr.Dataset
            velocities with replacements [time, x, y]

        """
        # TO-DO: make replacement decision dependent on amount of non-NaN values in neighbourhood
        u, v = copy.deepcopy(self._obj[v_x].values), copy.deepcopy(self._obj[v_y].values)
        for n in range(max_iter):
            u_move = helpers.neighbour_stack(u, stride=wdw)
            v_move = helpers.neighbour_stack(v, stride=wdw)
            # compute mean
            u_mean = np.nanmean(u_move, axis=0)
            v_mean = np.nanmean(v_move, axis=0)
            u[np.isnan(u)] = u_mean[np.isnan(u)]
            v[np.isnan(v)] = v_mean[np.isnan(v)]
            # all values with stride distance from edge have to be made NaN
            u[0:wdw, :] = np.nan;
            u[-wdw:, :] = np.nan;
            u[:, 0:wdw] = np.nan;
            u[:, -wdw:] = np.nan
            v[0:wdw, :] = np.nan;
            v[-wdw:, :] = np.nan;
            v[:, 0:wdw] = np.nan;
            v[:, -wdw:] = np.nan
        if inplace:
            self._obj[v_x][:] = u
            self._obj[v_y][:] = v
        else:
            ds = self._obj.copy(deep=True)
            ds[v_x][:] = u
            ds[v_y][:] = v
            return ds

    def set_encoding(self, enc_pars=const.ENCODING_PARAMS):
        """Set encoding parameters for all typical variables in a velocimetry dataset. This reduces the required storage
        for this dataset significantly, when stored to disk in e.g. a netcdf file using ``xarray.Dataset.to_netcdf``.

        Parameters
        ----------
        enc_pars : dict of dicts, optional
            per variable, a dict containing encoding parameters. When called without input, a standard set of encoding
            parameters is used that compresses well. (Default value = const.ENCODING_PARAMS)

        Returns
        -------

        """
        for k in const.ENCODE_VARS:
            self._obj[k].encoding = enc_pars
