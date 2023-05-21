import numpy as np
import rasterio
import xarray as xr
import warnings

from pyproj import CRS
from scipy.interpolate import interp1d
from .orcbase import ORCBase
from .plot import _Velocimetry_PlotMethods
from .mask import _Velocimetry_MaskMethods
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

    @property
    def is_velocimetry(self):
        """
        Checks if the data contained in the object seems to be velocimetry data by checking naming of dims
        and available variables.

        Returns
        -------
        is_velocimetry : bool
            If True, the dataset likely contains velocimetry data

        """
        # check for dims, difference between available and allowed should be zero in length
        unknown_dims = set(self._obj.dims).difference(set(["time", "y", "x"]))
        if len(unknown_dims) != 0:
            print(f"Unknown dimension(s) found: {unknown_dims}")
            return False
        missed_dims = set(["y", "x"]).difference(set(self._obj.dims))
        if len(missed_dims) != 0:
            print(f"Dimensions missing: {missed_dims}")
            return False
        # check for
        missed_vars = set(const.ENCODE_VARS).difference(set(self._obj.data_vars))
        if len(missed_vars) != 0:
            print(f"Variables missing: {missed_vars}")
            return False
        # check for available metadata
        if not(hasattr(self._obj, "camera_config")):
            print("camera_config metadata is missing")
            return False
        return True

    mask = utils.UncachedAccessor(_Velocimetry_MaskMethods)

    def get_transect(
            self, x, y, z=None, s=None,
            crs=None,
            v_eff=True,
            xs="xs",
            ys="ys",
            distance=None,
            wdw=1,
            wdw_x_min=None,
            wdw_x_max=None,
            wdw_y_min=None,
            wdw_y_max=None,
            rolling=None,
            tolerance=0.5,
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
        wdw : int, optional
            window size to use for sampling the velocity. zero means, only cell itself, 1 means 3x3 window.
            (default: 1) wdw is used to fill wdw_x_min and wdwd_y_min with its negative (-wdw) value, and wdw_y_min and
            wdw_y_max with its positive value, to create a sampling window.
        wdw_x_min : int, optional
            window size in negative x-direction of grid (must be negative), overrules wdw in negative x-direction if set
        wdw_x_max : int, optional
            window size in positive x-direction of grid, overrules wdw in positive x-direction if set
        wdw_y_min : int, optional
            window size in negative y-direction of grid (must be negative), overrules wdw in negative y-direction if set
        wdw_y_max : int, optional
            window size in positive y-direction of grid, overrules wdw in positive x-direction if set.
        tolerance : float (0-1), optional
            tolerance on the required amount of sampled data in the window defined by wdw and/or wdw_x_min, wdw_x_max,
            wdw_y_min and wdw_y_max (if set). At least this fraction of cells each time step must have a data value
            to return a value. Otherwise the location is given a nan as value.
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
            # transform coordinates of cross-section
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
            ds_points = self._obj.interp(x=_x, y=_y, method="nearest")
        else:
            # collect points within a stride, collate and analyze for outliers
            ds_wdw = helpers.stack_window(
                self._obj,
                wdw=wdw,
                wdw_x_min=wdw_x_min,
                wdw_x_max=wdw_x_max,
                wdw_y_min=wdw_y_min,
                wdw_y_max=wdw_y_max

            )
            # use the median (not mean) to prevent a large influence of serious outliers
            missing_tolerance = ds_wdw.mean(dim="time").count(dim="stride") > tolerance * len(ds_wdw.stride)
            # missing_tolerance = ds_wdw.count(dim="stride") > tolerance*len(ds_wdw.stride)
            ds_effective = ds_wdw.median(dim="stride", keep_attrs=True)
            # remove velocities that are too few in samples
            ds_effective = ds_effective.where(missing_tolerance)
            # scipy does not tolerate np.float32 since scipy=1.10.0, so first convert to np.float64
            for var in ds_effective:
                ds_effective[var] = ds_effective[var].astype(np.float64)
            for coord in ds_effective.coords:
                ds_effective[coord] = ds_effective[coord].astype(np.float64)

            ds_points = ds_effective.interp(x=_x, y=_y)
        if np.isnan(ds_points["v_x"].mean(dim="time")).all():
            warnings.warn(
                "No valid velocimetry points found over bathymetry. Check if the bathymetry is within the camera "
                "objective or anything is visible in objective. "
            )
        # add the xcoords and ycoords (and zcoords if available) originally assigned so that even points outside the
        # grid covered by ds can be found back from this dataset
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds_points = ds_points.quantile(quantiles, dim="time", keep_attrs=True)
        if v_eff:
            # add the effective velocity, perpendicular to cross-section direction
            ds_points.transect.vector_to_scalar()
        return ds_points

    # add .plot as group of methods, UncachedAccessor ensures that Velocimetry is passed as object
    plot = utils.UncachedAccessor(_Velocimetry_PlotMethods)

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
