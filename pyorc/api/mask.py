import copy
import functools
import numpy as np
import warnings
import xarray as xr

from pyorc.const import v_x, v_y, s2n, corr
from pyorc import helpers


commondoc = """
    Returns
    -------
    mask : xr.DataArray
        mask applicable to input dataset with ``ds.velocimetry.filter(mask)``.
        If ``inplace=True``, the dataset will be returned masked with ``mask``.
        
"""
def _base_mask(time_allowed=False, time_required=False):
    """
    wrapper generator for creating generalized structure masking methods for velocimetry

    Parameters
    ----------
    time_allowed : bool, optional
        If set, the dimension "time" is allowed, if not set, mask method can only be applied on datasets without "time"
    time_required
        If set, the dimension "time" is required, if not set, mask method does not require dimension "time" in dataset.

    Returns
    -------
    func : function
        masking method, decorated with standard procedures
    """
    def decorator_func(mask_func):
        mask_func.__doc__ = f"{mask_func.__doc__}{commondoc}"
        # wrap function so that it takes over the docstring and is seen as integral part of the class
        @functools.wraps(mask_func)
        def wrapper_func(ref, inplace=False, reduce_time=False, *args, **kwargs):
            # check if obj seems to contain velocimetry data
            if reduce_time and "time" in ref._obj:
                ds = ref._obj.mean(dim="time", keep_attrs=True)
            else:
                ds = ref._obj
            if not(ds.velocimetry.is_velocimetry):
                raise AssertionError("Dataset is not a valid velocimetry dataset")
            if time_required:
                # then automatically time is also allowed
                # time_allowed = True
                if not("time" in ds.dims):
                    raise AssertionError(
                f'This mask requires dimension "time". The dataset does not contain dimension "time" or you have set'
                f'reduce_time=True. Apply this mask without applying any reducers in time.'
            )
            # if not(time_allowed) and not(time_required):
            #     if "time" in ref._obj.dims:
            #         raise AssertionError(
            #     f'This mask can only work without dimension "time". The dataset contains dimension "time".'
            #     f'Reduce this by applying a reducer or selecting a time step. '
            #     f'Reducing can be done e.g. with ds.mean(dim="time", keep_attrs=True) or slicing with ds.isel(time=0)'
            # )
            if time_required:
                if not("time" in ds):
                    raise AssertionError(
                f'This mask requires dimension "time". The dataset does not contain dimension "time".'
                f'Apply this mask before applying any reducers in time.'
            )
            if not(time_allowed or time_required) and "time" in ds:
                # function must be applied per time step
                mask = ds.groupby("time").map(mask_func, **kwargs)
            else:
                # apply the wrapped mask function as is
                mask = mask_func(ds, **kwargs)
            # apply mask if inplace
            if inplace:
                # set the _obj data points
                for var in ref._obj.data_vars:
                    ref._obj[var] = ref._obj[var].where(mask)
            return mask
        return wrapper_func
    return decorator_func

class _Velocimetry_MaskMethods:
    """
    Enables use of ``ds.velocimetry.filter`` functions as attributes on a Dataset containing velocimetry results.
    For example, ``Dataset.velocimetry.filter.minmax``. This will return either the dataset with filtered data using
    For example, ``Dataset.velocimetry.filter.minmax``. This will return either the dataset with filtered data using
    the ``minmax`` filter when ``inplace=True`` or the mask that should be applied to filter when ``inplace=False``
    (default). ds.velocimetry.filter([mask1, mask2, ...]) applies the provided filters in the list of filters on
    the dataset by first combining all masks into one, and then applying that mask on the dataset
    """
    def __init__(self, velocimetry):
        # make the original dataset also available on the plotting object
        self.velocimetry = velocimetry
        self._obj = velocimetry._obj
        # Add to class _FilterMethods

    def __call__(self, mask, inplace=False, *args, **kwargs):
        """
        Parameters
        ----------
        mask : xr.DataArray or list of xr.DataArrays
            mask(s) to be applied on dataset, can have mix of y, x and time y, x dimensions
        *args :
        **kwargs :

        Returns
        -------
        ds : xr.Dataset
            Dataset containing filtered velocimetry results
        """
        if not(isinstance(mask, list)):
            # combine masks
            mask = [mask]
        if inplace:
            for m in mask:
                self._obj[v_x] = self._obj[v_x].where(m)
                self._obj[v_y] = self._obj[v_y].where(m)
                self._obj[corr] = self._obj[corr].where(m)
                self._obj[s2n] = self._obj[s2n].where(m)
        else:
            ds = copy.deepcopy(self._obj)
            for m in mask:
                ds[v_x] = ds[v_x].where(m)
                ds[v_y] = ds[v_y].where(m)
                ds[corr] = ds[corr].where(m)
                ds[s2n] = ds[s2n].where(m)
            return ds
    @_base_mask(time_allowed=True)
    def minmax(self, s_min=0.1, s_max=5.):
        """
    Masks values if the velocity scalar lies outside a user-defined valid range.

    Parameters
    ----------
    s_min : float, optional
        minimum scalar velocity [m s-1] (default: 0.1)
    s_max : float, optional
        maximum scalar velocity [m s-1] (default: 5.)

        """
        s = (self[v_x] ** 2 + self[v_y] ** 2) ** 0.5
        # create filter
        mask = (s > s_min) & (s < s_max)
        return mask

    @_base_mask(time_allowed=True)
    def angle(self, angle_expected=0.5 * np.pi, angle_tolerance=0.25 * np.pi):
        """
    filters on the expected angle. The function filters points entirely where the mean angle over time
    deviates more than input parameter angle_bounds (in radians). The function also filters individual
    estimates in time, in case the user wants this (filter_per_timestep=True), in case the angle on
    a specific time step deviates more than the defined amount from the average.
    note: this function does not work appropriately, if the expected angle (+/- anglebounds) are within
    range of zero, as zero is the same as 2*pi. This exception may be resolved in the future if necessary.

    Parameters
    ----------
    angle_expected : float
        angle (0 - 2*pi), measured clock-wise from vertical upwards direction, expected
        in the velocities, default: 0.5*np.pi (meaning from left to right in the x, y coordinate system)
    angle_tolerance : float (0 - 2*pi)
        maximum deviation from expected angle allowed (default: 0.25 * np.pi).
        """
        angle = np.arctan2(self[v_x], self[v_y])
        mask = np.abs(angle - angle_expected) < angle_tolerance
        return mask

    @_base_mask(time_required=True)
    def count(self, tolerance=0.33):
        """
    Masks locations with a too low amount of valid velocities in time, measured by fraction with ``tolerance``.
    Usually applied *after* having applied several other filters.

    Parameters
    ----------
    tolerance : float (0-1)
        tolerance for fractional amount of valid velocities after all filters. If less than the fraction is
        available, the entire velocity will be set to missings.
        """
        mask = self[v_x].count(dim="time") > tolerance * len(self.time)
        return mask


    @_base_mask(time_allowed=True)
    def corr(self, tolerance=0.1):
        """
    Masks values with too low correlation

    Parameters
    ----------
    tolerance : float (0-1)
        tolerance for correlation value (default: 0.1). If correlation is lower than tolerance, it is masked
        """
        return self[corr] > tolerance


    @_base_mask(time_required=True)
    def outliers(self, tolerance=1., mode="or"):
        """
    Mask outliers measured by amount of standard deviations from the mean.

    Parameters
    ----------
    tolerance :  float
        amount of standard deviations allowed from the mean
    mode : str
         can be "and" or "or" (default). If "or" ("and"), then only one (both) of two vector components need(s) to
         be within tolerance.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_std = self[v_x].std(dim="time")
            y_std = self[v_y].std(dim="time")
            x_mean = self[v_x].mean(dim="time")
            y_mean = self[v_y].mean(dim="time")
            x_condition = np.abs((self[v_x] - x_mean) / x_std) < tolerance
            y_condition = np.abs((self[v_y] - y_mean) / y_std) < tolerance
        if mode == "or":
            mask = x_condition | y_condition
        else:
            mask = x_condition & y_condition
        return mask

    @_base_mask(time_required=True)
    def variance(self, tolerance=5, mode="and"):
        """
    Masks locations if their variance (std/mean in time) is above a tolerance level for either or both
    x and y direction.

    Parameters
    ----------
    tolerance :  float
        amount of standard deviations allowed from the mean
    mode : str
         can be "and" (default) or "or". If "or" ("and"), then only one (both) of two vector components need(s) to
         be within tolerance.
        """
        x_std = self[v_x].std(dim="time")
        y_std = self[v_y].std(dim="time")
        x_mean = np.maximum(self[v_x].mean(dim="time"), 1e30)
        y_mean = np.maximum(self[v_y].mean(dim="time"), 1e30)
        x_var = np.abs(x_std / x_mean)
        y_var = np.abs(y_std / y_mean)
        x_condition = x_var < tolerance
        y_condition = y_var < tolerance
        if mode == "or":
            mask = x_condition | y_condition
        else:
            mask = x_condition & y_condition
        return mask


    @_base_mask(time_required=True)
    def rolling(self, wdw=5, tolerance=0.5):
        """
    Masks values if neighbours over a certain rolling length before and after, have a
    significantly higher velocity than value under consideration, measured by tolerance.

    Parameters
    ----------
    wdw : int, optional
        amount of time steps in rolling window (centred) (default: 5)
        """
        s = (self[v_x] ** 2 + self[v_y] ** 2) ** 0.5
        s_rolling = s.fillna(0.).rolling(time=wdw, center=True).max()
        mask = s > tolerance * s_rolling
        return mask


    @_base_mask()
    def window_nan(self, tolerance=0.7, wdw=1, **kwargs):
        """
    Masks values if their surrounding neighbours (inc. value itself) contain too many NaNs. Meant to remove isolated
    velocity estimates.

    Parameters
    ----------
    tolerance : float, optional
        minimum amount of valid values in search window measured as a fraction of total amount of values [0-1]
        (default: 0.3)
    wdw : int, optional
        window size to use for sampling neighbours. zero means, only cell itself, 1 means 3x3 window.
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
        """
        # collect points within a stride, collate and analyze for nan fraction
        ds_wdw = helpers.stack_window(self, wdw=wdw, **kwargs)
        valid_neighbours = ds_wdw[v_x].count(dim="stride")
        mask = valid_neighbours >= tolerance * len(ds_wdw.stride)
        return mask

    @_base_mask()
    def window_mean(self, tolerance=0.7, wdw=1, mode="or", **kwargs):
        """
    Masks values when their value deviates more than tolerance (measured as relative fraction) from the mean of its
    neighbours (inc. itself).

    Parameters
    ----------
    tolerance: float, optional
        amount of velocity relative to the mean velocity  (default: 0.7)
    wdw : int, optional
        window used to determine relevant neighbours
    wdw_x_min : int, optional
        window size in negative x-direction of grid (must be negative), overrules wdw in negative x-direction if set
    wdw_x_max : int, optional
        window size in positive x-direction of grid, overrules wdw in positive x-direction if set
    wdw_y_min : int, optional
        window size in negative y-direction of grid (must be negative), overrules wdw in negative y-direction if set
    wdw_y_max : int, optional
        window size in positive y-direction of grid, overrules wdw in positive x-direction if set.
    mode : str
         can be "and" (default) or "or". If "or" ("and"), then only one (both) of two vector components need(s) to
         be within tolerance.
        """
        # collect points within a stride, collate and analyze for median value and deviation
        ds_wdw = helpers.stack_window(self, wdw=wdw, **kwargs)
        ds_mean = ds_wdw.mean(dim="stride")
        x_condition = np.abs(self[v_x] - ds_mean[v_x]) / ds_mean[v_x] < tolerance
        y_condition = np.abs(self[v_y] - ds_mean[v_y]) / ds_mean[v_y] < tolerance
        if mode == "or":
            mask = x_condition | y_condition
        else:
            mask = x_condition & y_condition
        return mask

