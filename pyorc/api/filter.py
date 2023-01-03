import xarray as xr

from pyorc.const import v_x, v_y

def _base_filter(time_allowed=False):
    def decorator_func(filter_func):
        def wrapper_func(ref, inplace=False, *args, **kwargs):
            # Invoke the wrapped function first
            mask = filter_func(ref, **kwargs)
            print(time_allowed)
            print("Mask retrieved, now decide if it needs to be set or returned")
            # Now do something here with retval and/or action
            if inplace:
                # set the _obj data points
                ref._obj[v_x] = ref._obj[v_x].where(mask)
                ref._obj[v_y] = ref._obj[v_y].where(mask)
            return mask

        return wrapper_func
    return decorator_func

class _Velocimetry_FilterMethods:
    """
    Enables use of ``ds.velocimetry.filter`` functions as attributes on a Dataset containing velocimetry results.
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

    def __call__(self, mask, *args, **kwargs):
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
        if isinstance(mask, list):
            # combine masks
            masks = xr.concat(mask, dim="mask")
            mask = masks.all(dim="mask")
            self._obj[v_x] = self._obj[v_x].where(mask)
            self._obj[v_y] = self._obj[v_y].where(mask)

    @_base_filter(time_allowed=False)
    def minmax(self, s_min=0.1, s_max=0.6):
        """
        Masks values if the velocity scalar lies outside a user-defined valid range.
    
        Parameters
        ----------
        s_min : float, optional
            minimum scalar velocity [m s-1] (default: 0.1)
        s_max : float, optional
            maximum scalar velocity [m s-1] (default: 5.)
    
        Returns
        -------
        mask : xr.DataArray
            mask applicable to input dataset (only returned if ``inplace=False``)
        """
        s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        # create filter
        filter = (s > s_min) & (s < s_max)
        return filter
        # s_mean = s.mean(dim="time")
        # self._obj[v_x] = self._obj[v_x].where(s_mean > s_min)
        # self._obj[v_x] = self._obj[v_x].where(s_mean < s_max)
        # self._obj[v_y] = self._obj[v_y].where(s_mean > s_min)
        # self._obj[v_y] = self._obj[v_y].where(s_mean < s_max)

