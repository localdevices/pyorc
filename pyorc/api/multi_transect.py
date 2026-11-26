"""Multi-transect module for pyorc."""

import numpy as np
import pandas as pd
import xarray as xr

from pyorc.api.orcbase import ORCBase


@xr.register_dataset_accessor("multi_transect")
class MultiTransect(ORCBase):
    """Transect functionalities that can be applied on ``xarray.Dataset``."""

    def __init__(self, xarray_obj):
        """Initialize a multi-transect ``xarray.Dataset`` containing velocities in ``[points, videos]`` dimensions.

        Parameters
        ----------
        xarray_obj: xr.Dataset
            multi-transect data fields (from ``pyorc.MultiTransect.from_transects``)

        """
        super(MultiTransect, self).__init__(xarray_obj)

    @classmethod
    def from_transects(cls, transects, dim="videos", dim_coords=None, quantile=0.5):
        """Create a multi-transect by concatenating transects along videos.

        Parameters
        ----------
        transects : list of Transect
            list of transect objects to concatenate along the videos dimension
        dim : str, optional
            name of the dimension along which to concatenate the transects (default: "videos")
        dim_coords : list, optional
            coordinates for the concatenation dimension (default: None)
        quantile : float, optional
            quantile to use for water level aggregation (default: 0.5)

        Examples
        --------
        Create a ``MultiTransect`` from existing ``Transect`` objects:

        >>> multi_transect = MultiTransect.from_transects(
        ...     [transect_1, transect_2], dim_coords=["video_1", "video_2"]
        ... )

        If ``dim_coords`` is not provided, the videos dimension will be indexed by integers starting from 0.

        """
        # check that the coordinates are identical in all transects
        for name in ["xcoords", "ycoords", "zcoords"]:
            reference = transects[0][name]
            for transect in transects[1:]:
                if name not in transect or not np.array_equal(reference.values, transect[name].values, equal_nan=True):
                    raise ValueError(
                        f"Variable '{name}' must be identical in all transects to be combined in a multi-transect"
                    )
        # extract water level from each transect
        h = [transect.transect.h_a for transect in transects]
        if dim_coords is not None:
            if len(dim_coords) != len(transects):
                raise ValueError("Length of dim_coords must match the number of transects")
            dim_index = pd.Index(dim_coords, name=dim)
        else:
            dim_index = dim  # range of ints used and also re-created with further concats
        if quantile is None:
            multi_transect = xr.concat(
                [transect for transect in transects], dim=dim_index
            )  # keep all quantiles if quantile is None
        else:
            multi_transect = xr.concat([transect.sel(quantile=quantile) for transect in transects], dim=dim_index)
        # add the water levels
        multi_transect["water_level"] = ("videos", h)
        multi_transect["water_level"].attrs.update(
            {
                "units": "m",
                "standard_name": "water_surface_height_above_reference_datum",
            }
        )
        return multi_transect

    @property
    def water_levels(self):
        """Return the water levels for each video in the multi-transect."""
        return self._obj["water_level"]

    @property
    def h_a(self):
        """Override the original h_a definition. Actual water level belonging to the processed video."""
        return self.water_levels

    @property
    def z_levels(self):
        """Return the water level in the xyz coordinate system of the camera configuration."""
        return self.camera_config.h_to_z(self.water_levels)

    @property
    def depths(self):
        """Return the depth for each point and video in the multi-transect."""
        return np.maximum(self.z_levels - self._obj["zcoords"], 0)

    @property
    def measured_velocities(self):
        """Return the measured velocities for each point and video in the multi-transect."""
        return self._obj["v_eff_nofill"]

    @property
    def distance_to_shore(self):
        """Return the distance to the nearest point with non-positive depth.

        Distances use the units of ``xcoords`` and ``ycoords``. Transects
        without a valid shore point have NaN distances.
        """

        def nearest_shore(depth, x, y):
            """Compute the distance to the nearest shore point for each point with a depth."""
            valid = np.isfinite(x) & np.isfinite(y)
            shore = (depth <= 0) & valid
            result = np.full(depth.shape, np.nan, dtype=float)
            if np.any(shore):
                result[valid] = np.hypot(x[valid, None] - x[shore], y[valid, None] - y[shore]).min(axis=1)
            return result

        depths = self.depths
        return xr.apply_ufunc(
            nearest_shore,
            depths,
            self._obj["xcoords"],
            self._obj["ycoords"],
            input_core_dims=[["points"], ["points"], ["points"]],
            output_core_dims=[["points"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        ).transpose(*depths.dims)
