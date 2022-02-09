import numpy as np
import rasterio
import xarray as xr

from pyorc import helpers
from pyproj import CRS

def get_uv_points(ds, x, y, z=None, crs=None, xs="xs", ys="ys"):
    """
    Interpolate all variables to supplied x and y coordinates. This function assumes that the grid
    can be rotated and that xs and ys are supplied following the projected coordinates supplied in
    "xs" and "ys" coordinate variables in ds. x-coordinates and y-coordinates that fall outside the
    domain of ds, are still stored in the result. Original coordinate values supplied are stored in
    coordinates "x", "y" and (if supplied) "z"

    :param ds: xarray dataset
    :param x: tuple or list-like, x-coordinates on which interpolation should be done
    :param y: tuple or list-like, y-coordinates on which interpolation should be done
    :param z: tuple or list-like, z-coordinates on which interpolation should be done, defaults to None
    :param crs: coordinate reference system (e.g. EPSG code) in which x, y and z are measured, defaults to None, assuming crs is the same as crs of ds
    :param xs: str, name of variable that stores the x coordinates in the projection in which "x" is supplied
    :param ys: str, name of variable that stores the y coordinates in the projection in which "y" is supplied
    :return: ds_points: xarray dataset, containing interpolated data at the supplied x and y coordinates
    """

    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
    # TODO consider removing this function
    transform = helpers.affine_from_grid(ds[xs].values, ds[ys].values)
    if crs is not None:
        # transform coordinates of cross section
        x, y = helpers.xy_transform(x, y, crs_from=crs, crs_to=CRS.from_wkt(ds.crs))
    # make a cols and rows temporary variable
    coli, rowi = np.meshgrid(np.arange(len(ds["x"])), np.arange(len(ds["y"])))
    ds["cols"], ds["rows"] = (["y", "x"], coli), (["y", "x"], rowi)
    # compute rows and cols locations of coordinates (x, y)
    rows, cols = rasterio.transform.rowcol(transform, list(x), list(y))
    rows, cols = np.array(rows), np.array(cols)

    # select x and y coordinates from axes
    idx = np.all(
        np.array([cols >= 0, cols < len(ds["x"]), rows >= 0, rows < len(ds["y"])]),
        axis=0,
    )
    x = np.empty(len(cols))
    x[:] = np.nan
    y = np.empty(len(rows))
    y[:] = np.nan
    x[idx] = ds["x"].isel(x=cols[idx])
    y[idx] = ds["y"].isel(y=rows[idx])
    # interpolate values from grid to list of x-y coordinates to grid in xarray format
    x = xr.DataArray(list(x), dims="points")
    y = xr.DataArray(list(y), dims="points")
    if np.isnan(x).all():
        raise ValueError("All bathymetry points are outside valid domain")
    else:
        ds_points = ds.interp(x=x, y=y)
    # add the xcoords and ycoords (and zcoords if available) originally assigned so that even points outside the grid covered by ds can be
    # found back from this dataset
    ds_points = ds_points.assign_coords(xcoords=("points", list(x)))
    ds_points = ds_points.assign_coords(ycoords=("points", list(y)))
    if z is not None:
        ds_points = ds_points.assign_coords(zcoords=("points", list(z)))

    return ds_points

