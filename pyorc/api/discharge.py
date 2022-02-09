import numpy as np
import rasterio
import xarray as xr

from pyorc import helpers
from pyproj import CRS

def get_uv_points(ds, x, y, z=None, crs=None, v_eff=True, xs="xs", ys="ys"):
    """
    Interpolate all variables to supplied x and y coordinates of a cross section. This function assumes that the grid
    can be rotated and that xs and ys are supplied following the projected coordinates supplied in
    "xs" and "ys" coordinate variables in ds. x-coordinates and y-coordinates that fall outside the
    domain of ds, are still stored in the result for further interpolation. Original coordinate values supplied are
    stored in coordinates "x", "y" and (if supplied) "z".

    :param ds: xarray dataset
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

    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
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
    if v_eff:
        # add the effective velocity, perpendicular to cross section direction
        ds_points["v_eff"] = vector_to_scalar(
            ds_points["v_x"], ds_points["v_y"]
        )
    return ds_points

def vector_to_scalar(v_x, v_y, angle_method=0):
    """
    Turns velocity vectors into effective velocities over cross-section, by computing the perpendicular velocity component
    :param v_x: DataArray(t, points), time series in cross section points with x-directional velocities
    :param v_y: DataArray(t, points), time series in cross section points with y-directional velocities
    :param angle_method: if set to 0, then angle of cross section is determined with left to right bank coordinates,
        otherwise, it is determined per section
    :return: v_eff: DataArray(t, points), time series in cross section points with velocities perpendicular to cross section

    """
    xs = v_x["x"].values
    ys = v_x["y"].values
    # find points that are located on the area of interest
    idx = np.isfinite(xs)
    xs = xs[idx]
    ys = ys[idx]
    if angle_method == 0:
        x_left, y_left = xs[0], ys[0]
        x_right, y_right = xs[-1], ys[-1]
        angle_da = np.arctan2(x_right - x_left, y_right - y_left)
    else:
        # start with empty angle
        angle = np.zeros(ys.shape)
        angle_da = np.zeros(v_x["x"].shape)
        angle_da[:] = np.nan

        for n, (x, y) in enumerate(zip(xs, ys)):
            # determine the angle of the current point with its neighbours
            # check if we are at left bank
            # first estimate left bank angle
            undefined = True  # angle is still undefined
            m = 0
            while undefined:
                # go one step to the left
                m -= 1
                if n + m < 0:
                    # we are at the left bank, so angle with left neighbour is non-existing.
                    x_left, y_left = xs[n], ys[n]
                    # angle_left = np.nan
                    undefined = False
                else:
                    x_left, y_left = xs[n + m], ys[n + m]
                    if not ((x_left == x) and (y_left == y)):
                        # profile points are in another pixel, so estimate angle
                        undefined = False
                        angle_left = np.arctan2(x - x_left, y - y_left)

            # estimate right bank angle
            undefined = True  # angle is still undefined
            m = 0
            while undefined:
                # go one step to the left
                m += 1
                if n + m >= len(xs) - 1:
                    angle_right = np.nan
                    undefined = False
                else:
                    x_right, y_right = xs[n + m], ys[n + m]
                    if not ((x_right == x) and (y_right == y)):
                        # profile points are in another pixel, so estimate angle
                        undefined = False
                        angle_right = np.arctan2(x_right - x, y_right - y)
            angle[n] = np.nanmean([angle_left, angle_right])
        # add angles to array meant for data array
        angle_da[idx] = angle

    # compute angle of flow direction (i.e. the perpendicular of the cross section) and add as DataArray to ds_points
    flow_dir = angle_da - 0.5 * np.pi

    # compute per velocity vector in the dataset, what its angle is
    v_angle = np.arctan2(v_x, v_y)
    # compute the scalar value of velocity
    v_scalar = (v_x ** 2 + v_y ** 2) ** 0.5

    # compute difference in angle between velocity and perpendicular of cross section
    angle_diff = v_angle - flow_dir
    # compute effective velocity in the flow direction (i.e. perpendicular to cross section
    v_eff = np.cos(angle_diff) * v_scalar
    v_eff.attrs = {
        "standard_name": "velocity",
        "long_name": "velocity in perpendicular direction of cross section, measured by angle in radians, measured from up-direction",
        "units": "m s-1",
    }
    # set name
    v_eff.name = "v_eff"
    return v_eff

