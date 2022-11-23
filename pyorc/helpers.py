import copy
import cv2

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pyproj import Transformer
from rasterio.transform import Affine, xy
from rasterio.crs import CRS
from rasterio import warp
from scipy.optimize import differential_evolution
from scipy.signal import convolve2d
from scipy.interpolate import interp1d


def affine_from_grid(xi, yi):
    """Retrieve the affine transformation from a gridded set of coordinates.
    This function (unlike rasterio.transform functions) can also handle rotated grids

    Parameters
    ----------
    xi: np.ndarray (2D)
        gridded x-coordinates
    yi: np.ndarray (2D)
        gridded y-coordinates

    Returns
    -------
    obj : rasterio.transform.Affine
    """

    xul, yul = xi[0, 0], yi[0, 0]
    xcol, ycol = xi[0, 1], yi[0, 1]
    xrow, yrow = xi[1, 0], yi[1, 0]
    dx_col = xcol - xul
    dy_col = ycol - yul
    dx_row = xrow - xul
    dy_row = yrow - yul
    return Affine(dx_col, dy_col, xul, dx_row, dy_row, yul)


def depth_integrate(depth, v, v_corr=0.85, name="q"):
    """Integrate velocities [m s-1] to depth-integrated velocity [m2 s-1] using depth information

    Parameters
    ----------
    depth : DataArray (points)
        bathymetry depths (ref. CRS)
    v : DataArray (time, points)
        effective velocity at surface [m s-1]
    v_corr : float (range: 0-1), optional
        typically close to 1, correction factor from surface to depth-average (default: 0.85)
    name: str, optional
        name of DataArray (default: "q")

    Returns
    -------
    q: DataArray (time, points)
        depth integrated velocity [m2 s-1]
    """
    # compute the depth average velocity
    q = v * v_corr * depth
    q.attrs = {
        "standard_name": "velocity_depth",
        "long_name": "velocity averaged over depth",
        "units": "m2 s-1",
    }
    # set name
    q.name = name
    return q


def deserialize_attr(data_array, attr, dtype=np.array, args_parse=False):
    """Return a deserialized version of said property (assumed to be stored as a string) of DataArray.

    Parameters
    ----------
    obj: xr.DataArray
        attributes of interest
    attr: str
        name of attributes
    dtype: object type, optional
        function will try to perform type(eval(attr)), default np.array
    args_parse: boolean, optional
        if True, function will try to return type(*eval(attr)), assuming attribute contains list
        of arguments (default: False)

    Returns
    -------
    parsed_arg: type defined by user
        parsed attribute, of type defined by arg type
    """
    assert hasattr(data_array, attr), f'frames do not contain attribute with name "{attr}'
    attr_obj = getattr(data_array, attr)
    if args_parse:
        return dtype(*eval(attr_obj))
    return dtype(eval(attr_obj))


def get_axes(cols, rows, resolution):
    """Retrieve a locally spaced axes for surface velocimetry results on the basis of resolution and row and
    col distances from the original frames

    Parameters
    ----------
    cols: list
        ints, columns, sampled from the original projected frames
    rows: list
        ints, rows, sampled from the original projected frames
    resolution: float
        resolution of original frames

    Returns
    -------
    obj : np.ndarray
        x-axis with origin at the left
    obj2 : np.ndarray
        y-axis with origin on the top

    """
    spacing_x = np.diff(cols[0])[0]
    spacing_y = np.diff(rows[:, 0])[0]
    x = np.linspace(
        resolution / 2 * spacing_x,
        (len(cols[0]) - 0.5) * resolution * spacing_x,
        len(cols[0]),
    )
    y = np.flipud(
        np.linspace(
            resolution / 2 * spacing_y,
            (len(rows[:, 0]) - 0.5) * resolution * spacing_y,
            len(rows[:, 0]),
        )
    )
    return x, y


def get_geo_axes(tiles=None, extent=None, zoom_level=19, **kwargs):
    try:
        import cartopy
        import cartopy.io.img_tiles as cimgt
        import cartopy.crs as ccrs
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'Geographic plotting requires cartopy. Please install it with "conda install cartopy" and try '
            'again.')
    if tiles is not None:
        tiler = getattr(cimgt, tiles)(**kwargs)
        crs = tiler.crs
    else:
        crs = ccrs.PlateCarree()
    ax = plt.subplot(projection=crs)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    if tiles is not None:
        ax.add_image(tiler, zoom_level, zorder=1)
    return ax


def get_xs_ys(cols, rows, transform):
    """Computes rasters of x and y coordinates, based on row and column counts and a defined transform.

    Parameters
    ----------
    cols: list of ints
        column counts
    rows: list of ints
        row counts
    transform: np.ndarray (1D)
        rasterio compatible transform parameters

    Returns
    -------
    xs : np.ndarray (MxN)
        x-coordinates
    ys : np.ndarray (MxN)
        y-coordinates
    """
    xs, ys = xy(transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)
    return xs, ys


def get_lons_lats(xs, ys, src_crs, dst_crs=CRS.from_epsg(4326)):
    """Computes raster of longitude and latitude coordinates (default) of a certain raster set of coordinates in a local
    coordinate reference system. User can supply an alternative coordinate reference system if projection other than
    WGS84 Lat Lon is needed.    

    Parameters
    ----------
    xs : np.ndarray (MxN)
        x-coordinates
    ys : np.ndarray (MxN)
        y-coordinates
    src_crs : int, dict or str
        Coordinate Reference System (of source coordinates). Accepts EPSG codes (int or str) proj (str or dict) or wkt
        (str).
    dst_crs : int, dict or str, optional
        Coordinate Reference System (of target coordinates). Accepts EPSG codes (int or str) proj (str or dict) or wkt
        (str). default: CRS.from_epsg(4326) for wGS84 lat-lon

    Returns
    -------
    lons : np.ndarray (MxN)
        longitude coordinates
    lats: np.ndarray (MxN)
        latitude coordinates
    """
    lons, lats = warp.transform(src_crs, dst_crs, xs.flatten(), ys.flatten())
    lons, lats = (
        np.array(lons).reshape(xs.shape),
        np.array(lats).reshape(ys.shape),
    )
    return lons, lats


def log_profile(X, z0, k_max, s0=0., s1=0.):
    """Returns values of a log-profile function

    Parameters
    ----------
    X: tuple with np.ndarrays
        (depth [m], distance to bank [m]) arrays of equal length
    z0: float
        depth with zero velocity [m]
    k_max: float
        maximum scale factor of log-profile function [-]
    s0: float, optional
        distance from bank (default: 0.) where k equals zero (and thus velocity is zero) [m]
    s1: float, optional
        distance from bank (default: 0. meaning no difference over distance) where k=k_max (k cannot be larger than
        k_max) [m]

    Returns
    -------
    velocity : np.ndarray
        V values from log-profile, equal shape as arrays inside X [m s-1]
    """
    z, s = X
    k = k_max * np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
    v = k * np.maximum(np.log(np.maximum(z, 1e-6) / z0), 0)
    return v


def mse(pars, func, X, Y):
    """mean of sum of squares between evaluation of function with provided parameters and X input, and Y as dependent variable.

    Parameters
    ----------
    pars : list or tuple
        parameter passed as *args to func
    func: function def
        receiving X and *pars as input and returning predicted Y as result
    X: tuple with lists or array-likes
        indepent variable(s).
    Y: list or array-like
        dependent variable, predicted by func

    Returns
    -------

    Y_pred : list or array-like
        predicted Y from X and pars by func
    """
    Y_pred = func(X, *pars)
    mse = np.sum((Y_pred - Y) ** 2)
    return mse


def neighbour_stack(array, stride=1, missing=-9999.):
    """Builds a stack of arrays from a 2-D input array, constructed by permutation in space using a provided stride.

    Parameters
    ----------
    array : np.ndarray (2D)
        any values (may contain NaN)
    stride : int, optional
        stride used to determine relevant neighbours (default: 1)
    missing : float, optional
        a temporary missing value, used to be able to convolve NaNs

    Returns
    -------
    obj : np.array (3D)
        stack of 2-D arrays, with strided neighbours (length 1st dim : (stride*2+1)**2 )
    """
    array = copy.deepcopy(array)
    array[np.isnan(array)] = missing
    array_move = []
    for vert in range(-stride, stride + 1):
        for horz in range(-stride, stride + 1):
            conv_arr = np.zeros((abs(vert) * 2 + 1, abs(horz) * 2 + 1))
            _y = int(np.floor((abs(vert) * 2 + 1) / 2)) + vert
            _x = int(np.floor((abs(horz) * 2 + 1) / 2)) + horz
            conv_arr[_y, _x] = 1
            array_move.append(convolve2d(array, conv_arr, mode="same", fillvalue=np.nan))
    array_move = np.stack(array_move)
    # replace missings by Nan
    array_move[np.isclose(array_move, missing)] = np.nan
    return array_move


def optimize_log_profile(
        z,
        v,
        dist_bank=None,
        bounds=([0.001, 0.1], [-20, 20], [0., 5], [0., 100]),
        workers=2,
        popsize=100,
        updating="deferred",
        seed=0,
        **kwargs
):
    """optimize velocity log profile relation of v=k*max(z/z0) with k a function of distance to bank and k_max
    A differential evolution optimizer is used.

    Parameters
    ----------
    z : list
        depths [m]
    v : list
        surface velocities [m s-1]
    dist_bank : list, optional
        distances to bank [m]
    **kwargs : keyword arguments for scipy.optimize.differential_evolution

    Returns
    -------
    pars : dict
        fitted parameters of log_profile {z_0, k_max, s0 and s1}
    """
    # replace by infinites if not provided
    dist_bank = np.ones(len(v)) * np.inf if dist_bank is None else dist_bank
    v = np.array(v)
    z = np.array(z)
    X = (z, dist_bank)
    Y = v
    result = differential_evolution(
        wrap_mse,
        args=(log_profile, X, Y),
        bounds=bounds,
        workers=workers,
        popsize=popsize,
        updating=updating,
        seed=seed,
        **kwargs
    )
    # unravel parameters
    z0, k_max, s0, s1 = result.x
    return {"z0": z0, "k_max": k_max, "s0": s0, "s1": s1}


def rotate_u_v(u, v, theta, deg=False):
    """Rotate u and v components of vector counter clockwise by an amount of rotation.

    Parameters
    ----------
    u : float, np.ndarray or xr.DataArray
        x-direction component of vector
    v : float, np.ndarray or xr.DataArray
        y-direction component of vector
    theta : float
        amount of counter clockwise rotation in radians or degrees (dependent on deg)
    deg : boolean, optional
        if True, theta is defined in degrees, otherwise radians (default: False)

    Returns
    -------
    u_rot : float, np.ndarray or xr.DataArray
        rotated x-direction component of vector
    v_rot : float, np.ndarray or xr.DataArray
        rotated y-direction component of vector
    """
    theta = np.radians(theta) if deg else theta
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    # compute rotations with dot-product
    u2 = r[0, 0] * u + r[0, 1] * v
    v2 = r[1, 0] * u + r[1, 1] * v
    return u2, v2

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def staggered_index(start=0, end=100):
    """
    Returns a list of staggered indexes that start at the outer indexes and gradually move inwards

    Parameters
    ----------
    start : int, optional
        start index number (default: 0)
    end : int, optional
        end index number (default: 100)

    Returns
    -------
    idx : list
        staggered indexes from start to end
    """
    # make list of frames in order to read, starting with start + end frame
    idx_order = [start, end]
    # make sorted representation of frames
    idx_sort = np.array(idx_order)
    idx_sort.sort()
    while True:
        idx_new = (np.round((idx_sort[0:-1] + idx_sort[1:]) / 2)).astype("int")
        # check which of these are already on the list
        idx_new = list(set(idx_new).difference(idx_order))
        if len(idx_new) == 0:
            # we have treated all idxs
            break
        idx_order += idx_new
        idx_sort = np.array(idx_order)
        idx_sort.sort()
    return idx_order

def velocity_log_fit(v, depth, dist_shore, dim="quantile"):
    """Fill missing surface velocities using a velocity depth profile with

    Parameters
    ----------
    v : xr.DataArray (time, points)
        effective velocity at surface [m s-1]
    depth : xr.DataArray (points)
        bathymetry depths [m]
    dist_shore : xr.DataArray (points)
        shortest distance to a dry river bed point
    dim: str, optional
        dimension over which data should be grouped, default: "quantile", dimension must exist in v, typically
        "quantile" or "time"

    Returns
    -------

    v_fill: xr.DataArray (quantile or time, points)
        filled surface velocities  [m s-1]
    """

    def log_fit(_v):
        pars = optimize_log_profile(
            depth[np.isfinite(_v).values],
            _v[np.isfinite(_v).values],
            dist_shore[np.isfinite(_v).values]
        )
        _v[np.isnan(_v).values] = log_profile(
            (
                depth[np.isnan(_v).values],
                dist_shore[np.isnan(_v).values]
            ),
            **pars
        )
        # enforce that velocities are zero with zero depth
        _v[depth <= 0] = 0.
        return np.maximum(_v, 0)

    # fill per grouped dimension
    v.load()
    v_group = copy.deepcopy(v).groupby(dim)
    return v_group.map(log_fit)


def velocity_log_interp(v, dist_wall, d_0=0.1, dim="quantile"):
    """

    Parameters
    ----------
    v : xr.DataArray (time, points)
        effective velocity at surface [m s-1]
    dist_wall : xr.DataArray (points)
        shortest distance to the river bed
    d_0 : float, optional
        roughness length (default: 0.1)
    dim: str, optional
        dimension over which data should be grouped, default: "quantile", dimension must exist in v, typically
        "quantile" or "time"

    Returns
    -------

    """

    def log_interp(_v):
        # scale with log depth
        c = xr.DataArray(_v / np.log(np.maximum(dist_wall, d_0) / d_0))
        # fill dry points with the nearest valid value for c
        c[dist_wall == 0] = c.interpolate_na(dim="points", method="nearest", fill_value="extrapolate")[dist_wall == 0]
        # interpolate with linear interpolation
        c = c.interpolate_na(dim="points")
        # use filled c to interpret missing v
        _v[np.isnan(_v)] = (np.log(np.maximum(dist_wall, d_0) / d_0) * c)[np.isnan(_v)]
        return _v

    # fill per grouped dimension
    v.load()
    v_group = copy.deepcopy(v).groupby(dim)
    return v_group.map(log_interp)


def wrap_mse(pars_iter, *args):
    return mse(pars_iter, *args)


def xy_equidistant(x, y, distance, z=None):
    """Transforms a set of ordered in space x, y (and z if provided) coordinates into x, y (and z) coordinates with equal
    1-dimensional distance between them using piece-wise linear interpolation. Extrapolation is used for the last point
    to ensure the range of points covers at least the full range of x, y coordinates.

    Parameters
    ----------
    x : np.ndarray (1D)
        set of (assumed ordered) x-coordinates
    y : np.ndarray (1D)
        set of (assumed ordered) x-coordinates
    distance : float
        user demanded distance between equidistant samples measured in cumulated 1-dimensional distance from xy
        origin (first point)
    z : np.ndarray (1D), optional
        set of (assumed ordered) z-coordinates (default: None, meaning only x, y interpolated points are returned)

    Returns
    -------
    x_sample : np.ndarray (1D)
        interpolated x-coordinates for x, y, s (distance from first point), z
    y_sample : np.ndarray (1D)
        interpolated y-coordinates
    s_sample : np.ndarray (1D)
        interpolated s-coordinates, s being piece-wise linear distance from first point
    z_sample : np.ndarray (1D), optional
        interpolated z-coordinates (only returned if z is not None):
    """
    # estimate cumulative distance between points, starting with zero
    x_diff = np.concatenate((np.array([0]), np.diff(x)))
    y_diff = np.concatenate((np.array([0]), np.diff(y)))
    s = np.cumsum((x_diff ** 2 + y_diff ** 2) ** 0.5)

    # create interpolation functions for x and y coordinates
    f_x = interp1d(s, x, fill_value="extrapolate")
    f_y = interp1d(s, y, fill_value="extrapolate")

    # make equidistant samples
    s_sample = np.arange(s.min(), np.ceil((1 + s.max() / distance) * distance), distance)

    # interpolate x and y coordinates
    x_sample = f_x(s_sample)
    y_sample = f_y(s_sample)
    if z is None:
        return x_sample, y_sample, s_sample
    else:
        f_z = interp1d(s, z, fill_value="extrapolate")
        z_sample = f_z(s_sample)
        return x_sample, y_sample, z_sample, s_sample


def xy_angle(x, y):
    """Determine angle between x, y points.

    Parameters
    ----------
    x : np.ndarray (1D)
        set of (assumed ordered) x-coordinates
    y : np.ndarray (1D)
        set of (assumed ordered) x-coordinates

    Returns
    -------
    angle : np.ndarray (1D)
        angle between the point left and right of the point under consideration. The most left and right coordinates
        are based on the first and last 2 points respectively
    """
    angles = np.zeros(len(x))
    angles[1:-1] = np.arctan2(x[2:] - x[0:-2], y[2:] - y[0:-2])
    angles[0] = np.arctan2(x[1] - x[0], y[1] - y[0])
    angles[-1] = np.arctan2(x[-1] - x[-2], y[-1] - y[-2])
    return angles


def xy_to_perspective(x, y, resolution, M, reverse_y=None):
    """
    Back transform local meters-from-top-left coordinates from frame to original perspective of camera using M
    matrix, belonging to transformation from orthographic to local.

    Parameters
    ----------
    x : np.ndarray (1D)
        axis of x-coordinates in local projection with origin top-left, to be backwards projected
    y : np.ndarray (1D)
        axis of y-coordinates in local projection with origin top-left, to be backwards projected
    resolution : float
        resolution of original projected frames coordinates of x and y
    M : np.ndarray
        2x3 transformation matrix (generated with cv2.getPerspectiveTransform)

    Returns
    -------
    xp : np.ndarray (2D)
        perspective columns with shape len(y), len(x)
    yp : np.ndarray (2D)
        perspective rows with shape len(y), len(x)
    """
    cols, rows = x / resolution - 0.5, y / resolution - 0.5
    if reverse_y is not None:
        rows = reverse_y - rows
    # make list of coordinates, compatible with cv2.perspectiveTransform
    coords = np.float32([np.array([cols.flatten(), rows.flatten()]).transpose([1, 0])])
    coords_trans = cv2.perspectiveTransform(coords, M)
    xp = coords_trans[0][:, 0].reshape(cols.shape)
    yp = coords_trans[0][:, 1].reshape(cols.shape)
    return xp, yp


def xyz_transform(points, crs_from, crs_to):
    """transforms set of x and y coordinates from one CRS to another

    Parameters
    ----------
    points : list of lists
        xyz-coordinates or xy-coordinates in crs_from
    crs_from : int, dict or str, optional
        Coordinate Reference System (source). Accepts EPSG codes (int or str) proj (str or dict) or wkt (str).
    crs_to : int, dict or str, optional
        Coordinate Reference System (destination). Accepts EPSG codes (int or str) proj (str or dict) or wkt (str).

    Returns
    -------
    x_trans : np.ndarray
        x-coordinates transformed
    y_trans : np.ndarray
        y-coordinates transformed
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    transform = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    # transform dst coordinates to local projection
    x_trans, y_trans = transform.transform(x, y)
    # check if finites are found, if not raise error
    assert(
        not(
            np.all(np.isinf(x_trans))
        )
    ), "Transformation did not give valid results, please check if the provided crs of input coordinates is correct."
    points[:, 0] = x_trans
    points[:, 1] = y_trans
    return points.tolist()
    # return transform.transform(x, y)
