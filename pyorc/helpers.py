import copy
import cv2
import dask.array as da
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


def delayed_to_da(delayed_das, shape, dtype, coords, attrs={}, name=None, object_type=xr.DataArray):
    """Convert a list of delayed 2D arrays (assumed to be time steps of grids) into a 3D xr.DataArray with dask arrays
    with all axes.

    Parameters
    ----------
    delayed_das: xr.DataArray or list
    shape: tuple
        foreseen shape of data arrays (rows, cols)
    dtype: str or dtype
        e.g. "uint8" of data arrays
    coords: tuple of str
        dimensions of the xr.DataArray being prepared, usually ("time", "y", "x").
    attrs: dict, optional
        attributes for xr.DataArray (default: {})
    name: str, optional
        name of variable, default: None
    object_type: type
        type of object to create from lazy array (default: xr.DataArray)

    Returns
    -------
    object of object_type
        default: xr.DataArray
    """
    if isinstance(delayed_das, list):
        data_array = da.stack(
            [da.from_delayed(
                d,
                dtype=dtype,
                shape=shape
            ) for d in delayed_das],
            axis=0
        )
    else:
        data_array = da.from_delayed(
            delayed_das,
            dtype=dtype,
            shape=shape
        )
    for n, (coord, values) in enumerate(coords.items()):
        assert(len(values) == data_array.shape[n]), f"Length of {coord} axis {len(values)} is not equal to amount of data arrays {data_array.shape[n]}"
    return object_type(
        data_array,
        dims=tuple(coords.keys()),
        coords=coords,
        attrs=attrs,
        name=name
    )


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


def distance_pts(c1, c2):
    """Compute distance between c1 and c2

    Parameters
    ----------

    c1 : tuple
        (x, y), coordinate 1
    c2: tuple
        (x, y), coordinate 2

    Returns
    -------
    dist : float
        distance between c1 and c2
    """
    x1, y1 = c1
    x2, y2 = c2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


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


def get_xs_ys(cols, rows, transform):
    """
    Computes rasters of x and y coordinates, based on row and column counts and a defined transform.

    :param cols: list of ints, defining the column counts
    :param rows: list of ints, defining the row counts
    :param transform: np.ndarray, 1D, with 6 rasterio compatible transform parameters
    :return: 2 np.ndarray (MxN): xs: x-coordinates, ys: y-coordinates, lons: longitude coordinates, lats: latitude coordinates
    """
    xs, ys = xy(transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)
    return xs, ys


def get_lons_lats(xs, ys, src_crs, dst_crs=CRS.from_epsg(4326)):
    """
    Computes raster of longitude and latitude coordinates (default) of a certain raster set of coordinates in a local
    coordinate reference system. User can supply an alternative coordinate reference system if projection other than
    WGS84 Lat Lon is needed.    

    :param xs: x-coordinates in a given CRS
    :param ys: y-coordinates in a given CRS
    :param src_crs: source coordinate reference system of xs and ys 
    :param dst_crs: target coordinate reference system (default: CRS.from_epsg(4326) for wGS84 lat-lon)
    :return: 2 np.ndarray (MxN): lons: longitude coordinates, lats: latitude coordinates
    """
    lons, lats = warp.transform(src_crs, dst_crs, xs.flatten(), ys.flatten())
    lons, lats = (
        np.array(lons).reshape(xs.shape),
        np.array(lats).reshape(ys.shape),
    )
    return lons, lats


def log_profile(X, z0, k_max, s0=0., s1=0.):
    """
    Returns values of a log-profile function

    :param X: tuple with (depth [m], distance to bank [m]) arrays of equal length
    :param z0: float, depth with zero velocity [m]
    :param k_max: float, maximum scale factor of log-profile function [-]
    :param s0: float, distance from bank where k equals zero (and thus velocity is zero) [m]
    :param s1: float, distance from bank whrre k=k_max (k cannot be larger than k_max) [m]
    :return: values from log-profile, equal amount and shape as arrays inside X [m s-1]
    """
    z, s = X
    k = k_max * np.minimum(np.maximum((s-s0)/(s1-s0), 0), 1)
    v = k*np.maximum(np.log(np.maximum(z, 1e-6)/z0), 0)
    return v


def mse(pars, func, X, Y):
    """
    mean of sum of squares between evaluation of function with provided parameters and X input, and Y as dependent variable.

    :param pars: list or tuple, parameter passed as *args to func
    :param func: function, receiving X and *pars as input and returning predicted Y as result
    :param X: tuple with lists or array-likes, indepent variable(s).
    :param Y: list or array-like, dependent variable, predicted by func
    :return: list or array-like, predicted Y from X and pars by func
    """
    Y_pred = func(X, *pars)
    mse = np.sum((Y_pred - Y) ** 2)
    return mse


def neighbour_stack(array, stride=1, missing=-9999.):
    """
    Builds a stack of arrays from a 2-D input array, constructed by permutation in space using a provided stride.

    :param array: 2-D numpy array, any values (may contain NaN)
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: np.array, 3D containing stack of 2-D arrays, with strided neighbours
    """
    array[np.isnan(array)] = missing
    array_move = []
    for vert in range(-stride, stride+1):
        for horz in range(-stride, stride+1):
            conv_arr = np.zeros((abs(vert)*2+1, abs(horz)*2+1))
            _y = int(np.floor((abs(vert)*2+1)/2)) + vert
            _x = int(np.floor((abs(horz)*2+1)/2)) + horz
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

    """
    optimize velocity log profile relation of v=k*max(z/z0) with k a function of distance to bank and k_max
    A differential evolution optimizer is used.

    :param z: list of depths
    :param v: list of surface velocities
    :param dist_bank: list of distances to bank
    :param kwargs: keyword arguments for scipy.optimize.differential_evolution
    :return: dict, fitted parameters of log_profile {z_0, k_max, s0 and s1}
    """
    if dist_bank is None:
        dist_bank = np.inf(len(v))
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
    """
    Rotate u and v components of vector counter clockwise by an amount of rotation.

    :param u: float, np.ndarray or xr.DataArray, x-direction component of vector
    :param v: float, np.ndarray or xr.DataArray, y-direction component of vector
    :param theta: amount of counter clockwise rotation in radians or degrees (dependent on deg)
    :param deg: if True, theta is defined in degrees, otherwise radians (default: False)
    :return: u and v rotated
    """
    if deg:
        # convert to radians first
        theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    # compute rotations with dot-product
    u2 = r[0, 0] * u + r[0, 1] * v
    v2 = r[1, 0] * u + r[1, 1] * v
    return u2, v2


def velocity_fill(x, y, depth, v, groupby="quantile"):
    """
    Fill missing surface velocities using a velocity depth profile with

    :param x: DataArray(points), x-coordinates of bathymetry depths (ref. CRS)
    :param y: DataArray(points), y-coordinates of bathymetry depths (ref. CRS)
    :param z: DataArray(points), bathymetry depths (ref. CRS)
    :param v: DataArray(time, points), effective velocity at surface [m s-1]
    :param z_0: float, zero water level (ref. CRS)
    :param h_ref: float, water level measured during survey (ref. z_0)
    :param h_a: float, actual water level (ref. z_0)
    :param groupby: str, dimension over which data should be grouped, default: "quantile", dimension must exist in v,
        typically "quantile" or "time"
    :return: v_fill: DataArray(quantile or time, points), filled velocities  [m s-1]
    """
    def fit(_v):
        pars = optimize_log_profile(depth[np.isfinite(_v).values], _v[np.isfinite(_v).values], dist_bank[np.isfinite(_v).values])
        _v[np.isnan(_v).values] = log_profile((depth[np.isnan(_v).values], dist_bank[np.isnan(_v).values]), **pars)
        # enforce that velocities are zero with zero depth
        _v[depth<=0] = 0.
        return np.maximum(_v, 0)

    z_dry = depth <= 0
    dist_bank = np.array([(((x[z_dry] - _x) ** 2 + (y[z_dry] - _y) ** 2) ** 0.5).min() for _x, _y, in zip(x, y)])
    # per time slice or quantile, fill missings
    v_group = copy.deepcopy(v).groupby(groupby)
    return v_group.map(fit)


def wrap_mse(pars_iter, *args):
    return mse(pars_iter, *args)


def xy_equidistant(x, y, distance, z=None):
    """
    Transforms a set of ordered in space x, y (and z if provided) coordinates into x, y (and z) coordinates with equal
    1-dimensional distance between them using piece-wise linear interpolation. Extrapolation is used for the last point
    to ensure the range of points covers at least the full range of x, y coordinates.

    :param x: np.ndarray, set of (assumed ordered) x-coordinates
    :param y: np.ndarray, set of (assumed ordered) x-coordinates
    :param distance: float, distance between equidistant samples measured in cumulated 1-dimensional distance from xy
        origin (first point)
    :param z: np.ndarray, set of (assumed ordered) z-coordinates (default: None)
    :return: (x_sample, y_sample, s_sample, z_sample (only if z is not None)): np.ndarrays with 1D points for
        x, y, s (distance from first point), z
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
    """
    Determine angle between x, y points.
    :param x: set of x-coordinate
    :param y: set of y-coordinates
    :return: angles over section. Angle is determined as the angle between the point left and right of the point
        under consideration. The most left and right coordinates are based on the first and last 2 points respectively.
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

    :param x: np.ndarray, 1D axis of x-coordinates in local projection with origin top-left, to be backwards projected
    :param y: np.ndarray, 1D axis of y-coordinates in local projection with origin top-left, to be backwards projected
    :param resolution: resolution of original projected frames coordinates of x and y
    :param M: transformation matrix generated with cv2.getPerspectiveTransform
    :return: (xp, yp), np.ndarray of shape (len(y), len(x)) containing perspective columns (xp) and rows (yp) of data
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


def xy_transform(x, y, crs_from, crs_to):
    """
    transforms set of x and y coordinates from one CRS to another

    :param x: np.ndarray, 1D axis of x-coordinates in local projection with origin top-left, to be backwards projected
    :param y: np.ndarray, 1D axis of y-coordinates in local projection with origin top-left, to be backwards projected
    :param crs_from: source crs, compatible with rasterio.crs.CRS.from_user_input (e.g. a epsg number or proj string)
    :param crs_to: destination crs, compatible with rasterio.crs.CRS.from_user_input (e.g. a epsg number or proj string)
    :param y: np.ndarray, 1D axis of y-coordinates in local projection with origin top-left, to be backwards projected

    """
    transform = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    # transform dst coordinates to local projection
    x_trans, y_trans = transform.transform(x, y)
    if np.all(np.isinf(x_trans)):
        raise ValueError("Transformation did not give valid results, please check if the provided crs of input coordinates is correct.")
    return transform.transform(x, y)
