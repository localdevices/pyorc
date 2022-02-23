import copy
import cv2
import dask.array as da
import numpy as np
import xarray as xr

from pyproj import CRS, Transformer
from rasterio.transform import Affine
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from scipy.interpolate import interp1d

def add_xy_coords(ds, xy_coord_data, coords, attrs_dict):
    """
    add coordinate variables with x and y dimensions (2d) to existing xr.Dataset.

    :param ds: xr.Dataset with at least y and x dimensions
    :param xy_coord_data: list, one or several arrays with 2-dimensional coordinates
    :param coords: tuple with strings, indicating the dimensions of the data in xy_coord_data
    :param attrs_dict: list of dicts, containing attributes belonging to xy_coord_data, must have equal length as xy_coord_data
    :return: xr.Dataset, with added coordinate variables.
    """
    dims = tuple(coords.keys())
    xy_coord_data = [
        xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=name
        ) for data, (name, attrs) in zip(xy_coord_data, attrs_dict.items())]
    # assign the coordinates
    ds = ds.assign_coords({
        k: (dims, v) for k, v in zip(attrs_dict, xy_coord_data)
    })
    # add the attributes (not possible with assign_coords
    for k, v in zip(attrs_dict, xy_coord_data):
        ds[k].attrs = v.attrs

    return ds


def affine_from_grid(xi, yi):
    """
    Retrieve the affine transformation from a gridded set of coordinates.
    This function (unlike rasterio.transform functions) can also handle rotated grids

    :param xi: 2D numpy-like gridded x-coordinates
    :param yi: 2D numpy-like gridded y-coordinates

    :return: rasterio Affine
    """

    xul, yul = xi[0, 0], yi[0, 0]
    xcol, ycol = xi[0, 1], yi[0, 1]
    xrow, yrow = xi[1, 0], yi[1, 0]
    dx_col = xcol - xul
    dy_col = ycol - yul
    dx_row = xrow - xul
    dy_row = yrow - yul
    return Affine(dx_col, dy_col, xul, dx_row, dy_row, yul)


def delayed_to_da(delayed_das, shape, dtype, coords, attrs={}, name=None):
    """
    Convert a list of delayed 2D arrays (assumed to be time steps of grids) into a 3D xr.DataArray with dask arrays
        with all axes.

    :param delayed_das: Delayed dask data arrays (2D) or list of 2D delayed dask arrays
    :param shape: tuple, foreseen shape of data arrays (rows, cols)
    :param dtype: string or dtype, e.g. "uint8" of data arrays
    :param coords: tuple with strings, indicating the dimensions of the xr.DataArray being prepared, usually ("time", "y", "x").
    :param attrs: dict, containing attributes for xr.DataArray
    :param name: str, name of variable, default None
    :return: xr.DataArray with dask.array
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
        assert(len(values)==data_array.shape[n]), f"Length of {coord} axis {len(values)} is not equal to amount of data arrays {data_array.shape[n]}"
    return xr.DataArray(
        data_array,
        dims=tuple(coords.keys()),
        coords=coords,
        attrs=attrs,
        name=name
    )

def depth_integrate(z, v, z_0, h_ref, h_a, v_corr=0.85, name="q"):
    """
    :param z: DataArray(points), bathymetry depths (ref. CRS)
    :param v: DataArray(time, points), effective velocity at surface [m s-1]
    :param z_0: float, zero water level (ref. CRS)
    :param h_ref: float, water level measured during survey (ref. z_0)
    :param h_a: float, actual water level (ref. z_0)
    :param v_corr: float (range: 0-1, typically close to 1), correction factor from surface to depth-average (default: 0.85)
    :return: q: DataArray(time, points), depth integrated velocity [m2 s-1]
    """
    # compute depth, never smaller than zero. Depth is in words:
    #   + height of the level of staff gauge (z_0) measured during survey in gps CRS (e.g. WGS84)
    #   - z levels the bottom cross section observations measured in gps CRS (e.g. WGS84)
    #   + difference in water level measured with staff gauge during movie and during survey
    #   of course depth cannot be negative, so it is always maximized to zero when below zero
    depth = np.maximum(z_0 - z + h_a - h_ref, 0)
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
    """
    Compute distance between c1 and c2
    :param c1: tuple(x, y), coordinate 1
    :param c2: tuple(x, y), coordinate 2
    :return: float, distance between c1 and c2
    """
    x1, y1 = c1
    x2, y2 = c2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def deserialize_attr(data_array, attr, type=np.array, args_parse=False):
    """
    Return a deserialized version of said property (assumed to be stored as a string) of DataArray.

    :param data_array: xr.DataArray, containing attributes of interest
    :param attr: str, name of attributes
    :param type: object type to return, function will try to perform type(eval(attr)), default np.array
    :param args_parse: bool, if True, function will try to return type(*eval(attr)), assuming attribute contains list
        of arguments
    :return: parsed attribute, of type defined by arg type
    """
    assert hasattr(data_array, attr), f'frames do not contain attribute with name "{attr}'
    attr_obj = getattr(data_array, attr)
    if args_parse:
        return type(*eval(attr_obj))
    return type(eval(attr_obj))

def log_profile(z, z0, k):
    return k*np.maximum(np.log(np.maximum(z, 1e-6)/z0), 0)

def neighbour_stack(array, stride=1, missing=-9999.):
    """
    Builds a stack of arrays from a 2-D input array, constructed by permutation in space using a provided stride.

    :param array: 2-D numpy array, any values (may contain NaN)
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: 3-D numpy array, stack of 2-D arrays, with strided neighbours
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

def optimize_log_profile(z, v):
    """
    optimize velocity log profile relation of v=k*max(z/z0)
    :param z: list of depths
    :param v: list of surface velocities
    :return: {z_0, k}
    """
    if v.min() < 0:
        result = curve_fit(
            log_profile,
            np.array(z),
            np.array(-v),
            bounds=([0.05, 0.1], [0.050000001, 20]),
            p0=[0.05, 2]
        )
    else:
        result = curve_fit(
            log_profile,
            np.array(z),
            np.array(v),
            bounds=([0.05, 0.1], [0.050000001, 20]),
            p0=[0.05, 2]
        )

    z0, k = result[0]
    return {"z0": z0, "k": k}



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
    R = np.array(((c, -s), (s, c)))
    # compute rotations with dot-product
    u2 = R[0, 0] * u + R[0, 1] * v
    v2 = R[1, 0] * u + R[1, 1] * v
    return u2, v2

def velocity_fill(z, v, z_0, h_ref, h_a, groupby="quantile"):
    """
    Fill missing surface velocities using a velocity depth profile with

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
        pars = optimize_log_profile(depth[np.isfinite(_v)], _v[np.isfinite(_v)])
        if _v.min() < 0:
            _v[np.isnan(_v)] = -log_profile(depth[np.isnan(_v)], **pars)
        else:
            _v[np.isnan(_v)] = log_profile(depth[np.isnan(_v)], **pars)

        return _v
    depth = np.maximum(z_0 - z + h_a - h_ref, 0)
    # per slice, fill missings
    v_group = copy.deepcopy(v).groupby(groupby)
    return v_group.apply(fit)

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
    :return: (x_sample, y_sample, z_sample (only if z is not None)): np.ndarrays with 1D points
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
        return x_sample, y_sample
    else:
        f_z = interp1d(s, z, fill_value="extrapolate")
        z_sample = f_z(s_sample)
        return x_sample, y_sample, z_sample


def xy_to_perspective(x, y, resolution, M):
    """
    Back transform local meters-from-top-left coordinates from frame to original perspective of camera using M
    matrix, belonging to transformation from orthographic to local.

    :param x: np.ndarray, 1D axis of x-coordinates in local projection with origin top-left, to be backwards projected
    :param y: np.ndarray, 1D axis of y-coordinates in local projection with origin top-left, to be backwards projected
    :param resolution: resolution of original projected frames coordinates of x and y
    :param M: transformation matrix generated with cv2.getPerspectiveTransform
    :return: (xp, yp), np.ndarray of shape (len(y), len(x)) containing perspective columns (xp) and rows (yp) of data
    """
    # make a mesgrid of cols and rows
    if (len(x.shape) == 1 and len(y.shape) == 1):
        cols_i, rows_i = np.meshgrid(x / resolution - 0.5, y / resolution - 0.5)
    elif (len(x.shape) == 2 and len(y.shape) == 2):
        cols_i, rows_i = x / resolution - 0.5, y / resolution - 0.5
    else:
        raise ValueError(f"shape of x and y should both have a length of either 2, or 1, this is now {len(x.shape)} for x and {len(y.shape)} for y")
    # make list of coordinates, compatible with cv2.perspectiveTransform
    coords = np.float32([np.array([cols_i.flatten(), rows_i.flatten()]).transpose([1, 0])])
    coords_trans = cv2.perspectiveTransform(coords, M)
    xp = coords_trans[0][:, 0].reshape(cols_i.shape)
    yp = coords_trans[0][:, 1].reshape(cols_i.shape)
    return xp, yp

def xy_transform(x, y, crs_from, crs_to):
        try:
            crs = CRS.from_user_input(crs_from)
        except:
            raise ValueError(f"Input crs {crs_from} is not a valid Coordinate Reference System")
        try:
            crs = CRS.from_user_input(crs_from)
        except:
            raise ValueError(f"Output crs {crs_to} is not a valid Coordinate Reference System")
        transform = Transformer.from_crs(crs_from, crs_to, always_xy=True)
        # transform dst coordinates to local projection
        return transform.transform(x, y)
