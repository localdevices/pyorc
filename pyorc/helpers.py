import cv2
import dask.array as da
import numpy as np
import xarray as xr
from scipy.signal import convolve2d

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
            array_move.append(convolve2d(array, conv_arr, mode="same"))
    array_move = np.stack(array_move)
    # replace missings by Nan
    array_move[np.isclose(array_move, missing)] = np.nan
    return array_move

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
