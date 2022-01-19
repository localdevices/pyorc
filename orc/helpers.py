import cv2
import dask.array as da
import numpy as np
import xarray as xr
from scipy.signal import convolve2d


def add_xy_coords(ds, xy_coord_data, coords, attrs_dict):
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
    Convert a list of delayed 2D arrays (assumed to be time steps of grids) into a 3D DataArray with all axes

    :param delayed_das: Delayed dask data arrays (2D) or list of 2D delayed dask arrays
    :param shape: tuple, foreseen shape of data arrays (rows, cols)
    :param dtype: string or dtype, e.g. "uint8" of data arrays
    :param time: time axis to use in data array
    :param y: y axis to use in data array
    :param x: x axis to use in data array
    :param attrs: dict, containing attributes for data array
    :param name: str, name of attribute, default None
    :return: xr.DataArray
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
    Return a deserialized version of said property (assumed to be stored as a string) of DataArray

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
    Builds a stack of arrays from a 2-D input array, with its neighbours using a provided stride

    :param array: 2-D numpy array, any values (may contain NaN)
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: 3-D numpy array, stack of 2-D arrays, with strided neighbours

    """
    # array = np.copy(array)
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

def plot_as_colormesh(image, ax, **pcolormeshkwargs):
    # https://gist.github.com/tatome/e029e61a50d39090eb56
    raveled_pixel_shape = (image.shape[0]*image.shape[1], image.shape[2])
    color_tuple = image.values.transpose((1,0,2)).reshape(raveled_pixel_shape)

    if color_tuple.dtype == np.uint8:
        color_tuple = color_tuple / 255.

    # index = np.tile(np.arange(image.shape[0]), (image.shape[1],1))
    quad = ax.pcolormesh(image.lon.values, image.lat.values, image.values.mean(axis=-1), facecolor=color_tuple, linewidth=0., **pcolormeshkwargs)  # shading="nearest"
    quad.set_array(None)

def rotate_u_v(u, v, theta, deg=False):
    """
    Rotate u and v components of vector counter clockwise by an amount of rotation

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
    Back transform local col and row locations from ortho projected frame to original perspective of camera using M
    matrix, belonging to transformation from orthographic to local

    :param x: np.ndarray, 1D axis of x-coordinates in local projection with origin top-left, to be backwards projected
    :param y: np.ndarray, 1D axis of y-coordinates in local projection with origin top-left, to be backwards projected
    :param resolution: resolution of original projected frames coordinates of x and y
    :param M: transformation matrix generated with cv2.getPerspectiveTransform
    :return: (xp, yp), np.ndarray of shape (len(y), len(x)) containing perspective columns (xp) and rows (yp) of data
    """
    # make a mesgrid of cols and rows
    cols_i, rows_i = np.meshgrid(x / resolution - 0.5, y / resolution - 0.5)
    # make list of coordinates, compatible with cv2.perspectiveTransform
    coords = np.float32([np.array([cols_i.flatten(), rows_i.flatten()]).transpose([1, 0])])
    coords_trans = cv2.perspectiveTransform(coords, M)
    xp = coords_trans[0][:, 0].reshape((len(y), len(x)))
    yp = coords_trans[0][:, 1].reshape((len(y), len(x)))
    return xp, yp
