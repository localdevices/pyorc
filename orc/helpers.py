import numpy as np
import dask.array as da
import xarray as xr
from scipy.signal import convolve2d

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
    # assert(len(y)==shape[0]), f"Length of y-axis {len(y)} is not equal to expected shape in y-direction {shape[0]}"
    # assert(len(x)==shape[1]), f"Length of x-axis {len(x)} is not equal to expected shape in x-direction {shape[1]}"
    # data_array = [da.from_delayed(
    #     d,
    #     dtype=dtype,
    #     shape=shape
    # ) for d in delayed_das]
    return xr.DataArray(
        data_array,
        dims=tuple(coords.keys()),
        coords=coords,
        attrs=attrs,
        name=name
    )
