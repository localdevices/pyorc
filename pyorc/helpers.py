"""Helper functions for pyorc."""

import copy
import importlib.util
import json

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio import fill, warp
from rasterio.crs import CRS
from rasterio.transform import Affine, xy
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes
from scipy.optimize import differential_evolution
from scipy.signal import convolve2d, fftconvolve


def _check_cartopy_installed():
    if importlib.util.find_spec("cartopy") is None:
        raise ModuleNotFoundError(
            'Geographic plotting requires cartopy. Please install it with "conda install cartopy" and try again.'
        )


def _import_cartopy_modules():
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    return ccrs, cimgt


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


def densify_points(points, sample_size=1000):
    """Increase the amount of points through linear interpolation. points are assumed to be sorted in space.

    Parameters
    ----------
    points : np.ndarray
        at minimum one-dimensional array with point coordinates

    sample_size : int
        amount of samples to expand to by linear interpolation

    """
    idx = np.arange(len(points))
    # generate interpolation function
    f = interp1d(np.arange(len(points)), points, axis=0)
    # expand to amount of samples
    return f(np.linspace(0, idx.max(), sample_size))


def depth_integrate(depth, v, v_corr=0.85, name="q"):
    """Integrate velocities [m s-1] to depth-integrated velocity [m2 s-1] using depth information.

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
    data_array : xr.DataArray
        attributes of interest
    attr : str
        name of attributes
    dtype : object type, optional
        function will try to perform type(eval(attr)), default np.array
    args_parse : boolean, optional
        if True, function will try to return type(*eval(attr)), assuming attribute contains list
        of arguments (default: False)

    Returns
    -------
    parsed_arg : type defined by user
        parsed attribute, of type defined by arg type

    """
    assert hasattr(data_array, attr), f'frames do not contain attribute with name "{attr}'
    attr_obj = getattr(data_array, attr)
    if args_parse:
        return dtype(*eval(attr_obj))
    return dtype(eval(attr_obj))


def get_axes(cols, rows, x, y):
    """Retrieve a locally spaced axes for surface velocimetry results.

    axes are based on resolution and row and col distances from the original frames

    Parameters
    ----------
    cols: list
        ints, columns, sampled from the original projected frames
    rows: list
        ints, rows, sampled from the original projected frames
    x: array-like
        frames x-axis
    y: array-like
        frames y-axis

    Returns
    -------
    xax : np.ndarray
        x-axis sampled from columns
    yax : np.ndarray
        y-axis sampled from columns

    """
    xax = x[cols]
    yax = y[rows]
    return xax, yax


def get_geo_axes(tiles=None, extent=None, zoom_level=19, **kwargs):
    """Prepare a geographical axis, possibly with an image tiler background.

    Parameters
    ----------
    tiles : str
        Name of cartopy.io.img_tiles tiler
    extent : List[int]
        [xmin, xmax, ymin, ymax] extent in longitude, latitude coordinates
    zoom_level : int
        zoom level to use for image tiler, increase if you want higher resolution imagery
    **kwargs : dict
        additional keyword arguments to pass to the chosen image tiler.

    Returns
    -------
    ax
        Geographical cartopy axis object

    """
    _check_cartopy_installed()
    #
    # try:
    #     import cartopy
    #     import cartopy.crs as ccrs
    #     import cartopy.io.img_tiles as cimgt
    # except ModuleNotFoundError:
    #     raise ModuleNotFoundError(
    #         'Geographic plotting requires cartopy. Please install it with "conda install cartopy" and try ' "again."
    #     )
    ccrs, cimgt = _import_cartopy_modules()
    if tiles is not None:
        tiler = getattr(cimgt, tiles)(**kwargs)
        crs = tiler.crs
    else:
        tiler = None
        crs = ccrs.PlateCarree()
    ax = plt.subplot(projection=crs)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    if tiles is not None:
        ax.add_image(tiler, zoom_level, zorder=1)
    return ax


def get_enclosed_mask(data, stride=2):
    """Create zero/one binary mask that can be used to fill holes.

    The function fills holes in yes/no finite data in a numpy array with data and nans and
    only fills where NaN areas are entirely enclosed by finite values.

    Parameters
    ----------
    data : 2D numpy array
        dataset for which mask should be created
    stride : int, optional
        amount of cells that will be parsed at the edges to ensure holes at edges are also filled.

    Returns
    -------
    mask : 2D numpy array
        zero/one array containing which areas are expected to contain data after filling procedures


    """
    mask = np.zeros(data.shape)
    mask[np.isfinite(data)] = 1

    # explode mask by one pixel in all directions
    mask_edge = np.minimum(fftconvolve(mask, np.ones((stride * 2 + 1, stride * 2 + 1))), 1)

    # mask_edge = np.minimum(convolve2d(np.ones((stride*2 + 1, stride*2 + 1)), mask), 1)
    mask_edge[stride:-stride, stride:-stride] = mask

    # fill holes of areas that are entirely enclosed
    mask_edge_fill = binary_fill_holes(mask_edge)

    mask_filled = np.int8(mask_edge_fill[stride:-stride, stride:-stride])
    mask_filled[mask_filled == 0] = -1
    mask_filled[mask_filled == 1 & np.isnan(data)] = 0
    return mask_filled


def get_rotation_code(rotation_code):
    """Convert rotation in degrees into a opencv rotation code.

    Parameters
    ----------
    rotation_code : Literal[0, 90, 180, 270]
        degrees rotation of a video or image

    Returns
    -------
    int
        opencv rotation code

    """
    if rotation_code not in [0, 90, 180, 270, None]:
        raise ValueError(f"Rotation code must be in allowed codes 0, 90, 180 or 270. Provided code is {rotation_code}")
    if rotation_code == 90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rotation_code == 180:
        return cv2.ROTATE_180
    elif rotation_code == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return None


def get_xs_ys(cols, rows, transform):
    """Compute rasters of x and y coordinates, based on row and column counts and a defined transform.

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


def get_lons_lats(xs, ys, src_crs, dst_crs=4326):
    """Create lon-lat rasters from local projection coordinate grid.

    Computes raster of longitude and latitude coordinates (default) of a certain raster set of coordinates in a local
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
    dst_crs = CRS.from_user_input(dst_crs)
    lons, lats = warp.transform(src_crs, dst_crs, xs.flatten(), ys.flatten())
    lons, lats = (
        np.array(lons).reshape(xs.shape),
        np.array(lats).reshape(ys.shape),
    )
    return lons, lats


def log_profile(x, z0, k_max, s0=0.0, s1=0.0):
    """Return values of a log-profile function.

    Parameters
    ----------
    x: tuple with np.ndarrays
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
    z, s = x
    k = k_max * np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
    v = k * np.maximum(np.log(np.maximum(z, 1e-6) / z0), 0)
    return v


def pixel_to_map(cols, rows, transform):
    """Replace `transform.xy` in numpy using `rasterio.transform` order. This is much faster than rasterio.

    Parameters
    ----------
    cols : np.ndarray
        columns set
    rows : np.ndarray
        rows set
    transform : array-like or transform object
        transform of raster

    Returns
    -------
    xs : array-like
        x-coordinates
    ys : array-like
        y-coordinates

    """
    # Affine transformation (assuming transform is the raster's transform)
    x_map = transform[2] + rows * transform[1] + cols * transform[0]
    y_map = transform[5] + rows * transform[4] + cols * transform[3]

    return x_map, y_map


def map_to_pixel(xs, ys, transform):
    """Transform with `numpy` similar to `transform.rowcol` using `rasterio.transform`.

    Numpy is much faster than rasterio.

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates
    ys : np.ndarray
        y-coordinates
    transform : array-like or transform object
        transform of raster

    Returns
    -------
    rows : array-like
        row coordinates
    cols : array-like
        column coordinates

    """
    # Calculate the determinant of the upper-left 2x2 submatrix
    det = transform[1] * transform[3] - transform[0] * transform[4]

    # Calculate the inverse of the upper-left 2x2 submatrix
    inv_det = 1.0 / det
    inv_transform = [transform[3] * inv_det, -transform[0] * inv_det, -transform[4] * inv_det, transform[1] * inv_det]

    # Calculate the offsets
    dx = xs - transform[2]
    dy = ys - transform[5]

    # Calculate the pixel coordinates
    row = np.int64(np.round(inv_transform[0] * dx + inv_transform[1] * dy))
    col = np.int64(np.round(inv_transform[2] * dx + inv_transform[3] * dy))

    return row, col


def mask_fill(data, mask, radius=5):
    """Fill data where np.nan is found, if mask at those location is zero.

    Areas where mask is one are used to fill up these areas. Areas where mask is -1 are not filled and kept np.nan

    Parameters
    ----------
    data : np.ndarray (2D)
        data values with possibly np.nan values to fill
    mask : np.ndarray (2D, type int8)
        mask values to apply, either -1, 0 or 1
    radius : int, float
        search distance radius [pix]


    Returns
    -------
    data_fill : np.ndarray (2D)
        data filled

    """
    mask[np.isfinite(data)] = 1
    data_fill = copy.deepcopy(fill.fillnodata(data, mask=mask == 1, max_search_distance=radius))
    data_fill[mask == -1] = 0
    return data_fill


def mse(pars, func, x, y):
    """Give mean of sum of squares between evaluation of function with provided parameters and observations.

    Parameters
    ----------
    pars : list or tuple
        parameter passed as *args to func.
    func: function def
        receiving X and *pars as input and returning predicted Y as result
    x: tuple with lists or array-likes
        indepent variable(s).
    y: list or array-like
        dependent variable, predicted by func

    Returns
    -------
    y_pred : list or array-like
        predicted Y from X and pars by func

    """
    y_pred = func(x, *pars)
    ms_error = np.sum((y_pred - y) ** 2)
    return ms_error


def neighbour_stack(array, stride=1, missing=-9999.0):
    """Build stack of arrays from a 2-D input array, constructed by permutation in space using a provided stride.

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
    bounds=([0.001, 0.1], [-20, 20], [0.0, 5], [0.0, 100]),
    workers=2,
    popsize=100,
    updating="deferred",
    seed=0,
    **kwargs,
):
    """Optimize velocity log profile relation of v=k*max(z/z0) with k a function of distance to bank and k_max.

    A differential evolution optimizer is used.

    Parameters
    ----------
    z : list
        depths [m]
    v : list
        surface velocities [m s-1]
    dist_bank : list, optional
        distances to bank [m]
    bounds : tuple of int-pairs, optional
        search boundaries for the optimization function, default: ([0.001, 0.1], [-20, 20], [0.0, 5], [0.0, 100])
    workers : int, optional
        Amount of workers used to optimize log profile function (default=2) in `scipy.optimize.differential_evolution`
    popsize : int, optional
        size of search population (default: 100) in `scipy.optimize.differential_evolution`
    updating : str, optional
        method of updating in `scipy.optimize.differential_evolution` (default: "deferred").
    seed : float, optional
        seed point used in `scipy.optimize.differential_evolution` (default: 0).
    **kwargs : dict
        additional keyword arguments for scipy.optimize.differential_evolution

    Returns
    -------
    pars : dict
        fitted parameters of log_profile {z_0, k_max, s0 and s1}

    """
    # replace by infinites if not provided
    dist_bank = np.ones(len(v)) * np.inf if dist_bank is None else dist_bank
    v = np.array(v)
    z = np.array(z)
    x = (z, dist_bank)
    y = v
    result = differential_evolution(
        wrap_mse,
        args=(log_profile, x, y),
        bounds=bounds,
        workers=workers,
        popsize=popsize,
        updating=updating,
        seed=seed,
        **kwargs,
    )
    # unravel parameters
    z0, k_max, s0, s1 = result.x
    return {"z0": z0, "k_max": k_max, "s0": s0, "s1": s1}


def read_shape_safe_crs(fn):
    """Read a shapefile with geopandas, but ensure that CRS is set to None when not available.

    This function is required in cases where geometries must be read that do not have a specified CRS. Geopandas
    defaults to WGS84 EPSG 4326 if the CRS is not specified.
    """
    gdf = gpd.read_file(fn)
    # also read raw json, and check if crs attribute exists
    if isinstance(fn, str):
        with open(fn, "r") as f:
            raw_json = json.load(f)
    else:
        # apparently a file object was provided
        fn.seek(0)
        raw_json = json.load(fn)
    if "crs" not in raw_json:
        # override the crs
        gdf = gdf.set_crs(None, allow_override=True)
    return gdf


def rotate_u_v(u, v, theta, deg=False):
    """Rotate u and v components of vector counterclockwise by an amount of rotation.

    Parameters
    ----------
    u : float, np.ndarray or xr.DataArray
        x-direction component of vector
    v : float, np.ndarray or xr.DataArray
        y-direction component of vector
    theta : float
        amount of counterclockwise rotation in radians or degrees (dependent on deg)
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
    """Round number to a multiple of a certain number."""
    return multiple * round(number / multiple)


def stack_window(ds, wdw=1, wdw_x_min=None, wdw_x_max=None, wdw_y_min=None, wdw_y_max=None, dim="stride"):
    """Stack moved windows together over a new dimension.

    Typically used to average or take a quantile over a window of pixels in space

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to stack over
    wdw : int (positive), optional
        general all-directional stride to use (default: 1)
    wdw_x_min : int, optional
        Stride to use in negative x-axis direction, if not provided, -`wdw` is used.
    wdw_x_max
        Stride to use in positive x-axis direction, if not provided, `wdw` is used.
    wdw_y_min
        Stride to use in negative y-axis direction, if not provided, -`wdw` is used.
    wdw_y_max
        Stride to use in positive y-axis direction, if not provided, `wdw` is used.
    dim : str, optional
        Name of new stack dimension

    Returns
    -------
    xr.Dataset
        Containing all variables of `ds`, but then window shifted over x and y dimension and stacked.

    """
    # set strides
    wdw_x_min = -wdw if wdw_x_min is None else wdw_x_min
    wdw_x_max = wdw if wdw_x_max is None else wdw_x_max
    wdw_y_min = -wdw if wdw_y_min is None else wdw_y_min
    wdw_y_max = wdw if wdw_y_max is None else wdw_y_max

    return xr.concat(
        [
            ds.shift(x=x_stride, y=y_stride)
            for x_stride in range(wdw_x_min, wdw_x_max + 1)
            for y_stride in range(wdw_y_min, wdw_y_max)
        ],
        dim=dim,
    )


def staggered_index(start=0, end=100):
    """Create staggered indexes that start at the outer indexes and gradually move inwards.

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
    """Fill missing surface velocities using a velocity log-depth model, fitted with the known points in profile.

    Parameters
    ----------
    v : xr.DataArray (time, points)
        effective velocity at surface [m s-1]
    depth : xr.DataArray (points)
        bathymetry depths [m]
    dist_shore : xr.DataArray (points)
        the shortest distance to a dry river bed point
    dim: str, optional
        dimension over which data should be grouped, default: "quantile", dimension must exist in v, typically
        "quantile" or "time"

    Returns
    -------
    v_fill: xr.DataArray (quantile or time, points)
        effective surface velocities with interpolated values added  [m s-1]

    """

    def log_fit(_v):
        idx_finite = np.isfinite(_v).values.flatten()
        pars = optimize_log_profile(depth[idx_finite], _v[0, idx_finite], dist_shore[idx_finite])
        idx_miss = np.where(np.isnan(_v[0]).values)[0]
        _v[0, idx_miss] = log_profile((depth[idx_miss], dist_shore[idx_miss]), **pars)
        # enforce that velocities are zero with zero depth
        _v[0, depth <= 0] = 0.0
        return np.maximum(_v, 0)

    # fill per grouped dimension
    v.load()
    v_group = copy.deepcopy(v).groupby(dim, squeeze=False)
    return v_group.map(log_fit)


def velocity_log_interp(v, dist_wall, d_0=0.1, dim="quantile"):
    """Interpolate missing velocities over log-transform with depth.

    Parameters
    ----------
    v : xr.DataArray (time, points) or (quantile, points)
        effective velocity at surface [m s-1]
    dist_wall : xr.DataArray (points)
        the shortest distance to the river bed
    d_0 : float, optional
        roughness length (default: 0.1)
    dim: str, optional
        dimension over which data should be grouped, default: "quantile", dimension must exist in v, typically
        "quantile" or "time"

    Returns
    -------
    xr.DataArray (time, points) or (quantile, points)
        effective velocities with interpolated values added [m s-1]

    """

    def log_interp(_v):
        # scale with log depth
        c = xr.DataArray(_v / np.log(np.maximum(dist_wall, d_0) / d_0))
        # fill dry points with the nearest valid value for c
        c[0, np.where(dist_wall == 0)[0]] = c.interpolate_na(dim="points", method="nearest", fill_value="extrapolate")[
            0, np.where(dist_wall == 0)[0]
        ]
        # interpolate with linear interpolation
        c = c.interpolate_na(dim="points")
        # use filled c to interpret missing v
        _v[0, np.isnan(_v[0])] = (np.log(np.maximum(dist_wall, d_0) / d_0) * c)[
            0, np.where(np.isnan(_v[0]))[0]
        ]  # (np.log(np.maximum(dist_wall, d_0) / d_0) * c)[np.isnan(_v)]
        return _v

    # fill per grouped dimension
    v.load()
    v_group = copy.deepcopy(v).groupby(dim, squeeze=False)
    return v_group.map(log_interp)


def wrap_mse(pars_iter, *args):
    """Wrap function mse for optimization purposes."""
    return mse(pars_iter, *args)


def xy_equidistant(x, y, distance, z=None):
    """Transform ordered in space x, y (and z if provided) coordinates into equal spaced x, y (and z) coordinates.

    The 1-dimensional distance between points is used with piece-wise linear interpolation. Extrapolation is used for
    the last point to ensure the range of points covers at least the full range of x, y coordinates.

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
    s = np.cumsum((x_diff**2 + y_diff**2) ** 0.5)

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


def xy_to_perspective(x, y, resolution, trans_mat, reverse_y=None):
    """Back transform local meters-from-top-left coordinates from frame to original perspective of camera.

    Back transform is done using M matrix, belonging to transformation from orthographic to local.

    Parameters
    ----------
    x : np.ndarray (1D)
        axis of x-coordinates in local projection with origin top-left, to be backwards projected
    y : np.ndarray (1D)
        axis of y-coordinates in local projection with origin top-left, to be backwards projected
    resolution : float
        resolution of original projected frames coordinates of x and y
    trans_mat : np.ndarray
        2x3 transformation matrix (generated with cv2.getPerspectiveTransform)
    reverse_y : int, optional
        if set, it should be set to the amount of rows in the frame, then y will be transformed following
        y = reverse_y - y

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
    coords_trans = cv2.perspectiveTransform(coords, trans_mat)
    xp = coords_trans[0][:, 0].reshape(cols.shape)
    yp = coords_trans[0][:, 1].reshape(cols.shape)
    return xp, yp


def xyz_transform(points, crs_from, crs_to):
    """Transform set of x and y coordinates from one CRS to another.

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
    # with only one point, transformer must not provide an array of points, to prevent a deprecation warning numpy>=1.25
    if len(points) == 1:
        x = x[0]
        y = y[0]
    # transform dst coordinates to local projection
    x_trans, y_trans = transform.transform(x, y)
    # check if finites are found, if not raise error
    assert not (
        np.all(np.isinf(x_trans))
    ), "Transformation did not give valid results, please check if the provided crs of input coordinates is correct."
    points[:, 0] = np.atleast_1d(x_trans)
    points[:, 1] = np.atleast_1d(y_trans)
    return points.tolist()
    # return transform.transform(x, y)
