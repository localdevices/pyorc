import openpiv.tools
import openpiv.pyprocess
import numpy as np
import xarray as xr
import copy

from scipy.optimize import curve_fit

def log_profile(z, z0, k):
    return k*np.maximum(np.log(np.maximum(z, 1e-6)/z0), 0)

def optimize_log_profile(z, v):
    """
    optimize velocity log profile relation of v=k*max(z/z0)
    :param z: list of depths
    :param v: list of surface velocities
    :return: {z_0, k}
    """

    result = curve_fit(log_profile,
                       np.array(z),
                       np.array(v),
                       bounds=([0.05, 0.1], [0.050000001, 20]),
                       p0=[0.05, 2])
    z0, k = result[0]
    return {"z0": z0, "k": k}

def depth_integrate(z, v, z_0, h_a, v_corr=0.85):
    """

    :param z: DataArray(points), bathymetry depths (ref. CRS)
    :param v: DataArray(time, points), effective velocity at surface [m s-1]
    :param z_0: float, zero water level (ref. CRS)
    :param h_a: float, actual water level (ref. z_0)
    :param v_corr: float (range: 0-1, typically close to 1), correction factor from surface to depth-average (default: 0.85)
    :return: q: DataArray(time, points), depth integrated velocity [m2 s-1]
    """
    # compute depth, never smaller than zero. Depth is in words:
    #   + height of the zero level of staff gauge (z_0) measured in gps CRS (e.g. WGS84)
    #   + actual water height as measured by staff gauge (i.e. meters above z_0)
    #   - z levels the bottom cross section observations measured in gps CRS (e.g. WGS84)
    #   of course depth cannot be negative, so it is always maximized to zero when below zero
    depth = np.maximum(z_0 + h_a - z, 0)
    # compute the depth average velocity
    q = v * v_corr * depth
    q.attrs = {
        "standard_name": "velocity_depth",
        "long_name": "velocity averaged over depth",
        "units": "m2 s-1",
    }
    # set name
    q.name = "q"
    return q

def velocity_fill(z, v, z_0, h_a):
    def fit(_v):
        pars = optimize_log_profile(depth[np.isfinite(_v)], _v[np.isfinite(_v)])
        _v[np.isnan(_v)] = log_profile(depth[np.isnan(_v)], **pars)
        return _v
    depth = np.maximum(z_0 + h_a - z, 0)
    # per slice, fill missings
    v_group = copy.deepcopy(v).groupby("quantile")
    return v_group.apply(fit)

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


def imread(fn):
    return openpiv.tools.imread(fn)


def integrate_flow(q):
    """
    Integrates time series of depth averaged velocities [m2 s-1] into cross-section integrated flow [m3 s-1]
    estimating one or several quantiles over the time dimension.
    :param q: DataArray(time, points) depth integrated velocities [m2 s-1] over cross section
    :param quantile: float or list of floats (range: 0-1)  (default: 0.5)
    :return: Q: DataArray(quantile) River Flow [m3 s-1] for one or several quantiles. The time dimension no longer
        exists because of the quantile mapping, the point dimension no longer exists because of the integration over width
    """
    dist = [0.0]
    for n, (x1, y1, x2, y2) in enumerate(
        zip(q.xcoords[:-1], q.ycoords[:-1], q.xcoords[1:], q.ycoords[1:])
    ):
        _dist = distance_pts((x1, y1), (x2, y2))
        dist.append(dist[n] + _dist)

    # assign coordinates for distance
    q = q.assign_coords(dist=("points", dist))

    # if any missings are still present, fill with 0.0, integrate of dim dist
    Q = q.fillna(0.0).integrate(coord="dist")
    Q.attrs = {
        "standard_name": "river_discharge",
        "long_name": "River Flow",
        "units": "m3 s-1",
    }
    # set name
    Q.name = "Q"
    return Q


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


def piv(
    frame_a, frame_b, res_x=0.01, res_y=0.01, search_area_size=30, correlation=True, window_size=None, overlap=None,
        **kwargs
):
    """
    PIV analysis following keyword arguments from openpiv. This function also computes the correlations per
    interrogation window, so that poorly correlated values can be filtered out. Furthermore, the resolution is used to convert
    pixel per second velocity estimates, into meter per second velocity estimates. The centre of search area columns
    and rows are also returned so that a georeferenced grid can be written from the results.

    Note: Typical openpiv kwargs are for instance
    window_size=60, overlap=30, search_area_size=60, dt=1./25
    :param frame_a: 2-D numpy array, containing first frame
    :param frame_b: 2-D numpy array, containing second frame
    :param res_x: float, resolution of x-dir pixels in a user-defined unit per pixel (e.g. m pixel-1)
    :param res_y: float, resolution of y-dir pixels in a user-defined unit per pixel (e.g. m pixel-1)
    :param kwargs: dict, several keyword arguments related to openpiv. See openpiv manual for further information
    :return cols: 1-D numpy array, col number of centre of interrogation windows
    :return rows: 1-D numpy array, row number of centre of interrogation windows
    :return v_x: 2-D numpy array, raw x-dir velocities [m s-1] in interrogation windows (requires filtering to get
        valid velocities)
    :return v_y: 2-D numpy array, raw y-dir velocities [m s-1] in interrogation windows (requires filtering to get
        valid velocities)
    :return s2n: 2-D numpy array, signal to noise ratio, measured as maximum correlation found divided by the mean
        correlation (method="peak2mean") or second to maximum correlation (method="peak2peak") found within search area
    :return corr: 2-D numpy array, correlation values in interrogation windows

    """
    if isinstance(frame_a, xr.core.dataarray.DataArray):
        frame_a = frame_a.values
    if isinstance(frame_b, xr.core.dataarray.DataArray):
        frame_b = frame_b.values
    window_size = search_area_size if window_size is None else window_size
    overlap = int(round(window_size)/2) if overlap is None else overlap
    v_x, v_y, s2n = openpiv.pyprocess.extended_search_area_piv(
        frame_a, frame_b, search_area_size=search_area_size, overlap=overlap, window_size=window_size, **kwargs
    )
    cols, rows = openpiv.pyprocess.get_coordinates(
        image_size=frame_a.shape, search_area_size=search_area_size, overlap=overlap
    )

    if correlation:
        corr = piv_corr(
            frame_a,
            frame_b,
            search_area_size=search_area_size,
            overlap=overlap,
            window_size=window_size,
        )
    else:
        corr = np.zeros(s2n.shape)
        corr[:] = np.nan
    return cols, rows, v_x * res_x, v_y * res_y, s2n, corr


def piv_corr(
    frame_a,
    frame_b,
    search_area_size,
    overlap,
    window_size=None,
    correlation_method="circular",
    normalized_correlation=True,
):
    """
    Estimate the maximum correlation in piv analyses over two frames. Function taken from openpiv library.
    This is a temporary fix. If correlation can be exported from openpiv, then this function can be removed.
    :param frame_a: 2-D numpy array, containing first frame
    :param frame_b: 2-D numpy array, containing second frame
    :param search_area_size: int, size of search area in pixels (square shape)
    :param overlap: int, amount of overlapping pixels between search areas
    :param window_size: int, size of window to search for correlations
    :param correlation_method: method for correlation used, as openpiv setting
    :param normalized_correlation: return a normalized correlation number between zero and one.
    :return: maximum correlations found in search areas
    """
    # extract the correlation matrix
    window_size = search_area_size if window_size is None else window_size
    # get field shape

    n_rows, n_cols = openpiv.pyprocess.get_field_shape(
        frame_a.shape, search_area_size, overlap
    )

    # We implement the new vectorized code
    aa = openpiv.pyprocess.moving_window_array(frame_a, search_area_size, overlap)
    bb = openpiv.pyprocess.moving_window_array(frame_b, search_area_size, overlap)

    if search_area_size > window_size:
        # before masking with zeros we need to remove
        # edges

        aa = openpiv.pyprocess.normalize_intensity(aa)
        bb = openpiv.pyprocess.normalize_intensity(bb)

        mask = np.zeros((search_area_size, search_area_size)).astype(aa.dtype)
        pad = np.int((search_area_size - window_size) / 2)
        mask[slice(pad, search_area_size - pad), slice(pad, search_area_size - pad)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = openpiv.pyprocess.fft_correlate_images(
        aa,
        bb,
        correlation_method=correlation_method,
        normalized_correlation=normalized_correlation,
    )

    corr = corr.max(axis=-1).max(axis=-1).reshape((n_rows, n_cols))
    return corr

