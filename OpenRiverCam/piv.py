import openpiv.tools
import openpiv.pyprocess
import numpy as np
import xarray as xr

from scipy.signal import convolve2d

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


def integrate_flow(q, quantile=0.5):
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
    Q = q.quantile(quantile, dim="time").fillna(0.0).integrate(dim="dist")
    Q.attrs = {
        "standard_name": "river_discharge",
        "long_name": "River Flow",
        "units": "m3 s-1",
    }
    # set name
    Q.name = "Q"
    return Q

def neighbour_stack(array, stride=1, missing=-9999.):
    """
    Builds a stack of arrays from a 2-D input array, with its neighbours using a provided stride
    :param array: 2-D numpy array with values (may contain NaN)
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return:
    stack of arrays, with all neighbours included

    """
    array = np.copy(array)
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


def vector_to_scalar(v_x, v_y):
    """
    Turns velocity vectors into effective velocities over cross-section, by computing the perpendicular velocity component
    :param v_x: DataArray(t, points), time series in cross section points with x-directional velocities
    :param v_y: DataArray(t, points), time series in cross section points with y-directional velocities
    :return: v_eff: DataArray(t, points), time series in cross section points with velocities perpendicular to cross section
    """
    xs = v_x["x"].values
    ys = v_x["y"].values
    # find points that are located on the area of interest
    idx = np.isfinite(xs)
    xs = xs[idx]
    ys = ys[idx]
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
                angle_left = np.nan
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
    frame_a, frame_b, res_x=1.0, res_y=1.0, search_area_size=60, overlap=30, correlation=True, window_size=None, **kwargs
):
    """
    Typical kwargs are for instance
    window_size=60, overlap=30, search_area_size=60, dt=1./25
    :param fn1:
    :param fn2:
    :param res_x: float, resolution of x-dir pixels in a user-defined unit per pixel (e.g. m pixel-1)
    :param res_y: float, resolution of y-dir pixels in a user-defined unit per pixel (e.g. m pixel-1)
    :param kwargs:
    :return:
    """
    window_size = search_area_size if window_size is None else window_size
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
            search_area_size,
            overlap,
            window_size,
            **kwargs
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
    **kwargs
):
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

def filter_temporal(
    ds,
    v_x="v_x",
    v_y="v_y",
    filter_std=True,
    filter_angle=True,
    filter_velocity=True,
    filter_corr=True,
    filter_neighbour=True,
    kwargs_std={},
    kwargs_angle={},
    kwargs_velocity={},
    kwargs_corr={},
    kwargs_neighbour={},
):
    """
    Implementation of several filters that use temporal variations or comparison as basis
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param filter_std: boolean, if True (default, filtering on variance is applied)
    :param filter_angle: boolean, if True (default, filtering on angles is applied)
    :param filter_velocity: boolean, if True (default, filtering on velocity is applied)
    :param filter_corr: boolean, if True (default, filtering on correlation is applied)
    :param kwargs_std: dict, set of key-word arguments to pass on to filter_temporal_std
    :param kwargs_angle: dict, set of key-word arguments to pass on to filter_temporal_angle
    :param kwargs_velocity: dict, set of key-word arguments to pass on to filter_temporal_velocity
    :param kwargs_corr: dict, set of key-word arguments to pass on to filter_temporal_corr
    :return: xarray Dataset, containing temporally filtered velocity vectors as [time, y, x]
    """
    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
    if filter_std:
        ds = filter_temporal_std(ds, v_x=v_x, v_y=v_y, **kwargs_std)
    if filter_angle:
        ds = filter_temporal_angle(ds, v_x=v_x, v_y=v_y, **kwargs_angle)
    if filter_velocity:
        ds = filter_temporal_velocity(ds, v_x=v_x, v_y=v_y, **kwargs_velocity)
    if filter_corr:
        ds = filter_temporal_corr(ds, v_x=v_x, v_y=v_y, **kwargs_corr)
    if filter_neighbour:
        ds = filter_temporal_neighbour(ds, v_x=v_x, v_y=v_y, **kwargs_neighbour)
    return ds

def filter_temporal_angle(
    ds,
    v_x="v_x",
    v_y="v_y",
    angle_expected=0.5 * np.pi,
    angle_tolerance=0.25 * np.pi,
    filter_per_timestep=True,
):
    """
    filters on the expected angle. The function filters points entirely where the mean angle over time
    deviates more than input parameter angle_bounds (in radians). The function also filters individual
    estimates in time, in case the user wants this (filter_per_timestep=True), in case the angle on
    a specific time step deviates more than the defined amount from the average.
    note: this function does not work appropriately, if the expected angle (+/- anglebounds) are within
    range of zero, as zero is the same as 2*pi. This exception may be resolved in the future if necessary
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param angle_expected (=0.5*np.pi, assumes that flow is from left to right): angle [radians, 0-2*pi],
        measured clock-wise from vertical upwards direction, expected in the velocites
    :param angle_tolerance: float [0-2*pi] maximum deviation from expected angle allowed.
    :param filter_per_timestep (=True): if set to True, tolerances are also checked per individual time step
    :return: xarray Dataset, containing angle filtered velocity vectors as [time, y, x]
    """
    # TODO: make function working appropriately, if angles are close to zero (2*pi)
    angle = np.arctan2(ds[v_x], ds[v_y])
    angle_mean = angle.mean(dim="time")
    ds[v_x] = ds[v_x].where(np.abs(angle_mean - angle_expected) < angle_tolerance)
    ds[v_y] = ds[v_y].where(np.abs(angle_mean - angle_expected) < angle_tolerance)
    if filter_per_timestep:
        ds[v_x] = ds[v_x].where(np.abs(angle - angle_expected) < angle_tolerance)
        ds[v_y] = ds[v_y].where(np.abs(angle - angle_expected) < angle_tolerance)
    return ds

def filter_temporal_neighbour(ds, v_x="v_x", v_y="v_y", roll=5, tolerance=0.5):
    """
    Filters values if neighbours over a certain rolling length before and after, have a
    significantly higher velocity, measured by tolerance
    :param ds:
    :param v_x:
    :param v_y:
    :param stride:
    :param tolerance: Relative acceptable velocity of maximum found within stride
    :return: ds
    """
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5

    s_roll = s.fillna(0.).rolling(time=roll, center=True).max()
    ds[v_x] = ds[v_x].where(s > tolerance*s_roll)
    ds[v_y] = ds[v_y].where(s > tolerance*s_roll)
    return ds


def filter_temporal_std(
    ds, v_x="v_x", v_y="v_y", std_thres=1.0, filter_per_timestep=True
):
    """

    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param std_thres: float, representing a maximum standard
    :param filter_per_timestep:
    :return:
    """
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5
    s_std = s.std(dim="time")
    s_mean = s.mean(dim="time")
    s_var = s_std / s_mean

    ds[v_x] = ds[v_x].where(s_var < std_thres)
    ds[v_y] = ds[v_y].where(s_var < std_thres)
    if filter_per_timestep:
        ds[v_x] = ds[v_x].where((s - s_mean) / s_std < std_thres)
        ds[v_y] = ds[v_y].where((s - s_mean) / s_std < std_thres)

    return ds


def filter_temporal_velocity(ds, v_x="v_x", v_y="v_y", s_min=0.1, s_max=5.0):
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5
    ds[v_x] = ds[v_x].where(s > s_min)
    ds[v_y] = ds[v_y].where(s < s_max)
    return ds

def filter_temporal_corr(ds, v_x="v_x", v_y="v_y", corr="corr", corr_min=0.4):
    ds[v_x] = ds[v_x].where(ds[corr] > corr_min)
    ds[v_y] = ds[v_y].where(ds[corr] > corr_min)
    return ds

def filter_spatial(
    ds,
    kwargs_nan={},
    kwargs_median={},
    v_x="v_x",
    v_y="v_y",
):
    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
    ds_g = ds.groupby("time")
    ds = ds_g.apply(filter_spatial_nan, v_x=v_x, v_y=v_y, **kwargs_nan)
    ds_g = ds.groupby("time")
    ds = ds_g.apply(filter_spatial_median, v_x=v_x, v_y=v_y, **kwargs_median)
    return ds


def filter_spatial_nan(ds, v_x="v_x", v_y="v_y", max_nan_frac=0.8, stride=1, missing=-9999.):
    u, v = ds[v_x].values, ds[v_y].values
    u_move = neighbour_stack(u, stride=stride)
    # replace missings by Nan
    nan_frac = np.float64(np.isnan(u_move)).sum(axis=0)/float(len(u_move))
    u[nan_frac > max_nan_frac] = np.nan
    v[nan_frac > max_nan_frac] = np.nan

    # u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds[v_x][:] = u
    ds[v_y][:] = v
    return ds

def filter_spatial_median(ds, v_x="v_x", v_y="v_y", max_rel_diff=0.7, stride=1, missing=-9999.):
    u, v = ds[v_x].values, ds[v_y].values
    s = (u**2 + v**2)**0.5
    s_move = neighbour_stack(s, stride=stride)
    # replace missings by Nan
    s_median = np.nanmedian(s_move, axis=0)
    # now filter points that are very far off from the median
    filter = np.abs(s - s_median)/s_median > max_rel_diff
    u[filter] = np.nan
    v[filter] = np.nan

    # u, v, mask = validation.local_median_val(v_x, v_y, 0.5, 0.5, size=1)
    ds[v_x][:] = u
    ds[v_y][:] = v
    return ds

def replace_outliers(ds, v_x="v_x", v_y="v_y", stride=1, max_iter=1):
    u, v = ds[v_x].values, ds[v_y].values
    for n in range(max_iter):
        u_move = neighbour_stack(u, stride=stride)
        v_move = neighbour_stack(v, stride=stride)
        # compute mean
        u_mean = np.nanmean(u_move, axis=0)
        v_mean = np.nanmean(v_move, axis=0)
        u[np.isnan(u)] = u_mean[np.isnan(u)]
        v[np.isnan(v)] = v_mean[np.isnan(v)]
        # all values with stride distance from edge have to be made NaN
        u[0:stride, :] = np.nan; u[-stride:, :] = np.nan; u[:, 0:stride] = np.nan; u[:, -stride:] = np.nan
        v[0:stride, :] = np.nan; v[-stride:, :] = np.nan; v[:, 0:stride] = np.nan; v[:, -stride:] = np.nan
    ds[v_x][:] = u
    ds[v_y][:] = v
    return ds
