import openpiv.tools
import openpiv.pyprocess
import numpy as np
import xarray as xr

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
    frame_a, frame_b, res_x=1.0, res_y=1.0, search_area_size=60, overlap=30, correlation=True, **kwargs
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
    v_x, v_y, s2n = openpiv.pyprocess.extended_search_area_piv(
        frame_a, frame_b, search_area_size=search_area_size, overlap=overlap, **kwargs
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

    corr = openpiv.pyprocess.fft_correlate_strided_images(
        aa,
        bb,
        correlation_method=correlation_method,
        normalized_correlation=normalized_correlation,
    )

    corr = corr.max(axis=-1).max(axis=-1).reshape((n_rows, n_cols))
    return corr

def piv_filter(
    ds,
    v_x="v_x",
    v_y="v_y",
    angle_expected=0.5 * np.pi,
    angle_bounds=0.25 * np.pi,
    var_thres=1.0,
    s_min=0.1,
    s_max=5.0,
    corr="corr",
    corr_min=0.3,

):
    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)

    ds_filter = piv_filter_variance(ds, v_x=v_x, v_y=v_y, var_thres=var_thres)
    ds_filter = piv_filter_angle(ds_filter, v_x=v_x, v_y=v_y, angle_expected=angle_expected, angle_bounds=angle_bounds)
    ds_filter = piv_filter_velocity(ds_filter, v_x=v_x, v_y=v_y, s_min=s_min, s_max=s_max)
    ds_filter = piv_filter_corr(ds_filter, v_x=v_x, v_y=v_y, corr=corr, corr_min=corr_min)
    return ds_filter


def piv_filter_angle(
    ds,
    v_x="v_x",
    v_y="v_y",
    angle_expected=0.5 * np.pi,
    angle_bounds=0.25 * np.pi,
    filter_per_timestep=True,

):
    angle = np.arctan2(ds[v_x], ds[v_y])
    angle_mean = angle.mean(dim="time")
    ds[v_x] = ds[v_x].where(np.abs(angle_mean - angle_expected) < angle_bounds)
    ds[v_y] = ds[v_y].where(np.abs(angle_mean - angle_expected) < angle_bounds)
    if filter_per_timestep:
        ds[v_x] = ds[v_x].where(np.abs(angle - angle_mean) < angle_bounds)
        ds[v_y] = ds[v_y].where(np.abs(angle - angle_mean) < angle_bounds)
    # now filter values in time that deviate too much from the average angle.
    # if filter_angle_var:
    #     angle_std = angle.std(dim="time")
    #     angle_var = angle_std/angle_mean
    #     ds[v_x] = ds[v_x].where(angle - angle_mean < var_thres)
    #     ds[v_y] = ds[v_y].where(s_var < var_thres)

    return ds


def piv_filter_variance(
    ds, v_x="v_x", v_y="v_y", var_thres=1.0, filter_per_timestep=True
):
    s = (ds[v_x] ** 2 + ds[v_x] ** 2) ** 0.5
    s_std = s.std(dim="time")
    s_mean = s.mean(dim="time")
    s_var = s_std / s_mean

    ds[v_x] = ds[v_x].where(s_var < var_thres)
    ds[v_y] = ds[v_y].where(s_var < var_thres)
    if filter_per_timestep:
        ds[v_x] = ds[v_x].where((s - s_mean) / s_std < var_thres)
        ds[v_y] = ds[v_y].where((s - s_mean) / s_std < var_thres)

    return ds


def piv_filter_velocity(ds, v_x="v_x", v_y="v_y", s_min=0.1, s_max=5.0):
    s = (ds[v_x] ** 2 + ds[v_x] ** 2) ** 0.5
    ds[v_x] = ds[v_x].where(s > s_min)
    ds[v_y] = ds[v_y].where(s < s_max)
    return ds

def piv_filter_corr(ds, v_x="v_x", v_y="v_y", corr="corr", corr_min=0.4):
    ds[v_x] = ds[v_x].where(ds[corr] > corr_min)
    ds[v_y] = ds[v_y].where(ds[corr] > corr_min)
    return ds
