import cv2
import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import Affine
import xarray as xr

from orc import piv_process, io, helpers
from orc.const import GEOGRAPHICAL_ATTRS, PIV_ATTRS, PERSPECTIVE_ATTRS

def compute_piv(frames, **kwargs):
    # forward the computation to piv
    dask_piv = dask.delayed(piv_process.piv, nout=6)
    v_x, v_y, s2n, corr = [], [], [], []
    frames_a = frames[0:-1]
    frames_b = frames[1:]
    for frame_a, frame_b in zip(frames_a, frames_b):
        dt = frame_b.time - frame_a.time
        # determine time difference dt between frames
        cols, rows, _v_x, _v_y, _s2n, _corr = dask_piv(
            frame_a,
            frame_b,
            res_x=frames.resolution,
            res_y=frames.resolution,
            dt=float(dt.values),
            search_area_size=frames.window_size,
            **kwargs,
        )
        v_x.append(_v_x), v_y.append(_v_y), s2n.append(_s2n), corr.append(_corr)
    # compute one sample for the spacing
    cols, rows, _v_x, _v_y, _s2n, _corr = piv_process.piv(
        frame_a,
        frame_b,
        res_x=frames.resolution,
        res_y=frames.resolution,
        dt=float(dt.values),
        search_area_size=frames.window_size,
        **kwargs,
    )
    global_attrs = frames.attrs
    time = (frames.time[0:-1].values + frames.time[1:].values)/2  # as we use frame to frame differences, one time step gets lost
    x, y = io.get_axes(cols, rows, frames.resolution)
    xs, ys, lons, lats = io.get_xs_ys(
        cols,
        rows,
        helpers.deserialize_attr(frames, "proj_transform", Affine, args_parse=True),
        frames.crs
    )
    M = helpers.deserialize_attr(frames, "M_reverse", np.array)
    # compute row and column position of vectors in original reprojected background image col/row coordinates
    xp, yp = helpers.xy_to_perspective(x, np.flipud(y), frames.resolution, M)
    # dirty trick to ensure y coordinates start at the top in the right orientation
    shape_y, shape_x = helpers.deserialize_attr(frames, "camera_shape", np.array)
    yp = shape_y - yp
    coords = {
        "time": time,
        "y": y,
        "x": x
    }
    v_x, v_y, s2n, corr = [
        helpers.delayed_to_da(
            data,
            (len(y), len(x)),
            np.float32,
            coords=coords,
            attrs=attrs,
            name=name
        ) for data, (name, attrs) in zip((v_x, v_y, s2n, corr), PIV_ATTRS.items())]
    # prepare the xs, ys, lons and lats grids for geographical projections
    ds = xr.merge([v_x, v_y, s2n, corr])
    del coords["time"]
    # add all coordinate grids
    ds = helpers.add_xy_coords(ds, [xp, yp, xs, ys, lons, lats], coords, {**PERSPECTIVE_ATTRS, **GEOGRAPHICAL_ATTRS})
    # finally, add global attributes
    ds.attrs = global_attrs
    return ds

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
    Masks values using several filters that use temporal variations or comparison as basis
    :param ds: xarray Dataset, or file containing, with velocity vectors as [time, y, x]
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
    # load dataset in memory
    ds = ds.load()
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
    Masks values if neighbours over a certain rolling length before and after, have a
    significantly higher velocity than value under consideration, measured by tolerance
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param roll: amount of time steps in rolling window (centred)
    :param tolerance: Relative acceptable velocity of maximum found within rolling window
    :return: xarray Dataset, containing time-neighbour filtered velocity vectors as [time, y, x]
    """
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5
    s_roll = s.fillna(0.).rolling(time=roll, center=True).max()
    ds[v_x] = ds[v_x].where(s > tolerance*s_roll)
    ds[v_y] = ds[v_y].where(s > tolerance*s_roll)
    return ds


def filter_temporal_std(
    ds, v_x="v_x", v_y="v_y", tolerance=1.0):
    """
    Masks values if they deviate more than x standard deviations from the mean.
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param tolerance: float, representing amount of standard deviations
    :param filter_per_timestep:
    :return: xarray Dataset, containing standard deviation filtered velocity vectors as [time, y, x]
    """
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5
    s_std = s.std(dim="time")
    s_mean = s.mean(dim="time")
    s_var = s_std / s_mean
    ds[v_x] = ds[v_x].where((s - s_mean) / s_std < tolerance)
    ds[v_y] = ds[v_y].where((s - s_mean) / s_std < tolerance)

    return ds


def filter_temporal_velocity(ds, v_x="v_x", v_y="v_y", s_min=0.1, s_max=5.0):
    """
    Masks values if the velocity scalar lies outside a user-defined valid range
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param s_min: minimum scalar velocity [m s-1]
    :param s_max: maximum scalar velocity [m s-1]
    :return: xarray Dataset, containing velocity-range filtered velocity vectors as [time, y, x]
    """
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5
    ds[v_x] = ds[v_x].where(s > s_min)
    ds[v_y] = ds[v_y].where(s < s_max)
    return ds

def filter_temporal_corr(ds, v_x="v_x", v_y="v_y", corr="corr", tolerance=0.1):
    """
    Masks values with a too low correlation
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param corr: str, name of correlation variable
    :param tolerance: tolerance for correlation value. If correlation is lower than tolerance, it is masked
    :return: xarray Dataset, containing correlation filtered velocity vectors as [time, y, x]
    """
    ds[v_x] = ds[v_x].where(ds[corr] > tolerance)
    ds[v_y] = ds[v_y].where(ds[corr] > tolerance)
    return ds

def filter_spatial(
    ds,
    v_x="v_x",
    v_y="v_y",
    filter_nan=True,
    filter_median=True,
    kwargs_nan={},
    kwargs_median={},
):
    """
    Masks velocity values on a number of spatial filters
    :param ds: xarray Dataset, or file containing, with velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param kwargs_nan: dict, keyword arguments to pass to filter_spatial_nan
    :param kwargs_median: dict, keyword arguments to pass to filter_spatial_median
    :return: xarray Dataset, containing spatially filtered velocity vectors as [time, y, x]
    """
    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
    # work on v_x and v_y only
    ds_temp = ds[[v_x, v_y]]
    if filter_nan:
        ds_g = ds_temp.groupby("time")
        ds_temp = ds_g.apply(filter_spatial_nan, v_x=v_x, v_y=v_y, **kwargs_nan)
    if filter_median:
        ds_g = ds_temp.groupby("time")
        ds_temp = ds_g.apply(filter_spatial_median, v_x=v_x, v_y=v_y, **kwargs_median)
    # merge the temporary set with the original
    ds = xr.merge([ds.drop_vars([v_x, v_y]), ds_temp])
    return ds


def filter_spatial_nan(ds, v_x="v_x", v_y="v_y", tolerance=0.8, stride=1, missing=-9999.):
    """
    Masks values if their surrounding neighbours (inc. value itself) contain too many NaN. Meant to remove isolated
    velocity estimates
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param tolerance: float, amount of NaNs in search window measured as a fraction of total amount of values [0-1]
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: xarray Dataset, containing NaN filtered velocity vectors as [time, y, x]
    """
    # u, v = ds[v_x], ds[v_y]
    u, v = ds[v_x].values, ds[v_y].values
    u_move = helpers.neighbour_stack(u, stride=stride, missing=missing)
    # replace missings by Nan
    nan_frac = np.float64(np.isnan(u_move)).sum(axis=0)/float(len(u_move))
    u[nan_frac > tolerance] = np.nan
    v[nan_frac > tolerance] = np.nan
    ds[v_x][:] = u
    ds[v_y][:] = v
    return ds

def filter_spatial_median(ds, v_x="v_x", v_y="v_y", tolerance=0.7, stride=1, missing=-9999.):
    """
    Masks values when their value deviates more than x standard deviations from the median of its neighbours
        (inc. itself).
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param tolerance: amount of standard deviations tolerance
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: xarray Dataset, containing std filtered velocity vectors as [time, y, x]
    """
    u, v = ds[v_x].values, ds[v_y].values
    s = (u**2 + v**2)**0.5
    s_move = helpers.neighbour_stack(s, stride=stride)
    # replace missings by Nan
    s_median = np.nanmedian(s_move, axis=0)
    # now filter points that are very far off from the median
    filter = np.abs(s - s_median)/s_median > tolerance
    u[filter] = np.nan
    v[filter] = np.nan
    ds[v_x][:] = u
    ds[v_y][:] = v
    return ds

def plot(ds, scalar=True, quiver=True, background=None, mode="ortho", scalar_kwargs={}, quiver_kwargs={}, v_x="v_x", v_y="v_y"):
    if len(ds[v_x].shape) > 2:
        raise OverflowError(
            f'Dataset\'s variables should only contain 2 dimensions, this dataset '
            f'contains {len(ds[v_x].shape)} dimensions. Reduce this by applying a reducer or selecting a time step. '
            f'Reducing can be done e.g. with ds.mean(dim="time") or slicing with ds.isel(time=0)'
        )
    assert (scalar or quiver), "Either scalar or quiver should be set tot True, nothing to plot"
    assert mode in ["local", "geographical", "perspective"], 'Mode must be "ortho", "geographic" or "objective"'

    if mode == "local":
        x = "x"
        y = "y"
        theta = 0.
    elif mode == "geographical":
        x = "lon"
        y = "lat"
        aff = helpers.deserialize_attr(ds, "proj_transform", Affine, args_parse=True)
        theta = np.arctan2(aff.d, aff.a)
    else:
        x = "xp"
        y = "yp"
        theta = 0.
    if mode in ["local", "geographical"]:
        f = plt.figure()
    else:
        # in camera mode make a screen filling figure with black edges and faces
        f = plt.figure(figsize=(16, 9), frameon=False, facecolor="k")
        f.set_size_inches(16, 9, True)
        f.patch.set_facecolor("k")
        f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = f.add_subplot(111)
    # ax.axis("equal")


    u = ds[v_x].copy()
    v = ds[v_y].copy()
    s = (u**2 + v**2)**0.5

    if background is not None:
        if (len(background.shape) == 3 and background.shape[-1] == 3):
            facecolors = background.values.reshape(background.shape[0]*background.shape[1], 3)/255
            facecolors = np.hstack([facecolors, np.ones((len(facecolors), 1))])
            quad = ax.pcolormesh(background[x], background[y], background.mean(dim="rgb"), shading="nearest", facecolors=facecolors)
            quad.set_array(None)
        else:
            background.plot(x=x, y=y, add_colorbar=False)
        # ax.set_autoscale_on(False)
    if scalar:
        # plot the scalar velocity value as grid, return mappable
        # p = ax.pcolormesh(s[x], s[y], s, **scalar_kwargs)
        p = s.plot(ax=ax, x=x, y=y, add_colorbar=False, **scalar_kwargs)
    if quiver:
        if scalar:
            ax.quiver(
                ds[x],
                ds[y],
                *helpers.rotate_u_v(u, v, theta),
                **quiver_kwargs
            ) # , color="w", alpha=0.3, scale=75, width=0.0010)
        else:
            # if not scalar, then return a mappable here
            p = ax.quiver(
                ds[x],
                ds[y],
                *helpers.rotate_u_v(u, v, theta),
                s,
                **quiver_kwargs
            ) # , color="w", alpha=0.3, scale=75, width=0.0010)
    cax = f.add_axes([0.85, 0.05, 0.05, 0.5])
    cax.set_visible(False)
    cbar = f.colorbar(p, ax=cax)
    cbar.set_label(label="velocity [m/s]", size=15, weight='bold', color="w")
    cbar.ax.tick_params(labelsize=12, labelcolor="w")
    # finally, if a background is used, set xlim and ylim to the relevant axes
    if background is not None:
        ax.set_xlim([background[x].min(), background[x].max()])
        ax.set_ylim([background[y].min(), background[y].max()])
    return f, ax

def replace_outliers(ds, v_x="v_x", v_y="v_y", stride=1, max_iter=1):
    """
    Replace missing values using neighbourhood operators. Use this with caution as it creates data. If many samples
    in time are available to derive a mean or median velocity from, consider using a reducer on those samples instead
    of a spatial infilling method such as suggested here.
    :param ds: xarray Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param stride: int, stride used to determine relevant neighbours
    :param max_iter: number of iterations for replacement
    :return:
    """
    # TO-DO: make replacement decision dependent on amount of non-NaN values in neighbourhood
    u, v = ds[v_x].values, ds[v_y].values
    for n in range(max_iter):
        u_move = helpers.neighbour_stack(u, stride=stride)
        v_move = helpers.neighbour_stack(v, stride=stride)
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
