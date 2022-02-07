import dask
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import numpy as np
from rasterio.transform import Affine
import xarray as xr
import warnings

from pyorc import piv_process, io, helpers
from pyorc.const import GEOGRAPHICAL_ATTRS, PIV_ATTRS, PERSPECTIVE_ATTRS

def compute_piv(frames, **kwargs):
    """
    Perform PIV computation on projected frames. Only a pipeline graph to computation is setup. Call a result to
    trigger actual computation.

    :param frames: xr.DataArray, containing projected frames as 3D array or dask array
    :param kwargs: dict, keyword arguments to pass to dask_piv, used to control the manner in which openpiv.pyprocess
        is called.
    :return: xr.Dataset, containing the PIV results in a lazy dask.array form.
    """
    # forward the computation to piv
    dask_piv = dask.delayed(piv_process.piv, nout=6)
    v_x, v_y, s2n, corr = [], [], [], []
    frames_a = frames[0:-1]
    frames_b = frames[1:]
    for frame_a, frame_b in zip(frames_a, frames_b):
        # select the time difference in seconds
        dt = frame_b.time - frame_a.time
        # perform lazy piv graph computation
        cols, rows, _v_x, _v_y, _s2n, _corr = dask_piv(
            frame_a,
            frame_b,
            res_x=frames.resolution,
            res_y=frames.resolution,
            dt=float(dt.values),
            search_area_size=frames.window_size,
            **kwargs,
        )
        # append to result
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
    # extract global attributes from origin
    global_attrs = frames.attrs
    time = (frames.time[0:-1].values + frames.time[1:].values)/2  # as we use frame to frame differences, one time step gets lost
    # retrieve the x and y-axis belonging to the results
    x, y = io.get_axes(cols, rows, frames.resolution)
    # convert in projected and latlon coordinates
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
    # here establish the full xr.Dataset
    v_x, v_y, s2n, corr = [
        helpers.delayed_to_da(
            data,
            (len(y), len(x)),
            np.float32,
            coords=coords,
            attrs=attrs,
            name=name
        ) for data, (name, attrs) in zip((v_x, v_y, s2n, corr), PIV_ATTRS.items())]
    ds = xr.merge([v_x, v_y, s2n, corr])
    del coords["time"]
    # prepare the xs, ys, lons and lats grids for geographical projections and add to xr.Dataset
    ds = helpers.add_xy_coords(ds, [xp, yp, xs, ys, lons, lats], coords, {**PERSPECTIVE_ATTRS, **GEOGRAPHICAL_ATTRS})
    # finally, add global attributes and return xr.Dataset
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
    kwargs_corr={},
    kwargs_std={},
    kwargs_angle={},
    kwargs_velocity={},
    kwargs_neighbour={},
):
    """
    Masks values using several filters that use temporal variations or comparison as basis.

    :param ds: xr.Dataset, or file containing, with velocity vectors as [time, y, x]
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
    :return: xr.Dataset, containing temporally filtered velocity vectors as [time, y, x]
    """
    if not isinstance(ds, xr.Dataset):
        # assume ds is as yet a ref to a filename or buffer and first open
        ds = xr.open_dataset(ds)
    # load dataset in memory
    ds = ds.load()
    # start with entirely independent filters
    if filter_corr:
        ds = filter_temporal_corr(ds, v_x=v_x, v_y=v_y, **kwargs_corr)
    if filter_velocity:
        ds = filter_temporal_velocity(ds, v_x=v_x, v_y=v_y, **kwargs_velocity)
    if filter_neighbour:
        ds = filter_temporal_neighbour(ds, v_x=v_x, v_y=v_y, **kwargs_neighbour)
    # finalize with temporally dependent filters
    if filter_std:
        ds = filter_temporal_std(ds, v_x=v_x, v_y=v_y, **kwargs_std)
    if filter_angle:
        ds = filter_temporal_angle(ds, v_x=v_x, v_y=v_y, **kwargs_angle)
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
    range of zero, as zero is the same as 2*pi. This exception may be resolved in the future if necessary.

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param angle_expected float, angle (0-2*pi), measured clock-wise from vertical upwards direction, expected
        in the velocites, default: 0.5*np.pi (meaning from left to right)
    :param angle_tolerance: float (0-2*pi) maximum deviation from expected angle allowed.
    :param filter_per_timestep: if set to True, tolerances are also checked per individual time step
    :return: xr.Dataset, containing angle filtered velocity vectors as [time, y, x], default: True
    """
    # TODO: make function working appropriately, if angles are close to zero (2*pi)
    # first filter on the temporal mean. This is to ensure that widely varying results in angle are deemed not
    # to be trusted.
    v_x_mean = ds[v_x].mean(dim="time")
    v_y_mean = ds[v_y].mean(dim="time")
    angle_mean = np.arctan2(v_x_mean, v_y_mean)
    # angle_mean = angle.mean(dim="time")
    ds[v_x] = ds[v_x].where(np.abs(angle_mean - angle_expected) < angle_tolerance)
    ds[v_y] = ds[v_y].where(np.abs(angle_mean - angle_expected) < angle_tolerance)
    # refine locally if user wishes so
    if filter_per_timestep:
        angle = np.arctan2(ds[v_x], ds[v_y])
        ds[v_x] = ds[v_x].where(np.abs(angle - angle_expected) < angle_tolerance)
        ds[v_y] = ds[v_y].where(np.abs(angle - angle_expected) < angle_tolerance)
    return ds

def filter_temporal_neighbour(ds, v_x="v_x", v_y="v_y", roll=5, tolerance=0.5):
    """
    Masks values if neighbours over a certain rolling length before and after, have a
    significantly higher velocity than value under consideration, measured by tolerance.

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param roll: int, amount of time steps in rolling window (centred)
    :param tolerance: float (0-1), Relative acceptable velocity of maximum found within rolling window
    :return: xr.Dataset, containing time-neighbour filtered velocity vectors as [time, y, x]
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

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param tolerance: float, representing amount of standard deviations
    :return: xr.Dataset, containing standard deviation filtered velocity vectors as [time, y, x]
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
    Masks values if the velocity scalar lies outside a user-defined valid range.

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param s_min: float, minimum scalar velocity [m s-1]
    :param s_max: float, maximum scalar velocity [m s-1]
    :return: xr.Dataset, containing velocity-range filtered velocity vectors as [time, y, x]
    """
    s = (ds[v_x] ** 2 + ds[v_y] ** 2) ** 0.5
    ds[v_x] = ds[v_x].where(s > s_min)
    ds[v_y] = ds[v_y].where(s < s_max)
    return ds

def filter_temporal_corr(ds, v_x="v_x", v_y="v_y", corr="corr", tolerance=0.4):
    """
    Masks values with a too low correlation.

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param corr: str, name of correlation variable
    :param tolerance: float (0-1), tolerance for correlation value. If correlation is lower than tolerance, it is masked
    :return: xr.Dataset, containing correlation filtered velocity vectors as [time, y, x]
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
    Masks velocity values on a number of spatial filters.

    :param ds: xr.Dataset, or file containing, with velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param kwargs_nan: dict, keyword arguments to pass to filter_spatial_nan
    :param kwargs_median: dict, keyword arguments to pass to filter_spatial_median
    :return: xr.Dataset, containing spatially filtered velocity vectors as [time, y, x]
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
    velocity estimates.

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param tolerance: float, amount of NaNs in search window measured as a fraction of total amount of values [0-1]
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: xr.Dataset, containing NaN filtered velocity vectors as [time, y, x]
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

    :param ds: xr.Dataset, containing velocity vectors as [time, y, x]
    :param v_x: str, name of x-directional velocity
    :param v_y: str, name of y-directional velocity
    :param tolerance: float, amount of standard deviations tolerance
    :param stride: int, stride used to determine relevant neighbours
    :param missing: float, a temporary missing value, used to be able to convolve NaNs
    :return: xr.Dataset, containing std filtered velocity vectors as [time, y, x]
    """
    u, v = ds[v_x].values, ds[v_y].values
    s = (u**2 + v**2)**0.5
    s_move = helpers.neighbour_stack(s, stride=stride)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # replace missings by Nan
        s_median = np.nanmedian(s_move, axis=0)
    # now filter points that are very far off from the median
    filter = np.abs(s - s_median)/s_median > tolerance
    u[filter] = np.nan
    v[filter] = np.nan
    ds[v_x][:] = u
    ds[v_y][:] = v
    return ds

def get_uv_camera(ds, dt=0.1, v_x="v_x", v_y="v_y"):
    """
    Returns row, column locations in the camera objective, and u (x-directional) and v (y-directional) vectors, scaled
    and transformed to the camera objective (i.e. vectors far away are smaller than closeby, and follow the river direction)
    applied on u and v, so that they plot in a geographical space. This is needed because the raster of PIV results
    is usually rotated geographically, so that water always flows from left to right in the grid. The results can be
    used to plot velocities in the camera perspective, e.g. overlayed on a background image directly from the camera.

    :param ds: xr.Dataset, established using compute_piv. This dataset's attributes are used to construct the rotation
    :param dt: float, time difference [s] used to scale the u and v velocities to a very small distance to project with
        default: 0.1, usually not needed to modify this.
    :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
    :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
    :return: 5 outputs: 4 np.ndarrays containing camera perspective column location, row location, transformed u and v
        velocity vectors (no unit) and the scalar velocities (m/s). Rotation is not needed because the transformed
        u and v components are already rotated to match the camera perspective. counter-clockwise rotation in radians.
    """
    # retrieve the backward transformation array
    M = helpers.deserialize_attr(ds, "M_reverse", np.array)
    # get the shape of the original frames
    shape_y, shape_x = helpers.deserialize_attr(ds, "camera_shape", np.array)
    xi, yi = np.meshgrid(ds.x, ds.y)
    # flip the y-coordinates to match the row order used by opencv
    yi = np.flipud(yi)

    x_moved, y_moved = xi + ds[v_x] * dt, yi + ds[v_y] * dt
    xp_moved, yp_moved = helpers.xy_to_perspective(x_moved.values, y_moved.values, ds.resolution, M)

    # convert row counts to start at the top of the frame instead of bottom
    yp_moved = shape_y - yp_moved

    # missing values end up at the top-left, replace these with nan
    yp_moved[yp_moved == shape_y] = np.nan # ds["yp"].values[yp_moved == shape_y]
    xp_moved[xp_moved == 0] = np.nan  # ds["xp"].values[xp_moved == 0]

    u, v = xp_moved - ds["xp"], yp_moved - ds["yp"]
    s = (ds[v_x]**2 + ds[v_y]**2)**0.5
    return "xp", "yp", u, v, s


def get_uv_geographical(ds, v_x="v_x", v_y="v_y"):
    """
    Returns lon, lat coordinates and u (x-directional) and v (y-directional) velocities, and a rotation that must be
    applied on u and v, so that they plot in a geographical space. This is needed because the raster of PIV results
    is usually rotated geographically, so that water always flows from left to right in the grid. The results can be
    used to plot velocities in a geographical map, e.g. overlayed on a background image, projected to lat/lon.

    :param ds: xr.Dataset, established using compute_piv. This dataset's attributes are used to construct the rotation
    :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
    :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
    :return: 6 outputs: 5 np.ndarrays containing longitude, latitude coordinates, u and v velocities and scalar velocity;
        and float with counter-clockwise rotation in radians, to be applied on u, v to plot in geographical space.

    """
    # select lon and lat variables as coordinates
    u = ds[v_x]
    v = ds[v_y]
    s = (u**2 + v**2)**0.5
    aff = helpers.deserialize_attr(ds, "proj_transform", Affine, args_parse=True)
    theta = np.arctan2(aff.d, aff.a)
    return "lon", "lat", u, v, s, theta


def plot(
        ds,
        ax=None,
        scalar=True,
        quiver=True,
        background=None,
        mode="local",
        background_kwargs={},
        scalar_kwargs={},
        quiver_kwargs={},
        v_x="v_x",
        v_y="v_y",
        cbar_color="w",
        cbar_fontsize=15
):
    """
    Extensive functionality to plot PIV results. PIV results can be plotted on a background frame, and can be plotted as
    scalar values (i.e. a mesh) or as quivers, or both by setting the inputs 'scalar' and 'quiver' to True or False.
    Plotting can be done in three modes:
    - "local": a simple planar view plot, with a local coordinate system in meters, with the top-left coordinate
      being the 0, 0 point, and ascending coordinates towards the right and bottom. If a background frame is provided,
      then this must be a projected background frame (i.e. resulting from `pyorc.frames.project`)
    - "geographical": a geographical plot, requiring the package `cartopy`, the results are plotted on a geographical
      axes, so that combinations with tile layers such as OpenStreetMap, or shapefiles can be made. If a background
      frame is provided, then this must be a projected background frame, i.e. resulting from `pyorc.frames.project`.
    - "camera": plots velocities as augmented reality (i.e. seen from the camera perspective). This is the most
      intuitive view for end users. If a background frame is provided, then this must be a frame from the camera
      perspective, i.e. as derived from a `pyorc.Video` object, with the method `pyorc.Video.get_frames`.

    :param ds: xr.Dataset, established using compute_piv. This dataset's attributes are used to construct the rotation
    :param ax: pre-defined axes object. If not set, a new axes will be prepared. In case `mode=="geographical"`, a
        cartopy GeoAxes needs to be provided, or will be made in case ax is not set.
    :param scalar: boolean, if set to True, velocities are plotted as scalar values in a mesh (default: True)
    :param quiver: boolean, if set to True, velocities are plotted as quiver (i.e. arrows). In case scalar is also True,
        quivers will be plotted with a single color (defined in `quiver_kwargs`), if not, the scalar values are used
        to color the arrows.
    :param background: xr.DataArray, a single frame capture to be used as background, taken from pyorc.Video.get_frames in case
        `mode=="camera"` and from `pyorc.frames.project` in case `mode=="local"` or `mode=="geographical"`.
    :param mode: can be "local", "geographical", or "camera". To select the perspective of plotting, see description.
    :param background_kwargs: dict, plotting parameters to be passed to matplotlib.pyplot.pcolormesh, for plotting the
        background frame.
    :param scalar_kwargs: dict, plotting parameters to be passed to matplotlib.pyplot.pcolormesh, for plotting scalar
        values.
    :param quiver_kwargs: dict, plotting parameters to be passed to matplotlib.pyplot.quiver, for plotting quiver arrows.
    :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
    :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
    :param cbar_color: color to use for the colorbar
    :param cbar_fontsize: fontsize to use for the colorbar title (fontsize of tick labels will be made slightly smaller).
    :return: ax, axes object resulting from this function.
    """

    if len(ds[v_x].shape) > 2:
        raise OverflowError(
            f'Dataset\'s variables should only contain 2 dimensions, this dataset '
            f'contains {len(ds[v_x].shape)} dimensions. Reduce this by applying a reducer or selecting a time step. '
            f'Reducing can be done e.g. with ds.mean(dim="time") or slicing with ds.isel(time=0)'
        )
    assert (scalar or quiver), "Either scalar or quiver should be set tot True, nothing to plot"
    assert mode in ["local", "geographical", "camera"], 'Mode must be "local", "geographical" or "camera"'
    if mode == "local":
        x = "x"
        y = "y"
        theta = 0.
        u = ds[v_x]
        v = ds[v_y]
        s = (u ** 2 + v ** 2) ** 0.5
    elif mode == "geographical":
        # import some additional packages
        import cartopy.crs as ccrs
        # add transform for GeoAxes
        scalar_kwargs["transform"] = ccrs.PlateCarree()
        quiver_kwargs["transform"] = ccrs.PlateCarree()
        background_kwargs["transform"] = ccrs.PlateCarree()
        x, y, u, v, s, theta = get_uv_geographical(ds)
    else:
        # mode is camera
        x, y, u, v, s = get_uv_camera(ds)
        theta = 0.
    # prepare an axis for the provided mode
    ax = plot_prepare_axes(ax=ax, mode=mode)
    f = ax.figure  # handle to figure

    if background is not None:
        if (len(background.shape) == 3 and background.shape[-1] == 3):
            facecolors = background.values.reshape(background.shape[0]*background.shape[1], 3)/255
            facecolors = np.hstack([facecolors, np.ones((len(facecolors), 1))])
            quad = ax.pcolormesh(background[x], background[y], background.mean(dim="rgb"), shading="nearest", facecolors=facecolors, **background_kwargs)
            # remove array values, override .set_array, needed in case GeoAxes is provided, because GeoAxes asserts if array has dims
            QuadMesh.set_array(quad, None)
        else:
            ax.pcolormesh(background[x], background[y], background, **background_kwargs)
    if quiver:
        if scalar:
            p = plot_quiver(ax, ds[x].values, ds[y].values, *[v.values for v in helpers.rotate_u_v(u, v, theta)], None,
                            **quiver_kwargs)
        else:
            p = plot_quiver(ax, ds[x].values, ds[y].values, *[v.values for v in helpers.rotate_u_v(u, v, theta)], s,
                            **quiver_kwargs)
    if scalar:
        # plot the scalar velocity value as grid, return mappable
        p = ax.pcolormesh(s[x], s[y], s, zorder=2, **scalar_kwargs)
        if mode == "geographical":
            ax.set_extent([ds[x].min() - 0.00005, ds[x].max() + 0.00005, ds[y].min() - 0.00005, ds[y].max() + 0.00005],
                  crs=ccrs.PlateCarree())
    cbar = plot_cbar(ax, p, mode=mode, size=cbar_fontsize, color=cbar_color)
    # finally, if a background is used, set xlim and ylim to the relevant axes
    if (background is not None and mode != "geographical"):
        ax.set_xlim([background[x].min(), background[x].max()])
        ax.set_ylim([background[y].min(), background[y].max()])
    return ax

def plot_prepare_axes(ax=None, mode="local"):
    """
    Prepares the axes, needed to plot results, called from `pyorc.piv.plot`.

    :param mode: str, mode to plot, can be "local", "geographical" or "camera", default: "local"
    :return: ax, axes object.
    """
    if ax is not None:
        if mode=="geographical":
            # ensure that the axes is a geoaxes
            from cartopy.mpl.geoaxes import GeoAxesSubplot
            assert (
                isinstance(ax, GeoAxesSubplot)), "For mode=geographical, the provided axes must be a cartopy GeoAxesSubplot"
        return ax

    # make a screen filling figure with black edges and faces
    f = plt.figure(figsize=(16, 9), frameon=False, facecolor="k")
    f.set_size_inches(16, 9, True)
    f.patch.set_facecolor("k")
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    if mode == "geographical":
        import cartopy.crs as ccrs
        ax = f.add_subplot(111, projection=ccrs.PlateCarree())
    else:
        ax = plt.subplot(111)
    return ax

def plot_quiver(ax, x, y, u, v, s=None, zorder=3, **kwargs):
    """
    Add quiver plot to existing axes.

    :param ax: axes object
    :param x: np.ndarray (2D), x-coordinate grid
    :param y: np.ndarray (2D), y-coordinate grid
    :param u: np.ndarray (2D), x-directional (u) velocity components [m/s]
    :param v: np.ndarray (2D), y-directional (v) velocity components [m/s]
    :param s: np.ndarray (2D), scalar velocities [m/s]
    :param zorder: int, zorder in plot (default: 3)
    :param kwargs: dict, keyword arguments to pass to matplotlib.pyplot.quiver
    :return: mappable, result from matplotlib.pyplot.quiver (can be used to construct a colorbar or legend)
    """
    if s is None:
        p = ax.quiver(x, y, u, v, zorder=zorder, **kwargs)
    else:
        # if not scalar, then return a mappable here
        p = ax.quiver(x, y, u, v, s, zorder=zorder, **kwargs)
    return p

def plot_cbar(ax, p, mode="local", size=15, color="w"):
    """
    Add colorbar to existing axes. In case camera mode is used, the colorbar will get a bespoke layout and will
    be placed inside of the axes object.

    :param ax: axes object
    :param p: mappable, used to define colorbar
    :param mode: plotting mode, see pyorc.piv.plot
    :param size: fontsize, used for colorbar title, only used with `mode="camera"`
    :param color: color, used for fonts of colorbar title and ticks, only used with `mode="camera"`
    :return: handle to colorbar
    """
    if mode == "camera":
        # place the colorbar nicely inside
        cax = ax.figure.add_axes([0.9, 0.05, 0.05, 0.5])
        cax.set_visible(False)
        cbar = ax.figure.colorbar(p, ax=cax)
        cbar.set_label(label="velocity [m/s]", size=size, weight='bold', color=color)
        cbar.ax.tick_params(labelsize=size-3, labelcolor=color)
    else:
        cbar = ax.figure.colorbar(p)
    return cbar


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
    :return: xr.Dataset, containing filtered velocities
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
