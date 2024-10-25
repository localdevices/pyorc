"""Plot methods for pyorc."""

import copy
import functools

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import patheffects
from matplotlib.collections import QuadMesh

from pyorc import cv, helpers

path_effects = [
    patheffects.Stroke(linewidth=3, foreground="w"),
    patheffects.Normal(),
]

LINE_COLOR = "#385895"


def _base_plot(plot_func):
    commondoc = """

    Parameters
    ----------
    mode : str, optional
        perspective mode to use for plotting. Can be "local" (default), "geographical", or "camera". For
        "geographical" a velocimetry result that contains "lon" and "lat" coordinates must be provided
        (i.e. produced with known CRS for control points).
    x : str, optional
        Coordinate for *x* axis. If ``None``, use ``darray.dims[1]``.
    y : str, optional
        Coordinate for *y* axis. If ``None``, use ``darray.dims[0]``.
    ax : plt.axes, optional
        If None (default), use the current axes. Not applicable when using facets.
    **kwargs : additional keyword arguments to wrapped Matplotlib function.

    Returns
    -------
    artist : matplotlib mappable
        The same type of primitive artist that the wrapped Matplotlib
        function returns.

    """
    # This function is largely based on xarray.Dataset function _dsplot
    # Build on the original docstring
    plot_func.__doc__ = f"{plot_func.__doc__}{commondoc}"

    # apply wrapper to allow for partial update of the function, with updated docstring
    @functools.wraps(plot_func)
    def get_plot_method(
        ref,
        mode="local",
        ax=None,
        add_colorbar=False,
        add_cross_section=True,
        add_text=False,
        text_prefix="",
        text_suffix="",
        kwargs_line=None,
        **kwargs,
    ):
        """Retrieve plot method with all required inputs.

        Parameters
        ----------
        ref : xr.Dataset
            velocimetry or transect object
        mode : str, optional
            perspective mode to use for plotting. Can be "local", "geographical", or "camera". For
            "geographical" a velocimetry result that contains "lon" and "lat" coordinates must be provided
            (i.e. produced with known CRS for control points).
        ax : plt.axes object, optional
            If None (default), use the current axes. Not applicable when using facets.
        add_colorbar : boolean, optional
            if True, a colorbar is added to axes (default: False)
        add_cross_section : boolean, optional
            if True, and a transect is plotted, the transect coordinates are plotted (default: True)
        add_text : boolean, optional
            if True, add a text label in the axes displaying information about the video's transect
        text_prefix : str, optional
            string to add in front of standard text on transect plot. Only used if ``add_text=True``
        text_suffix : str, optional
            String to add after standard text on transect plot. Only used if ``add_text=True``
        kwargs_line : dict, optional
            additional keyword arguments passed to matplotlib.pyplot.plot for plotting cross-section.
            (Default value = {})
        **kwargs : dict
            additional keyword arguments to wrapped Matplotlib function.

        Returns
        -------
        p : matplotlib mappable
            mappable of wrapped matplotlib function

        """
        if not kwargs_line:
            kwargs_line = {}
        # in case persistent transform is in kwargs_line, remove this
        if mode == "geographical":
            try:
                pass
            except ImportError:
                raise ImportError("Cartopy not found, please install with 'mamba install -c conda-forge cartopy")
        if "transform" in kwargs_line:
            del kwargs_line["transform"]
        ax = _prepare_axes(ax=ax, mode=mode)
        # update ax

        if len(ref._obj["v_x"].shape) > 2:
            raise OverflowError(
                f'Dataset\'s variables should only contain 2 dimensions, this dataset '
                f'contains {len(ref._obj["v_x"].shape)} dimensions. Apply a reducer or select a time step. '
                f'Reducing can be done e.g. with ds.mean(dim="time", keep_attrs=True) or slicing with ds.isel(time=0)'
            )
        if "time" in ref._obj:
            if isinstance(ref._obj["time"].values.tolist(), list):
                raise OverflowError(
                    'Dimension "time" exists in dataset and has multiple values. Reduce this by applying a reducer or '
                    "selecting a time step."
                )
        if "quantile" in ref._obj:
            if isinstance(ref._obj["quantile"].values.tolist(), list):
                raise OverflowError(
                    'Dimension "quantile" exists in dataset and contains multiple values. Reduce this by selecting one'
                    "quantile, e.g. using ds.isel(quantile=2) or ds.sel(quantile=0.5)"
                )
        if plot_func.__name__ == "streamplot":
            if mode != "local":
                raise NotImplementedError(f"Streamplot only works in local mode, not in {mode} mode.")

        # check if dataset is a transect or not
        is_transect = True if "points" in ref._obj.dims else False

        assert mode in ["local", "geographical", "camera"], 'Mode must be "local", "geographical" or "camera"'
        if mode == "local":
            x = ref._obj["x"].values
            y = ref._obj["y"].values
            u, v, s = ref.get_uv_local()
            if not (is_transect) and plot_func.__name__ == "scatter":
                x, y = np.meshgrid(x, y)

            if plot_func.__name__ == "streamplot":
                # flipping of variables is needed
                y = np.flipud(y)
                u = np.flipud(u)
                v = np.flipud(v)
                s = np.flipud(s)

        elif mode == "geographical":
            # import some additional packages
            import cartopy.crs as ccrs

            # add transform for GeoAxes
            kwargs["transform"] = ccrs.PlateCarree()
            kwargs_line["transform"] = ccrs.PlateCarree()

            x = ref._obj["lon"].values
            y = ref._obj["lat"].values
            u, v, s = ref.get_uv_geographical()
        else:
            # mode is camera
            x = ref._obj["xp"].values
            y = ref._obj["yp"].values
            u, v, s = ref.get_uv_camera()
        if plot_func.__name__ in ["quiver", "streamplot"]:
            primitive = plot_func("", x, y, u, v, s, ax, **kwargs)
        else:
            primitive = plot_func("", x, y, s, ax, **kwargs)
        if add_colorbar:
            cbar(ax, primitive)
        if mode == "local":
            ax.set_aspect("equal")
        if is_transect:
            if add_cross_section:
                if mode == "camera":
                    # # lens position is needed, so check this
                    # if hasattr(ref._obj.camera_config, "lens_position"):
                    # x_bottom, y_bottom = ref._obj.transect.get_xyz_perspective()
                    points = ref._obj.transect.get_wetted_perspective(h=ref._obj.transect.h_a)
                    ax.add_patch(
                        plt.Polygon(
                            points,
                            alpha=0.3,
                            linewidth=2.0,
                            facecolor=LINE_COLOR,  # "#00FF88",
                            path_effects=path_effects,
                            edgecolor="w",
                        )
                    )

                    # get cross section points in camera perspective
                    points_cross = ref._obj.transect.get_transect_perspective()

                    depth_lines = ref._obj.transect.get_depth_perspective(h=ref._obj.transect.h_a)
                    for l in depth_lines:
                        line = np.array(l)
                        ax.plot(
                            line[:, 0],
                            line[:, 1],
                            color="w",
                            alpha=0.5,
                            linewidth=2.0,
                            # path_effects=path_effects
                        )
                    ax.plot(
                        points_cross[:, 0],
                        points_cross[:, 1],
                        color=LINE_COLOR,
                        linewidth=4.0,
                        # path_effects=path_effects,
                        alpha=0.5,
                        **kwargs_line,
                    )
                else:
                    ax.plot(x, y, LINE_COLOR, path_effects=path_effects, alpha=0.7, **kwargs_line)
                if add_text:
                    plot_text(ax, ref._obj, text_prefix, text_suffix)

        if mode == "geographical" and not (is_transect):
            ax.set_extent(
                [x.min() - 0.0001, x.max() + 0.0001, y.min() - 0.0001, y.max() + 0.0001], crs=ccrs.PlateCarree()
            )
        return primitive

    return get_plot_method


def _frames_plot(ref, ax=None, mode="local", **kwargs):
    """Create QuadMesh plot from a RGB or grayscale frame on a new or existing (if ax is not None) axes.

    Wraps :py:func:`matplotlib:matplotlib.collections.QuadMesh`.

    Parameters
    ----------
    ref : plot reference
        This parameter should not be specified, ignore.
    ax : plt.axes, optional
        If None (default), use the current axes. Not applicable when using facets.
    mode : str, optional
        perspective mode to use for plotting. Can be "local" (default), "geographical", or "camera".
        For "geographical" a frames set from Frames.project should be used that contains "lon" and "lat" coordinates.
        For "camera", a non-projected frames set should be used.
    **kwargs : dict
        additional keyword arguments to wrapped Matplotlib function.

    Returns
    -------
    p : matplotlib.collections.QuadMesh

    """  # noqa: D417
    # prepare axes
    if mode == "geographical":
        try:
            pass
        except ImportError:
            raise ImportError("Cartopy not found, please install with 'mamba install -c conda-forge cartopy")
    if "time" in ref._obj.coords:
        if ref._obj.time.size > 1:
            raise AttributeError(
                f'Object contains dimension "time" with length {len(ref._obj.time)}. Reduce dataset by selecting'
                "one time step or taking a median, mean or other statistic."
            )
    if mode == "camera":
        rotation = ref.camera_config.rotation
    else:
        rotation = None
    ax = _prepare_axes(ax=ax, mode=mode, rotation=rotation)
    if mode == "local":
        x = "x"
        y = "y"
    elif mode == "geographical":
        # import some additional packages
        import cartopy.crs as ccrs

        # add transform for GeoAxes
        kwargs["transform"] = ccrs.PlateCarree()
        x = "lon"
        y = "lat"
    else:
        # mode is camera
        x = "xp"
        y = "yp"
    assert all(v in ref._obj.coords for v in [x, y]), f'required coordinates "{x}" and/or "{y}" are not available'
    if len(ref._obj.shape) == 3 and ref._obj.shape[-1] == 3:
        # looking at an rgb image
        facecolors = ref._obj.values.reshape(ref._obj.shape[0] * ref._obj.shape[1], 3) / 255
        facecolors = np.hstack([facecolors, np.ones((len(facecolors), 1))])
        primitive = ax.pcolormesh(
            ref._obj[x],
            ref._obj[y],
            ref._obj.mean(dim="rgb"),
            shading="nearest",
            facecolors=facecolors,
            **kwargs,
        )
        # remove array values, override .set_array, needed in case GeoAxes is provided, because GeoAxes asserts if
        # array has dims
        QuadMesh.set_array(primitive, None)
    else:
        primitive = ax.pcolormesh(ref._obj[x], ref._obj[y], ref._obj, **kwargs)
    # fix axis limits to min and max of extent of frames
    if mode == "geographical":
        ax.set_extent(
            [
                ref._obj[x].min() - 0.0001,
                ref._obj[x].max() + 0.0001,
                ref._obj[y].min() - 0.0001,
                ref._obj[y].max() + 0.0001,
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_xlim([ref._obj[x].min(), ref._obj[x].max()])
        ax.set_ylim([ref._obj[y].min(), ref._obj[y].max()])
    return primitive


class _Transect_PlotMethods:
    """Enable use of `ds.velocimetry.plot` functions as attributes on a Dataset containing velocimetry results.

    For example, `ds.velocimetry.plot.pcolormesh`. When called without a subfunction, quiver will be used.
    """

    def __init__(self, transect):
        # make the original dataset also available on the plotting object
        self.transect = transect
        self._obj = transect._obj
        # Add to class _PlotMethods

    def __call__(self, method="quiver", **kwargs):
        """call.

        Parameters
        ----------
        method : str, optional
            plot method to use (default: "quiver")
        **kwargs : dict
            additional keyword arguments to wrapped Matplotlib function.

        Returns
        -------
        p : matplotlib mappable
            mappable of wrapped matplotlib function

        """
        return getattr(self, method)(**kwargs)

    def get_uv_camera(self, dt=0.1):
        """Get x-directional (u), y-directional (v) and scalar velocity in camera projection from transect dataset.

        Parameters
        ----------
        dt : float, optional
            time step [s] used to transpose velocities over a short distance for projection (default: 0.1)

        Returns
        -------
        u : np.ndarray
            x-directional velocity
        v : np.ndarray
            y-directional velocity
        s : np.ndarray
            scalar velocity

        """
        v_eff = "v_eff"
        v_dir = "v_dir"
        # retrieve the backward transformation array
        transect = self._obj.transect
        camera_config = transect.camera_config
        # M = velocimetry.camera_config.get_M(velocimetry.h_a, to_bbox_grid=True, reverse=True)
        src = camera_config.get_bbox(camera=True, h_a=transect.h_a, expand_exterior=False).exterior.coords[0:4]
        dst_xy = camera_config.get_bbox(expand_exterior=False).exterior.coords[0:4]
        # get geographic coordinates bbox corners
        # get geographic coordinates bbox corners
        dst = cv.transform_to_bbox(dst_xy, camera_config.bbox, camera_config.resolution)
        trans_mat = cv.get_M_2D(src, dst, reverse=True)

        x, y = self._obj.x, self._obj.y
        _u = self._obj[v_eff] * np.sin(self._obj[v_dir])
        _v = self._obj[v_eff] * np.cos(self._obj[v_dir])
        s = np.abs(self._obj[v_eff].values)
        x_moved, y_moved = x + _u * dt, y + _v * dt
        xp, yp = transect.get_xyz_perspective(trans_mat=trans_mat, xs=x.values, ys=y.values)
        xp_moved, yp_moved = transect.get_xyz_perspective(trans_mat=trans_mat, xs=x_moved.values, ys=y_moved.values)
        # remove vectors that have nan on moved pixels
        xp_moved[np.isnan(x_moved)] = np.nan
        yp_moved[np.isnan(y_moved)] = np.nan

        self._obj["xp"][:] = xp[:]
        self._obj["yp"][:] = yp[:]
        u, v = xp_moved - self._obj["xp"], yp_moved - self._obj["yp"]
        return u, v, s

    def get_uv_geographical(self):
        """Get x-directional (u), y-directional (v) and scalar velocity in geo projection from transect dataset.

        Returns
        -------
        u : np.ndarray
            x-directional velocity
        v : np.ndarray
            y-directional velocity
        s : np.ndarray
            scalar velocity

        """
        v_eff = "v_eff"
        v_dir = "v_dir"
        u = self._obj[v_eff] * np.sin(self._obj[v_dir])
        v = self._obj[v_eff] * np.cos(self._obj[v_dir])
        s = self._obj[v_eff]
        aff = self.transect.camera_config.transform
        theta = np.arctan2(aff.d, aff.a)
        # rotate velocity vectors along angle theta to match the requested projection. this only changes values
        # in case of camera projections
        u, v = helpers.rotate_u_v(u, v, theta)
        return u, v, s

    def get_uv_local(self):
        """Get x-directional (u), y-directional (v) and scalar velocity in local projection from transect dataset.

        Returns
        -------
        u : np.ndarray
            x-directional velocity
        v : np.ndarray
            y-directional velocity
        s : np.ndarray
            scalar velocity

        """
        v_eff = "v_eff"
        v_dir = "v_dir"
        u = self._obj[v_eff] * np.sin(self._obj[v_dir])
        v = self._obj[v_eff] * np.cos(self._obj[v_dir])
        s = self._obj[v_eff]
        return u, v, s


class _Velocimetry_PlotMethods:
    """Enable use of `ds.velocimetry.plot` functions as attributes on a Dataset containing velocimetry results.

    For example, `ds.velocimetry.plot.pcolormesh`. When called without a subfunction, quiver will be used
    """

    def __init__(self, velocimetry):
        # make the original dataset also available on the plotting object
        self.velocimetry = velocimetry
        self._obj = velocimetry._obj
        # Add to class _PlotMethods

    def __call__(self, method="quiver", **kwargs):
        """call.

        Parameters
        ----------
        method : str, optional
            plot method to use "quiver", "scatter", "streamplot" or "pcolormesh" (default: "quiver")
        **kwargs : dict
            additional keyword arguments to wrapped Matplotlib function.

        Returns
        -------
        p : matplotlib mappable
            mappable of wrapped matplotlib function

        """
        return getattr(self, method)(**kwargs)

    def get_uv_geographical(self):
        """Get x-directional (u), y-directional (v) and scalar velocity in camera projection from velocimetry dataset.

        Returns
        -------
        u : np.ndarray
            x-directional velocity
        v : np.ndarray
            y-directional velocity
        s : np.ndarray
            scalar velocity

        """
        # select lon and lat variables as coordinates
        velocimetry = self._obj.velocimetry
        u = self._obj["v_x"]
        v = -self._obj["v_y"]
        s = (u**2 + v**2) ** 0.5
        aff = velocimetry.camera_config.transform
        theta = np.arctan2(aff.d, aff.a)
        # rotate velocity vectors along angle theta to match the requested projection. this only changes values
        # in case of camera projections
        u, v = helpers.rotate_u_v(u, v, theta)
        return u, v, s

    def get_uv_local(self):
        """Get x-directional (u), y-directional (v) and scalar velocity in local projection from velocimetry dataset.

        Returns
        -------
        u : np.ndarray
            x-directional velocity
        v : np.ndarray
            y-directional velocity
        s : np.ndarray
            scalar velocity

        """
        u = self._obj["v_x"].values
        v = -self._obj["v_y"].values
        s = (u**2 + v**2) ** 0.5
        return u, v, s

    def get_uv_camera(self, dt=0.1):
        """Get x-directional (u), y-directional (v) and scalar velocity in camera projection from velocimetry dataset.

        Parameters
        ----------
        dt : float, optional
            time step [s] used to transpose velocities over a short distance for projection (default: 0.1)

        Returns
        -------
        u : np.ndarray
            x-directional velocity
        v : np.ndarray
            y-directional velocity
        s : np.ndarray
            scalar velocity

        """
        # retrieve the backward transformation array from x, y to persective row column
        velocimetry = self._obj.velocimetry
        camera_config = velocimetry.camera_config
        # M = velocimetry.camera_config.get_M(velocimetry.h_a, to_bbox_grid=True, reverse=True)
        src = camera_config.get_bbox(camera=True, h_a=velocimetry.h_a, expand_exterior=False).exterior.coords[0:4]
        dst_xy = camera_config.get_bbox(expand_exterior=False).exterior.coords[0:4]
        # get geographic coordinates bbox corners
        dst = cv.transform_to_bbox(dst_xy, camera_config.bbox, camera_config.resolution)
        M = cv.get_M_2D(src, dst, reverse=True)

        # get the shape of the original frames
        shape_y, shape_x = velocimetry.camera_shape
        xi, yi = np.meshgrid(self._obj.x, self._obj.y)
        # flip the y-coordinates to match the row order used by opencv
        yi = np.flipud(yi)

        # follow the velocity vector over a short distance (dt*velocity)
        x_moved, y_moved = xi + self._obj["v_x"] * dt, yi + self._obj["v_y"] * dt
        # project the found displacement points to camera projection
        xp_moved, yp_moved = helpers.xy_to_perspective(
            x_moved.values, y_moved.values, velocimetry.camera_config.resolution, M
        )
        xp, yp = helpers.xy_to_perspective(xi, yi, velocimetry.camera_config.resolution, M)
        # convert row counts to start at the top of the frame instead of bottom
        yp_moved = shape_y - yp_moved
        yp = shape_y - yp
        # missing values end up at the top-left, replace these with nan
        yp_moved[yp_moved == shape_y] = np.nan  # ds["yp"].values[yp_moved == shape_y]
        xp_moved[xp_moved == 0] = np.nan  # ds["xp"].values[xp_moved == 0]

        # estimate the projected velocity vector
        u, v = xp_moved - xp, yp_moved - yp
        self._obj["xp"][:] = xp[:]
        self._obj["yp"][:] = yp[:]
        s = ((self._obj["v_x"] ** 2 + self._obj["v_y"] ** 2) ** 0.5).values
        return u, v, s


@_base_plot
def quiver(_, x, y, u, v, s=None, ax=None, **kwargs):
    """Create quiver plot from velocimetry results on new or existing axes.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.quiver`.
    """
    if "color" in kwargs:
        # replace scalar colors by one single color
        primitive = ax.quiver(x, y, u, v, **kwargs)
    else:
        primitive = ax.quiver(x, y, u, v, s, **kwargs)
    #
    return primitive


@_base_plot
def scatter(_, x, y, c=None, ax=None, **kwargs):
    """Create scatter plot of velocimetry or transect results on new or existing axes.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.scatter`.
    """
    primitive = ax.scatter(x, y, c=c, **kwargs)
    return primitive


@_base_plot
def streamplot(_, x, y, u, v, s=None, ax=None, linewidth_scale=None, **kwargs):
    """Create streamplot of velocimetry results on new or existing axes.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.streamplot`. Additional input arguments:
    """
    if linewidth_scale is not None:
        kwargs["linewidth"] = s * linewidth_scale
    if "color" in kwargs:
        primitive = ax.streamplot(x, y, u, v, **kwargs)
    else:
        primitive = ax.streamplot(x, y, u, v, color=s, **kwargs)
    return primitive.lines


@_base_plot
def pcolormesh(_, x, y, s=None, ax=None, **kwargs):
    """Create pcolormesh plot from velocimetry results on new or existing axes.

    Wraps :py:func:`matplotlib:matplotlib.pyplot.pcolormesh`.
    """
    primitive = ax.pcolormesh(x, y, s, **kwargs)
    return primitive


def cbar(ax, p, size=12, **kwargs):
    """Add colorbar to existing axes.

    In case camera mode is used, the colorbar will get a bespoke layout and will be placed inside the axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the colorbar will be added.
    p : matplotlib mappable
        used to define colorbar
    size : float, optional
        fontsize, used for colorbar title
    **kwargs :
        dict, additional settings passed to plt.colorbar

    Returns
    -------
    cbar : matplotlib colorbar
        handle to colorbar

    """
    label_format = "{:,.2f}"
    path_effects = [
        patheffects.Stroke(linewidth=2, foreground="w"),
        patheffects.Normal(),
    ]
    cax = ax.inset_axes([0.05, 0.05, 0.02, 0.25])
    cb = ax.figure.colorbar(p, cax=cax, **kwargs)
    ticks_loc = cb.get_ticks().tolist()
    cb.set_ticks(mticker.FixedLocator(ticks_loc))
    cb.set_ticklabels([label_format.format(x) for x in ticks_loc], path_effects=path_effects, fontsize=size)
    cb.set_label(label="velocity [m/s]", size=size, path_effects=path_effects)
    return cb


def plot_text(ax, ds, prefix, suffix):
    """Add text with info on transect to plot in standardized manner.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object where the text will be plotted.
    ds : xarray.Dataset
        The dataset containing transect and river flow information.
    prefix : str
        The string that will appear before the main content of the text.
    suffix : str
        The string that will appear after the main content of the text.

    """
    if "q" not in ds:
        return
    _ds = copy.deepcopy(ds)
    xloc = 0.95
    yloc = 0.95
    _ds.transect.get_river_flow(q_name="q")
    Q = np.abs(_ds.river_flow)
    string = prefix
    string += "Water level: {:1.2f} m\nDischarge: {:1.2f} m3/s".format(_ds.transect.h_a, Q.values)
    if "q_nofill" in ds:
        _ds.transect.get_river_flow(q_name="q_nofill")
        Q_nofill = np.abs(_ds.river_flow)
        perc_measured = Q_nofill / Q * 100  # fraction that is truly measured compared to total
        string += " ({:1.0f}% measured)".format(perc_measured.values)
    # reset river flow if necessary
    string += suffix
    ax.text(
        xloc,
        yloc,
        string,
        size=24,
        horizontalalignment="right",
        verticalalignment="top",
        path_effects=path_effects,
        transform=ax.transAxes,
    )


def _prepare_axes(ax=None, mode="local", rotation=None):
    """Prepare the axes, needed to plot results, called from `pyorc.PIV.plot`.

    Parameters
    ----------
    ref : Frames, Transect or Velocimetry
        object to derive plot from
    ax : plt.axes, optional
        if not provided (default), a new axes is prepared (default: None)
    mode : str, optional
        mode to plot, can be "local" (default), "geographical" or "camera".
    rotation : Int[0, 90, 180, 270], optional
        Rotation of image (if vertically oriented, axis sizes are also rotated).


    Returns
    -------
    ax : plt.axes

    """
    if ax is not None:
        if mode == "geographical":
            # ensure that the axes is a geoaxes
            from cartopy.mpl.geoaxes import GeoAxesSubplot

            assert isinstance(
                ax, GeoAxesSubplot
            ), "For mode=geographical, the provided axes must be a cartopy GeoAxesSubplot"
        return ax

    # make a screen filling figure with black edges and faces
    if rotation in [90, 270]:
        f = plt.figure(figsize=(9, 16), frameon=False, facecolor="k")
        f.set_size_inches(9, 16, True)
    else:
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


# add all required plot methods below
_Velocimetry_PlotMethods.quiver = quiver
_Velocimetry_PlotMethods.pcolormesh = pcolormesh
_Velocimetry_PlotMethods.scatter = scatter
_Velocimetry_PlotMethods.streamplot = streamplot
_Transect_PlotMethods.quiver = quiver
_Transect_PlotMethods.scatter = scatter
