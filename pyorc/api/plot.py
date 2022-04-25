import functools
import matplotlib.pyplot as plt
import numpy as np

import pyorc.plot as plot_orc
from pyorc import helpers

def _base_plot(plot_func):
    commondoc = """
    :param ds: Dataset containing velocimetry results
    :param mode: str, perspective mode to use for plotting. Can be "local", "geographical", or "camera". For 
        "geographical" a velocimetry result that contains "lon" and "lat" coordinates must be provided 
        (i.e. produced with known CRS for control points).
    :param ax: matplotlib axes object, optional
        If ``None``, use the current axes. Not applicable when using facets.
    :param **kwargs:  optional, additional keyword arguments to wrapped Matplotlib function.
    :return: mappable from wrapped Matplotlib function
    """
    # Build on the original docstring
    plot_func.__doc__ = f"{plot_func.__doc__}{commondoc}"

    # apply wrapper to allow for partial update of the function, with updated docstring
    @functools.wraps(plot_func)
    def get_plot_method(ref, mode="local", ax=None, *args, **kwargs):
        """
        Retrieve plot method with all required inputs
        :param ref: velocimetry object
        :param mode: str, perspective mode to use for plotting. Can be "local", "geographical", or "camera". For
            "geographical" a velocimetry result that contains "lon" and "lat" coordinates must be provided
            (i.e. produced with known CRS for control points).
        :param ax:
        :param args:
        :param kwargs:
        :return:
        """
        ax = plot_orc.prepare_axes(ax=ax, mode=mode)
        # update ax
        if len(ref._obj["v_x"].shape) > 2:
            raise OverflowError(
                f'Dataset\'s variables should only contain 2 dimensions, this dataset '
                f'contains {len(ref._obj["v_x"].shape)} dimensions. Reduce this by applying a reducer or selecting a time step. '
                f'Reducing can be done e.g. with ds.mean(dim="time", keep_attrs=True) or slicing with ds.isel(time=0)'
            )
        assert mode in ["local", "geographical", "camera"], 'Mode must be "local", "geographical" or "camera"'
        if mode == "local":
            x = ref._obj["x"].values
            y = ref._obj["y"].values
            u = ref._obj["v_x"].values
            v = -ref._obj["v_y"].values
            s = (u ** 2 + v ** 2) ** 0.5
        elif mode == "geographical":
            # import some additional packages
            import cartopy.crs as ccrs
            # add transform for GeoAxes
            kwargs["transform"] = ccrs.PlateCarree()
            x = ref._obj["lon"].values
            y = ref._obj["lat"].values
            u, v, s = ref.get_uv_geographical()
        else:
            # mode is camera
            x = ref._obj["xp"].values
            y = ref._obj["yp"].values
            u, v, s = ref.get_uv_camera()
        return plot_func(x, y, u, v, s, ax, *args, **kwargs)
    return get_plot_method

class _Velocimetry_PlotMethods:
    """
    Enables use of ds.velocimetry.plot functions as attributes on a Dataset containing velocimetry results.
    For example, Dataset.velocimetry.plot.pcolormesh
    """

    def __init__(self, velocimetry):
        # make the original dataset also available on the plotting object
        self.velocimetry = velocimetry
        self._obj = velocimetry._obj
        # Add to class _PlotMethods
        setattr(_Velocimetry_PlotMethods, "quiver", _quiver)

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Dataset.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ds.plot.scatter(...)"
        )


    def get_uv_geographical(self):
        """
        Returns lon, lat coordinate names and u (x-directional) and v (y-directional) velocities, and a rotation that must be
        applied on u and v, so that they plot in a geographical space. This is needed because the raster of PIV results
        is usually rotated geographically, so that water always flows from left to right in the grid. The results can be
        directly forwarded to a plot function for velocities in a geographical map, e.g. overlayed on a background
        image, projected to lat/lon.

        :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
        :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
        :return: 6 outputs: 5 np.ndarrays containing longitude, latitude coordinates, u and v velocities and scalar velocity;
            and float with counter-clockwise rotation in radians, to be applied on u, v to plot in geographical space.

        """
        # select lon and lat variables as coordinates
        u = self._obj["v_x"]
        v = -self._obj["v_y"]
        aff = self.camera_config.transform
        theta = np.arctan2(aff.d, aff.a)
        # rotate velocity vectors along angle theta to match the requested projection. this only changes values
        # in case of camera projections
        u, v = helpers.rotate_u_v(u, v, theta)

        return u, v

    def get_uv_camera(self, dt=0.1, v_x="v_x", v_y="v_y"):
        """
        Returns row, column locations in the camera objective, and u (x-directional) and v (y-directional) vectors, scaled
        and transformed to the camera objective (i.e. vectors far away are smaller than closeby, and follow the river direction)
        applied on u and v, so that they plot in a geographical space. This is needed because the raster of PIV results
        is usually rotated geographically, so that water always flows from left to right in the grid. The results can be
        used to plot velocities in the camera perspective, e.g. overlayed on a background image directly from the camera.

        :param dt: float, time difference [s] used to scale the u and v velocities to a very small distance to project with
            default: 0.1, usually not needed to modify this.
        :param v_x: str, name of variable in ds, containing x-directional (u) velocity component (default: "v_x")
        :param v_y: str, name of variable in ds, containing y-directional (v) velocity component (default: "v_y")
        :return: 5 outputs: 4 np.ndarrays containing camera perspective column location, row location, transformed u and v
            velocity vectors (no unit) and the scalar velocities (m/s). Rotation is not needed because the transformed
            u and v components are already rotated to match the camera perspective. counter-clockwise rotation in radians.
        """
        # retrieve the backward transformation array
        M = self.camera_config.get_M(self.h_a, reverse=True)
        # get the shape of the original frames
        shape_y, shape_x = self.camera_shape
        xi, yi = np.meshgrid(self._obj.x, self._obj.y)
        # flip the y-coordinates to match the row order used by opencv
        yi = np.flipud(yi)

        x_moved, y_moved = xi + self._obj[v_x] * dt, yi + self._obj[v_y] * dt
        xp_moved, yp_moved = helpers.xy_to_perspective(x_moved.values, y_moved.values, self.camera_config.resolution, M)

        # convert row counts to start at the top of the frame instead of bottom
        yp_moved = shape_y - yp_moved

        # missing values end up at the top-left, replace these with nan
        yp_moved[yp_moved == shape_y] = np.nan  # ds["yp"].values[yp_moved == shape_y]
        xp_moved[xp_moved == 0] = np.nan  # ds["xp"].values[xp_moved == 0]

        u, v = xp_moved - self._obj["xp"], yp_moved - self._obj["yp"]
        s = (self._obj[v_x] ** 2 + self._obj[v_y] ** 2) ** 0.5
        s.name = "radial_sea_water_velocity_away_from_instrument"
        return "xp", "yp", u, v, s

@_base_plot
def _quiver(x, y, u, v, s=None, theta=0., ax=None, *args, **kwargs):
    """
    Creates quiver plot from velocimetry results on new or existing axes

     Wraps :py:func:`matplotlib:matplotlib.pyplot.quiver`.
    """
    if "color" in kwargs:
        # replace scalar colors by one single color
        primitive = ax.quiver(x, y, *[v for v in helpers.rotate_u_v(u, v, theta)], *args, **kwargs)
    else:
        primitive = ax.quiver(x, y, *[v for v in helpers.rotate_u_v(u, v, theta)], s, *args, **kwargs)
    #
    return primitive

@_base_plot
def _pcolormesh(x, y, u, v, s=None, theta=0., ax=None, *args, **kwargs):
    """
    Creates pcolormesh plot from velocimetry results on new or existing axes

     Wraps :py:func:`matplotlib:matplotlib.pyplot.pcolormesh`.
    """
    primitive = ax.pcolormesh(x, y, *[v for v in helpers.rotate_u_v(u, v, theta)], s, **kwargs)
# )
    # primitive = plot_orc.quiver(
    #     ax,
    #     x,
    #     y,
    #     *[v for v in helpers.rotate_u_v(u, v, theta)],
    #     s,
    #     **kwargs
    # )
    # ax.quiver(x, y, u, v, s, *args, **kwargs)
    return primitive


# if scalar:
#     p = plot_orc.quiver(
#         ax,
#         self._obj[x].values,
#         self._obj[y].values,
#         *[v.values for v in helpers.rotate_u_v(u, v, theta)],
#         None,
#         **quiver_kwargs
#     )
