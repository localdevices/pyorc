import matplotlib.pyplot as plt

def prepare_axes(ax=None, mode="local"):
    """
    Prepares the axes, needed to plot results, called from `pyorc.PIV.plot`.

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

def quiver(ax, x, y, u, v, s=None, zorder=3, **kwargs):
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

def cbar(ax, p, mode="local", size=15, color="w"):
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
