.. _velocimetry_ug:

Velocimetry
===========

Estimating surface velocity is one of the core methods of *pyorc*. Typically surface velocity is derived from a set
of frames, by analyzing frame to frame movements in time. This can be done with several methods. *pyorc* currently
has implemented one of these methods called "Particle Image Velocimetry"

Particle Image Velocimetry
--------------------------

Particle Image Velocimetry uses cross-correlation methods to estimate the most likely position of an observed pattern
from frame to frame. Therefore there is no interdependency over more than 2 frames. To observe patterns, the total area
of interest is subdivided into small blocks of pixels with overlap, also called "interrogation windows". Currently the
overlap is fixed in *pyorc* on 50% of each block. Say that you have reprojected your data to a 0.02 m resolution with
``frames.project()``, and you select an interrogation window of 20 pixels, then each interrogation window will be
0.02 * 20 = 0.4 m in size in both x- and y-direction. The next window will share 10 pixels with its neighbouring window
and hence a velocity will be resolved for each 0.2 x 0.2 meters.

The only direct requirement for calculating velocimetry is that you have:

- opened a :ref:`camera_config_ug` video with ``pyorc.Video`` (e.g. to an object called ``video``) using a video file and a camera configuration;
- retrieved frames from it using ``video.get_frames`` (e.g. into a ``DataArray`` called ``frames``;
- orthorectified the frames using ``frames.project`` (e.g. into ``frames_proj``.

Naturally, before orthorectification you may decide to apply one or several preprocessing methods as described in
:ref:`frames_man` to improve the visibility of tracers. We highly recommend that in many cases, and you can follow our
examples section to see how you may apply these.

Getting surface velocity from the orthoprojected set of frames stored in object ``frames_proj`` is as easy as calling

.. code:: python

    piv = frames_proj.get_piv()

The size of the interrogation window can already be set in the camera configuration, as shown in :ref:`camera_config_ug`.
However, you may also provide a different window size on the fly using the keyword ``window_size``:

.. code:: python

    # ignore the window size in the camera config and set this to 10 pixels
    piv = frame_proj.get_piv(window_size=10)

Interrogating and storing PIV results
-------------------------------------

The object ``piv`` is a normal ``xarray.Dataset`` object. Therefore, you can use any ``xarray`` functionality to
interrogate the data. An important functionality for instance that you may require, is to reduce the data to a
time-averaged mean or one or more quantiles such as a median. This is important for instance when you want to plot
results in a spatial view:

.. code:: python

    # derive the mean over time
    piv_mean = piv.mean(dim="time")
    # derive the median
    piv_median = piv.median(dim="time")

If you apply such reducers, ``xarray`` can no longer guarantee that the metadata attributes of your data variables remain
valid. Therefore, you normally loose metadata, important to for instance reproject data onto the camera perspective.
Therefore we highly recommend to apply such reducers with ``keep_attrs=True`` to prevent that important attributes
get lost in the process.

.. code:: python

    # derive the mean over time, while keeping the attributes in place
    piv_mean = piv.mean(dim="time", keep_attrs=True)
    # derive the median, while keeping the attributes in place
    piv_median = piv.median(dim="time", keep_attrs=True)

Storing your piv results, either with time in place or after applying reducers can also be done. We recommend using
the NetCDF standard as the data model. *pyorc* also follows the `Climate and Forecast conventions <https://cfconventions.org/>`_.

.. code:: python

    # store results in a file, this will take a while
    piv.to_netcdf("piv_results.nc")

Only when you store or otherwise retrieve data resulting from ``get_piv``, the computations will actually be performed.
Therefore it is normal that only after calling a command that retrieves data, you will need to wait for a while before
data is returned. This may take several minutes for small problems, but for large areas of interest or large amounts of
time steps (or a slow machine) it can also take half an hour or longer. To keep track of progress you can also first
prepare the storage process and the wrap a ``ProgressBar`` from the ``dask.diagnostics`` library.
Below you can find an example how to store data with such a progress bar.

.. code:: python

    # import ProgressBar
    from dask.diagnostics import ProgressBar
    # store results with a progress bar
    delayed_obj = piv.to_netcdf("piv_results.nc", compute=False)
    with ProgressBar():
        results = delayed_obj.compute()

You should then see a progressing bar on your screen while data is stored. If you wish to load your results into
memory after having stored it in a previous session, you can simply use ``xarray`` functionality to do so.

.. code:: python

    import xarray as xr
    piv = xr.open_dataset("piv_results.nc")
    piv

Masking spurious velocities
---------------------------

In many cases, you may find that velocities are not accurately resolved, either consistently in a given location or
region in the area of interest, or in specific time steps for given frame to frame results. Nevertheless, the
``get_piv`` method will return results in those cases, even though these may be incorrect or very inaccurate. Causes
for such spurious or poorly estimated velocities may be:

- very little visible patterns available to trace: this can cause many moments in time in which no velocities are
  observed. If only sometimes a traceable pattern passes by, longer integration time may be needed, e.g. 30 or 60 seconds.
  With low flow velocities (typically 0.5 m/s or lower) longer integration times are often needed to capture enough valid
  velocities.
- poor light conditions, e.g. too dark: causes patterns to be difficult to be distinguished. The pre-processing method
  ``pyorc.Frames.edge_detect`` is useful in this case to strengthen gradients in the patterns before estimating velocities.
- very strong visibility of the bottom: causes patterns on the surface to be more difficult to distinguish from non-moving
  bottom patterns. In part this can be resolved with the ``pyorc.Frames.normalize`` method.
- wind: you may find very nice velocity vectors which show a very different process than what you are looking for.
  Especially when the wind waves are oriented in the same direction as the flow, this is very difficult to resolve.
- poor quality footage: water is typically a relatively uniform and relatively dark surface. If your footage has a low
  bitrate (e.g. 1080p with 2Mbps), then the compression algorithm used will usually decide that the water surface
  contains very little interesting information to store. This results in strong loss of visibility of patterns and hence
  poor results, usually resulting in underestimation of velocities. Cheap IP cameras are notorious condidates for poor
  quality videos and underestimation of velocities and river discharge.

To accomodate masking out valid velocities from spurious ones, we have developed a subclass ``pyorc.Velocimetry.mask``
which contains many masking methods to remove spurious velocities. We refer to the API descriptions on how to apply
each specific masking method, and what parameters may be influenced. Here we provide a general description of how to
apply masks and what to be aware of.

Usually one will use a set of masks, either organized in combination or in cascade, to improve the results. So there
are two ways to apply masks, both can lead to quite different results.

Independent masks
~~~~~~~~~~~~~~~~~

With this approach, you first assemble a set of masks by analyzing your raw results several times independently.
Only after having derived the masks, do you apply them on your data in one go. Below we show a small code example
how that works in the code.

.. code:: python

    # get a mask to remove values that are based on a too low correlation
    mask_corr = piv.velocimetry.mask.corr(tolerance=0.3)
    # get a mask to remove velocities that are lower or higher than a user defined threshold (default 0.1 and 5 m s-1)
    mask_minmax = piv.velocimetry.mask.minmax()
    # get a mask for outliers, that deviate a lot from the mean, measured in standard deviations in time
    mask_outliers = piv.velocimetry.mask.outliers()
    # count per grid cell, how many valid (i.e. non masked) values we have, only when there this is above 50% do we trust
    # the results
    mask_count = ds_mask.velocimetry.mask.count(tolerance=0.5)

    # now apply the resulting masks
    piv_masked = piv.velocimetry.mask([
        mask_corr,
        mask_minmax,
        mask_outliers,
        mask_count
    ])

It is easy to understand that here the order in which we derive the masks will not matter. This is because we only
apply the masks at the very end. Following this approach the last mask we derive will not do anything, because the
variable ``mask_count`` is basically derived from the raw results, which are not likely to contain any masked out values
yet. Hence in many cases it may make sense to first apply a set of masks, for instance those that work on individual
values rather than using a full analysis in time, or a neighbourhood analysis of neighbouring grid cells, and only after
that apply other masks that use counts of valid values, check how well neighbouring values match the value under
consideration or compute standard deviations or variance in time to evaluate how valid a velocity may be.

Conditional masking by cascading masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Therefore, we recommend to consider using cascades of masks, so that already applied masks influence the result of
later applied masks for which an analysis of the values through time is essential. For instance, ``mask.outlier``
checks for each grid cell what the mean and standard deviation of velocities through time is, and then assesses
which velocity values are above or under a certain amount of standard deviations. If this mask method is applied before any
of the masks that work on individual values, of the outliers that may have been removed with those masks will
influence the results of this mask, making it less effective. Cascading can be done, by either deriving a few masks, applying
them, and then deriving more masks using the results of the first masking, or simply by using the ``inplace=True`` flag
which immediate overwrites the velocity vectors with missings where the mask is indicating so. Below we show how that
works.

.. code:: python

    # directly apply a mask to remove values that are based on a too low correlation
    piv.velocimetry.mask.corr(tolerance=0.3, inplace=True)
    # directly apply a mask to remove velocities that are lower or higher than a user defined threshold (default 0.1 and 5 m s-1)
    piv.velocimetry.mask.minmax(inplace=True)
    # directly apply a mask for outliers, that deviate a lot from the mean, measured in standard deviations in time
     piv.velocimetry.mask.outliers(inplace=True)
    # directly apply another mask that remove grid cells entirely when their variance is deemed too high to be trustworthy
    piv.velocimetry.mask.variance(inplace=True)
    # count per grid cell, how many valid (i.e. non masked) values we have, only when there this is above 50% do we trust
    # the results
    piv.velocimetry.mask.count(tolerance=0.5, inplace=True)

In this case, as masks are already applied before ``mask.count`` is called, ``mask.count`` will have effect!
In our experience cascading of masks leads to much better results than independently combined masks.

Some masks may also be applied on time averaged data.

Some specific masks
~~~~~~~~~~~~~~~~~~~

A few masks are worthwhile to mention specifically, as they may lead to unexpected results if you don't know how they
work.

* ``mask.angle``: this mask removes velocities that do not follow an expected flow direction. The default for this
  is left-to right oriented flow in the orthorectified x, y grid, with a tolerance of 0.5 * pi (i.e. 90 degrees).
  This means that if flow in your x, y grid is oriented from bottom to top, almost all your velocities will be removed
  and your filtered result will be empty. In streams with a very clear dominant flow direction however, this filter is
  very useful. To ensure your flow follows a left-to-right direction, the selection of corner points in your camera
  configuration is important. If you select these in the right order, the orientation will be correct. The right order
  should be:

  - upstream left bank
  - downstream left bank
  - downstream right bank
  - upstream right bank

  The angle masking method can then be applied as follows:

  .. code::

      mask_angle = piv.velocimetry.mask.angle() # add inplace=True if you want to apply directly

* ``mask.window_median``: this mask can only be applied on time-reduced results and analyses (instead of time series) values
  of neighbours in a certain window defined by parameter ``wdw``. ``wdw=1`` means that a one left/right/above/under
  window is analyzed resulting in a 3x3 window. If the velocity in the cell under consideration is very different
  from the mean of its surrounding cells (defined by ``tolerance`` and measured as a relative velocity to the mean)
  the value is removed. Windows can also be defined with specific strides in x and y direction. See :ref:`api`

* ``mask.window_nan``: this mask can only be applied on time-reduced results and analyses (instead of time series) values
  of neighbours in a certain window defined by parameter ``wdw``. If there are too many missings in the window, then the value considered
  is also removed. This is meant to remove isolated values.


Plotting
--------

Another subclass is available for very easy and convenient plotting of results under ``pyorc.Velocimetry.plot``.
The plotting methods so far are only geospatial of nature (i.e. no time series) and therefore require that you first
reduce any time variable results over time, for instance with:

.. code::

    # assume that processed and masked results are in piv
    piv_mean = piv.mean(dim="time", keep_attrs=True)

.. note::
    The ``keep_attrs=True`` flag is quite important here as we may need the attributes to reproject data from the default
    x, y projection to the original camera perspective for instance.

After reducing, we can use a set of methods to make plots, in a very similar manner as used for plotting
:ref:`frames_man`. In fact, you can smartly combine plots from a frame, with plots of your velocimetry results and by
doing so create beautiful augmented reality views, or geospatial views. The different plotting method all use an
underlying ``matplotlib.pyplot`` function and are also named accordingly. Hence they can receive keyword arguments
specific to these underlying functions. In addition, a few additional important keywords can be added:

* ``add_colorbar=True``: this flag will add a default colorbar to the axes
* ``mode``: this flag can be set to ``local`` (default), ``geographical`` (for plots in a geospatial view) and
  ``camera`` for augmented reality in the camera's original perspective.

Each plotting method always returns the mappable so that you can make your own colorbars, and refer to its parent axes
and in turn the axes parent figure.

The different plotting methods are summarised below.

+----------------------------------------+---------------------------------------------------------------------------------+
| Plot function                          | Description                                                                     |
+========================================+================================================+================================+
| ``pyorc.Velocimetry.plot.pcolormesh``  | computes scalar velocities and plots these as a gridded mesh                    |
+----------------------------------------+---------------------------------------------------------------------------------+
| ``pyorc.Velocimtry.plot.scatter``      | computes scalar velocities and plots as colored dots                            |
+----------------------------------------+---------------------------------------------------------------------------------+
| ``pyorc.Velocimetry.plot.streamplot``  | draws a streamplot through the x and y-directional velocities. Only works with  |
|                                        | ``mode="local"``                                                                |
+----------------------------------------+---------------------------------------------------------------------------------+
| ``pyorc.Velocimetry.plot.quiver``      | draws a quiver plot using the x and y-directional velocities. If                |
|                                        | ``pyorc.Velocimetry.plot`` is called directly, this method is used              |
+----------------------------------------+---------------------------------------------------------------------------------+

Assuming you have rgb frames in an object called ``frames_rgb`` and your mean-in-time piv results in ``piv_mean``, a very nice
augmented reality view can be drawn as follows:

.. code:: python

    # use Normalize to create a scale for quiver (matplotlib.pyplot.quiver does not accept vmin or vmax)
    from matplotlib.colors import Normalize

    plot the first frame
    p = da_rgb_proj[0].frames.plot()

    #...and then plot the filtered velocimetry (quiver is default) on top of the axes of mappable p
    p = piv_mean.velocimetry.plot(
        ax=p.axes,
        alpha=0.4,
        cmap="rainbow",
        scale=20,
        width=0.0015,
        norm=Normalize(vmax=0.6, clip=False),
        add_colorbar=True
    )

    # store the parent figure object to a file
    p.axes.figure.savefig("piv_result.jpg", dpi=200)

.. image:: ../_images/example_geul_small.jpg

