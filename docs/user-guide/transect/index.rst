.. _transect_ug:

Transects
=========

For many use cases, a user may require to extract velocities over a certain transect. The most famous application for
this is measuring river flow. *pyorc* is equipped with very smart approaches to extract a transect, interpolate
missing values that may appear on the transect, and extracting cross-section cubic meter per second river flow estimates
with confidence intervals based on temporal variations in velocities. It is highly recommended to consider applying
masks on your results before extracting velocities and flow over transects.

Extracting a transect
---------------------

The starting point for a transect is always a velocimetry result, containing surface velocities, gridded in space and
time and a set of coordinates of a cross section, measured in the same coordinate reference system as the ground control
points and using also the same vertical datum as the measurement of the control video water level (``z_0`` in the
camera configuration).

.. tab-set::

    .. tab-item:: Command-line

        Transects can be derived by supplying a ``transect`` section in your recipe. Under this section, one or several
        transects can be derived, by supplying the path to a shapefile or geojson file, that contains x, y, z Point
        coordinates of the measured cross section as well as the coordinate reference system of the coordinates.
        Any metadata or properties are not used. These files are typical outputs of GNSS equipment and can one on
        one be passed onto *pyorc* unless you wish to remove some wrong points, or points that belong to another survey.

        The approach is best explained with an example ``transect`` section for one transect, with some comments

        .. code-block:: yaml

            # we start our transect section
            transect:
                # extracted transects will be written to disk
                write: True
                # we have measured one cross section in the Ngwerere river in Lusaka - Zambia
                ngwerere_1:
                    # path to the shapefile or geojson, containing our points
                    shapefile: ../examples/ngwerere/cross_section1.geojson
                    # supply some extra arguments to get_transect (if not provided, defaults will be used)
                    get_transect:
                        # we do a rolling mean over time to smooth out small fps changes of the camera
                        rolling: 4
                        # we sample velocity by averaging in a one-sided window size of 2 (5x5)
                        wdw: 2
                    # let's derive depth integrated velocity also
                    get_q:
                        # we fill missing with a log interpolation method
                        fill_method: log_interp
                        # assume depth averaged velocity is 0.85 * surface velocity
                        v_corr: 0.85
                    # integrate to m3/s river flow without additional arguments
                    get_river_flow:

        If ``get_transect`` is not supplied in the command-line recipe then default parameters will be used. When
        ``write: True`` is provided, the resulting extracted velocities and flow will be written to a file in the
        output folder. This file will have the following naming convention: ``transect_<name>.nc``. From the example
        above the file would be called ``transect_ngwerere_1.nc``.

    .. tab-item:: API

        You may load x, y, z points from a shapefile or csv file with ``pandas`` or ``geopandas``. Then these can be
        supplied with a velocimetry result, using the ``get_transect`` method. Below a full working example is given, based
        on our Ngwerere example dataset.

        .. code-block:: python

            cross_section = pd.read_csv("ngwerere/ngwerere_cross_section.csv")
            # the points are assigned to numpy arrays, coordinates are in crs 32735
            x = cross_section["x"]
            y = cross_section["y"]
            z = cross_section["z"]

            # extract points, ensuring pyorc knows their projection is 32735.
            ds_points = ds.velocimetry.get_transect(x, y, z, crs=32735, rolling=4)
            # add depth averaged velocity, filling missings and assuming a average velocity / surface velocity of 0.85
            ds_points_q = ds_points.transect.get_q(fill_method="log_interp", v_corr: 0.85)
            # finally, obtain cross sectional river flow (added as variable ["river_flow"])
            ds_points_q.transect.get_river_flow()
            print(ds_points_q["river_flow"])

The ``get_transect`` method derives the transect from our gridded velocimetry results. Many options
may be supplied to improve sampling. For instance, if flow has a highly predominant direction, then it makes
sense to increase sampling over a longer longitudinal window, and reduce sampling in perpendicular direction.
This can be controlled by arguments ``wdw_x_min``, ``wdw_x_max``, ``wdw_y_min`` and ``wdw_y_max``.
There are also options to refine or coarsen the sampling interval over space (``distance``) as well as an option to
discard values if not enough valid (i.e. non masked) values in the sampled neighbourhood are found (``tolerance``).
For a full list of parameters, please investigate the :ref:`API documentation <transects>`. ``get_q`` and ``get_river_flow`` are not
mandatory to use, but recommended if river flow is the required variable of interest. Since river flow requires
estimates of velocities without any gaps, fill methods are available that can be applied before the depth averaged
velocity is derived. The relevant API pages that describe the possible input arguments for both are described
:ref:`here <river_flow>`. Stored results in a NetCDF structure will contain the following variables:

* ``v_eff_nofill``: the effective velocities per point in the cross section [m/s]
* ``v_eff``: if ``get_q`` is supplied, this will contain velocities, with gaps filled with a selected filling
  method within the ``get_q`` method.
* ``v_dir``: the direction of the velocity per point in the cross section determined as the perpendicular direction
  of the supplied cross section. Note that if the cross section is not entirely straight, this may vary from point to
  point.
* ``q_nofill``: depth integrated velocity [m2/s], using a correction factor supplied with ``v_corr`` (default: 0.9)
* ``q``: same as ``v_eff`` but then for depth integrated velocity.
* ``river_flow``: available when both ``get_q`` and ``get_river_flow`` are applied. This variable contains the river flow
  in m3/s.

In the processing to transects, the ``time`` dimension will be replaced by a ``quantile`` dimension. By default 5
quantiles are derived (0.05, 0.25, 0.5, 0.75, and 0.95) for each point in the transect based on the time variability of
the extracted velocities. These quantiles allow you to estimate the variance of the velocities. A very high variance
may indicate more uncertainty. Note that this variance is also integrated into the ``river_flow`` variable, assuming the
variability is entirely correlated in space. This is likely not to be true, especially if variances are caused by
remaining variability in the frames-per-second of the camera, hence the confidence intervals of the ``river_flow``
variable should be interpreted as highly conservative.
