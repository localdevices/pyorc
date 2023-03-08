.. _camera_config_cli_geo:

To provide a geographical coordinate reference system to your camera configuration,
you may simply add the option ``--crs`` to the earlier command. Below an example
is given where as local projection, the UTM37S projection is provided. This projection
has EPSG code 32737 and is for instance applicable over Tanzania. A projection is meant
to provide a mostly undistorted real distance map for a given area. Because the globe
is round, a suitable projection system must be chosen, that belongs to the area of
interest. Visit https://epsg.io/ to find a good projection system for your area of
interest.

.. code-block:: shell

    $ pyorc camera-config --crs 32737 ........

The dots represent additional required to make a full camera configuration.


