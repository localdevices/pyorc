.. _camera_config_api_gcps:

The ground control points are a simple python dictionary that should follow a certain schema. The schema looks as
follows:

.. code-block::

    {
        "src": [
            [int, int],
            [int, int],
            ...,
        ],
        "dst": [
            [float, float, Optional(float)],
            [float, float, Optional(float)],
            ...,
        ],
        "z_0": Optional(float),
        "h_ref": Optional(float),
        "crs": Optional(int, str)

    }

The coordinates in the ``src`` field are simply the pixel coordinates in your video, where the GCPS are located.
You can look these up by plotting the first frame with ``plt.imshow`` or storing
the first frame to a file and open that in your favorite photo editor and count the pixels there.

``dst`` contains the real-world coordinates, that belong to the same points, indicated in ``src``.
``dst`` must therefore contain either 4 x, y (if the left situation is chosen) or 6 x, y, z coordinates (if the right
situation is chosen.

In both cases you provide the points as a list of lists.

``z_0`` must be provided if 6 randomly placed points are used. If you intend to provide multiple videos with a locally
measured water level, then also provide ``h_ref`` as explained above. In case you have used 4 x, y points at the water surface, then also provide ``z_0``. With this information
the perspective of the water surface is reinterpreted with each video, provided that a water level (as measured with the
installed device) is provided by the user with each new video.

.. note::

    For drone users that only make nadir aimed videos, we are considering to also make an option with only 2 GCPs
    possible. If you are interested in this, kindly make an issue in GitHub. For the moment we suggest to use the 4
    control point option and leave ``z_0`` and ``h_ref`` empty.

Finally a coordinate reference system (CRS) may be provided, that indicates in which CRS the survey was done if this
is available. This is only useful if you also have provided a CRS when creating the camera configuration. If you
for instance measure your control points in WGS84 lat-lon (EPSG code 4326) then pass ``crs=4326`` and your coordinates
will be automatically transformed to the local CRS used for your camera configuration.

A full example that supplies GCPs to the existing camera configuration in variable ``cam_config`` is shown below:

.. code-block:: python

    src = [
        [1335, 1016],
        [270, 659],
        [607, 214],
        [1257, 268]
    ]  # source coordinates on the frames of the movie
    dst = [
        [6.0478836167, 49.8830484917],
        [6.047858455, 49.8830683367],
        [6.0478831833, 49.8830964883],
        [6.0479187017, 49.8830770317]
    ]  # destination locations in long/lat locations, as measured with RTK GNSS.
    z_0 = 306.595  # measured water level in the projection of the GPS device
    crs = 4326  # coordinate reference system of the GPS device, EPSG code 4326 is WGS84 longitude/latitude.

    cam_config.set_gcps(src=src, dst=dst, z_0=z_0, crs=crs)

If you have not supplied ``camera_matrix`` and ``dist_coeffs`` to the camera configuration, then these can be optimized
using the provided GCPs after these have been set using the following without any arguments.

.. code-block:: python

    cam_config.set_intrinsic()

