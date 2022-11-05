.. _camera_config_ug:

Camera configurations
=====================

.. note::

   This manual is a work in progress.

An essential element in doing optical velocity estimates is understanding how the Field Of View (FOV) of a camera
relates to the real-world coordinates. This is needed so that a camera's FOV can be "orthorectified", meaning it can
be transposed to real-world coordinates with equal pixel distances in meters. For this we need understanding of the
lens characteristics, and understanding of where a pixel (2-dimensional with column and row coordinates, a.k.a.
image coordinates) is located in the real world (3-dimensional, a.k.a. geographical coordinates).
The camera configuration methods of **pyorc** are meant for this purpose.

Setting up a camera configuration
---------------------------------
In **pyorc** all camera configuration is collected into one single class called ``CameraConfig``. It can be imported
from the main library and then be used to add information about the lens characteristics and the relation between
real-world and row-column coordinates. Once completed, the camera configuration can be added to a video to make it lens
and geographically aware. You can also add a camera configuration, stored in a JSON-formatted file to a video.
Once this is done, the video has added intelligence about the real-world, and you can orthorectify its frames.
Below a simple example is shown, where only the expected size of the objective in ``height`` and ``width`` is provided.

.. code-block:: python

    cam_config = pyorc.CameraConfig(height=1080, width=1920)
    cam_config

    {
        "height": 1080,
        "width": 1920,
        "resolution": 0.05,
        "window_size": 10,
        "dist_coeffs": [
            [
                0.0
            ],
            [
                0.0
            ],
            [
                0.0
            ],
            [
                0.0
            ]
        ],
        "camera_matrix": [
            [
                1920.0,
                0.0,
                960.0
            ],
            [
                0.0,
                1920.0,
                540.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ]
    }

You can see that the inline representation of the ``CameraConfig`` object is basically a dictionary with pieces of
information in it. In this example we can already see a few components that are estimated from default values. These
can all be modified, or updated with several methods after the object has been established. The different parts we can
see here already are as follows:

* ``height`` and ``width``: these are simply the height and width of the expected objective of a raw video. You must
  at minimum provide these to generate a ``CameraConfig`` object.
* ``resolution``: this is the resolution in meters, in which you will get your orthoprojected frames, once you have
  a complete ``CameraConfig`` including information on geographical coordinates and image coordinates, and a bounding
  box, that defines which area you are interested in for velocity estimation. As you
  can see, a default value of 0.05 is selected, which in many cases is suitable for velocimetry purposes.
* ``window_size``: this is the amount of orthorectified pixels in which one may expect to find a pattern and also.
  the size of a window in which a velocity vector will be generated. A default is here set at 10. In smaller streams
  you may decide to reduce this number, but we highly recommend not to make it lower than 5, to ensure that there are
  enough pixels to distinguish patterns in. If patterns seem to be really small, then you may decide to reduce the resolution
  instead. **pyorc** automatically uses an overlap between windows of 50% to use as much information as possible over
  the area of interest. With the default settings this would mean you would get a velocity vector every
  0.05 * 10 / 2 = 0.25 meters.
* ``dist_coeffs``: this is a vector of at minimum 4 numbers defining respectively radial (max. 6) and tangential (2)
  distortion coefficients of the used camera lens. As you can see these default to zeros only, meaning we assume no
  significant distortion if you do not explicitly provide information for this.
* ``camera_matrix``: the intrinsic matrix of the camera lens, allowing to transform a coordinate relative to the camera lens
  coordinate system (still in 3D) into an image coordinate (2D, i.e. a pixel coordinate). More on this can be found
  e.g. on `this blog. <https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec>`_

As you can see, the camera configuration does not yet have any real-world information and therefore is not sufficient
to perform orthorectification. Below we describe how you can establish a full camera configuration.

Making the camera configuration geographically aware
----------------------------------------------------
In case you are able to perform your field measurements with a RTK GNSS device, then your camera configuration
can be made entirely geographically aware. You can then export or visualize your results in a geographical map later
on, or use your results in GIS software such as QGIS. You do this simply by passing the keyword ``crs`` to the camera
configuration and enter a projection. Several ways to pass a projection are possible such as:

* EPSG codes (see EPSG.io)
* proj4 strings
* Well-Know-Text format strings (WKT)

Because **pyorc** intends to measure velocities in distance metrics, it is compulsory to select a locally valid meter
projected coordinate reference system, and not for instance an ellipsoidal coordinate system such as the typical
WGS84 latitude longitude CRS. For instance in the Netherlands you may use Rijksdriehoek (EPSG code 28992). In Zambia
the UTM35S projection (EPSG code 32735) is appropriate, whilst in Tanzania, we may select the UTM37S projection (EPSG code
32737). IF you use a non-appropriate or non-local system, you may get either very wrong results, or get errors during
the process. To find a locally relevant system, we strongly recommend to visit the `EPSG site <https://epsg.io>`_ and
search for your location. If you do not have RTK GNSS, then simply skip this step and ensure you make your own local
coordinate system, with unit meter distances.

Once your camera configuration is geographically aware, we can pass all other geographical information we may need in
any projection, as long as we notify the camera configuration which projection that is. For instance, if we measure
our ground control points (GCPs, see later in this manual) with an RTK GNSS set, and store our results as WGS84 lat-lon
points, then we do not have to go through the trouble of converting these points into the system we chose for our camera
configuration. Instead we just pass the CRS of the WGS84 lat-lon (e.g. using the EPSG code 4326) while we add the GCPs
to our configuration. We will see this later in this manual.

Below, we show what the configuration would look like if we would add the Rijksdriehoek projection to our camera
configuration. You can see that the code is converted into a Well-Known-Text format, so that it can also easily be
stored in a generic text (json) format.

.. code-block:: python

    cam_config = pyorc.CameraConfig(height=1080, width=1920, crs=28992)
    cam_config

    {
        "height": 1080,
        "width": 1920,
        "crs": "PROJCRS[\"Amersfoort / RD New\",BASEGEOGCRS[\"Amersfoort\",DATUM[\"Amersfoort\",ELLIPSOID[\"Bessel 1841\",6377397.155,299.1528128,LENGTHUNIT[\"metre\",1]]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433]],ID[\"EPSG\",4289]],CONVERSION[\"RD New\",METHOD[\"Oblique Stereographic\",ID[\"EPSG\",9809]],PARAMETER[\"Latitude of natural origin\",52.1561605555556,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8801]],PARAMETER[\"Longitude of natural origin\",5.38763888888889,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8802]],PARAMETER[\"Scale factor at natural origin\",0.9999079,SCALEUNIT[\"unity\",1],ID[\"EPSG\",8805]],PARAMETER[\"False easting\",155000,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8806]],PARAMETER[\"False northing\",463000,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8807]]],CS[Cartesian,2],AXIS[\"easting (X)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],AXIS[\"northing (Y)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]],USAGE[SCOPE[\"Engineering survey, topographic mapping.\"],AREA[\"Netherlands - onshore, including Waddenzee, Dutch Wadden Islands and 12-mile offshore coastal zone.\"],BBOX[50.75,3.2,53.7,7.22]],ID[\"EPSG\",28992]]",
        "resolution": 0.05,
        "window_size": 10,
        "dist_coeffs": [
            [
                0.0
            ],
            [
                0.0
            ],
            [
                0.0
            ],
            [
                0.0
            ]
        ],
        "camera_matrix": [
            [
                1920.0,
                0.0,
                960.0
            ],
            [
                0.0,
                1920.0,
                540.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ]
    }

.. note::

   A smart phone also has a GNSS chipset, however, this is by far not accurate enough to provide the measurements needed
   for **pyorc**. We recommend using a (ideal!) RTK GNSS device with a base station setup close enough to warrant
   accurate results, or otherwise a total station or spirit level.


Storing and loading a camera configuration
------------------------------------------

TODO

Camera intrinsic matrix and distortion coefficients
---------------------------------------------------
An essential component to relate the FOV to the real world is the camera's *intrinsic* parameters, i.e. parameters
that define the dimensions and characteristics of the used camera lens and its possible distortion. As an example, a
smartphone camera has a very flat lens, with a short focal distance. This often results in the fact that objects or
people at the edges of the field of view seem stretched, while the middle is quite reliable as is.
With a simple transformation, such distortions can be corrected.
Fish eye lenses, which are very popular in trail cameras, IP cameras and extreme sport cameras, are constructed to
increase the field of view at the expense of so-called radial distortions. With such lenses, straight lines may become
distorted into bend lines in your objective. Imagine that this happens with a video you wish to use for velocimetry,
then your geographical referencing can easily be very wrong (even in the order of meters with wide enough streams)
if you do not properly account for these. If for example your real-world coordinates are measured somewhere in the
middle of the FOV, then velocities at the edges are likely to be overestimated.

The default parameters (i.e. no distortion and an ideal world camera intrinsic matrix) may therefore be insufficient
and can lead to unnecessary error in the interpretation of the real world distances in the FOV. To
establish a more sound camera intrinsic matrix and distortion coefficients, we recommend to take a video of
a checkerboard pattern using the exact settings you will use in the field and perform camera calibration with this.
Below you can see an animated .gif of such a video. Basically, you print a checkerboard pattern, hold it in front of
your camera, ensure that you run video at the exact settings at which you intend to record in the field,
and capture the printed checkerboard pattern from as many angles as possible. Include rotation and movements in all
directions.

Preparing a video for camera calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a method available to establish an intrinsic matrix and distortion coefficients. It reads in a video in which
a user shows a chessboard pattern and holds it in front of the camera in many different poses and at as many different
locations in the field of view as possible. It then strips frames in a staggered manner starting with the first and
last frame, and then the middle frame, and then the two frames in between the first, last and middle, and so on, until
a satisfactroy number of frames have been found in which the chessboard pattern was found. The intrinsic matrix and
distortion coefficients are then calculated based on the results, and added to the camera configuration.

.. note::

   Making a video of a chessboard pattern and calibrating on it is only uyseful if you do it the right way. Take care
   of the following guidelines:

   * ensure that the printed chessboard is carefully fixed or glued to a hard object, like a strong straight piece of
     cardboard or a piece of wood. Otherwise, the pattern may look wobbly and cause incorrect calibration
   * a larger chessboard pattern (e.g. A0 printed) shown at a larger distance may give better results because the
     focal length is more similar to field conditions.
   * make sure that while navigating you cover all degrees of freedom. This means you should move the checkerboard
     from top to bottom and left to right; in all positions, rotate the board around its horizontal and vertical
     middle line; and rotate it clockwise.
   * make sure you record the video in exactly the same resolution as you intend to use during the taking of the videos
     in the field.

  If the calibration process is not carefully followed it may do more harm than good!!!

An example of extracts from a calibration video with found corner points is shown below. It gives an impression of how
you can move the chessboard pattern around. As said above, it is better to print a larger chessboard and show that to
the camera at a larger distance.

.. image:: ../../_images/camera_calib.gif

Lens calibration method
~~~~~~~~~~~~~~~~~~~~~~~
Once you have your video, the camera calibration is very simple. After creating your camera configuration you can
supply the video in the following manner:

.. code-block:: python

    calib_video = "calib_video.mp4"
    cam_config.set_lens_calibration(calib_video)

When you execute this code, the video will be scanned for suitable images, and will select frames that are relatively
far apart from each other. When a suitable image with patterns is found, the algorithm will show the image and the found
chessboard pattern. There are several options you may supply to the algorithm to influence the amount of internal corner
points of the chessboard (default is 9x6), the maximum frames number that should be used for calibration,
filtering of poorly performing images, switch plotting and writing plots to files (for later checking of the results)
on or off.

.. note::
   the camera calibration is still experimental. If you have comments or issues kindly let us know by making a github
   issue.

Ground control points
---------------------
Besides the characterization of the lens used for taking the video, we must also characterise the camera to real-world
coordinate system. In other words: we must know where a row and column in our camera perspective may lie in the real
world. Naturally, this is a poorly defined problem as your camera's perspective can only be 2D, whilst the real world
has 3 dimensions. However, our problem is such that we can always fix one dimension, i.e. the elevation. If we already
know and fix the level of the water (z-coordinate), then we can interpret the remaining x-, and y-coordinates if we
give the camera calibration enough information to interpret the perspective. We do this by providing so-called ground
control points, that are visible in the FOV, and of which we know the real-world coordinates.

Structure of ground control points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

The fields have the following meaning:

* ``src`` contains [column, row] locations of the control points in the FOV.
* ``dst``: contains [x, y] locations (in case you use 4 control points on one vertical plane) or [x, y, z] locations (
  in case you use 6 control points with arbitrary elevation).
* ``z_0``: water level measured in the vertical reference of your measuring device (e.g. RTK GNSS)
* ``h_ref``: water level as measured by a local measurement device such as a staff gauge
* ``crs``: the CRS in which the control points are measured. This can be different from the CRS of the camera
  configuration itself in which case the control points are automatically transformed to the CRS of the camera
  configuration. If left empty, then it is assumed the CRS of the measured points and the camera configuration is the
  same.

Measuring the GCP information
-----------------------------

Below we describe how the information needed should be measured in the field during a dedicated survey. This is
typically done every time when you do an incidental observation, or once during the installation of a fixed camera.
If you leave the camera in place, you can remove control points after the survey.

Example of survey situations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You will notice in the next sections that you can typically measure either 4 control points at one vertical plane
(water surface) or 6 or more points at random elevations. You prepare this situation by spreading easy to recognize
markers over your Field of View. In the figure below you see two examples, one where 4 sticks were placed in the water
and the interface of the sticks with the water (red dots) is measured. And one where 6 black-and-whiter markers are
spread over the field of view.

.. table:: Examples of ground control markers and situations

    +----------------------------------------------------------------------------------------------------------------+
    | 4 GCPt at water surface - Chuo Kikuu River, Dar es Salaam, Tanzania                                            |
    +----------------------------------------------------------------------------------------------------------------+
    | |gcps_4|                                                                                                       |
    +----------------------------------------------------------------------------------------------------------------+
    |  6 (+) GCPs spread over banks and FOV - Geul River, Limburg, The Netherlands                                   |
    +----------------------------------------------------------------------------------------------------------------+
    | |gcps_6|                                                                                                       |
    +----------------------------------------------------------------------------------------------------------------+

The schematic below shows in a planar view what the situation looks like. It is important that the control points are
nicely spread over the Field of View, and this is actually more important than an equal spread of points of left and
right bank. In the schematic we show this by having only 2 control points at the bank close to the camera, and 4 at
the opposite side. If you have your camera on a bridge in the middle of the bridge deck, then having 3 (or more) points
left as well as right makes the most sense.

.. figure:: ../../_images/site_schematic_planar.svg

   Planar schematic view of site survey situation.

Ensuring that the vertical plane is fully understood is also important.
The ``z_0`` and ``h_ref`` optional keys are meant to allow a user to provide multiple videos with different water
levels. If you intend to do this, you may install a water level measuring device on-site such as a staff gauge or
pressure gauge, that has its own vertical zero-level reference. Therefore, to use this option the following should be
measured and entered:

* measure the water level during the survey with your local device (e.g. staff gauge) and insert this in ``h_ref``
* also measure the water level with your survey device such as total station or RTK GPS. This has its own vertical zero
  level. This level must be inserted in ``z_0``. Any other surveyed properties such as the lens position and the
  river cross section must be measured with the same horizontal and vertical coordinate system as ``z_0``.

The overview of these measures is also provided in the schematic below.

.. figure:: ../../_images/site_schematic_cs.svg

   Cross-section schematic view of site survey situation.


Entering control points in the camera configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Setting the lens position
-------------------------

TODO

Setting the area of interest
----------------------------

TODO

CameraConfig properties
-----------------------

TODO

CameraConfig plots
------------------

TODO





.. |gcps_4| image:: ../../_images/ChuoKikuu_GCPs.jpg

.. |gcps_6| image:: ../../_images/Geul_GCPs.jpg

