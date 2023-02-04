.. _camera_config_api:

In **pyorc** all camera configuration is collected into one single class called ``CameraConfig``. It can be imported
from the main library and then be used to add information about the lens characteristics and the relation between
real-world and row-column coordinates. Once completed, the camera configuration can be added to a video to make it lens
and geographically aware. You can also add a camera configuration, stored in a JSON-formatted file to a video.
Once this is done, the video has added intelligence about the real-world, and you can orthorectify its frames.
Below a simple example is shown, where only the expected size of the objective in ``height`` and ``width`` is provided.

    .. literalinclude:: ./cam_config.py
        :language: python

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
