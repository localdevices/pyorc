.. _videos:

Videos
======


.. note::

   This manual is a work in progress.

Videos and video files are at the core of *pyorc*. Any analysis starts with a video file. Notably, these can be acquired
from different platforms, with suggestions given in the table below.


+----------------------------------+-----------------------------------------------------------------------------------+
| Platform                         | Use cases                                                                         |
+==================================+===================================================================================+
| Smartphone                       | Ad-hoc collection of data, e.g. during a flood.                                   |
|                                  | Permanent or temporary (e.g. for rating curve updates) and affordable observation |
|                                  | at one location that are orthorectified per video.                                |
|                                  | Permanent or temporary observations at one location with a fixed mould to         |
|                                  | keep a permanent stable objective.                                                |
+----------------------------------+-----------------------------------------------------------------------------------+
| Drone                            | Incidental observations e.g. to monitor velocity distribution around river        |
|                                  | restoration sites, new riverine infrastructure, sediment or waste trapping        |
|                                  | constructions or other environmental related uses.                                |
+----------------------------------+-----------------------------------------------------------------------------------+
| IP-camera                        | Permanent or temporary observations at measurement site without need for physical |
|                                  | presence. This requires a modem and power source. A power cycling setup is        |
|                                  | recommended to take videos in regular intervals.                                  |
+----------------------------------+-----------------------------------------------------------------------------------+
| Trap camera                      | Temporary observations at measurement site without need for physical presence,    |
|                                  | Ideal for wet season flow event capturing over a limited time until batteries     |
|                                  | run out or SD card is full. Typically uses AA batteries                           |
+----------------------------------+-----------------------------------------------------------------------------------+

*pyorc* therefore uses a so-called ``Video`` object to interrogate a video file, add understanding of video's
perspective (using a ``CameraConfig`` object and the water level during which the video was taken)
and define which frame range you wish to use to perform analysis on.

Therefore you can pass a number of properties to a ``Video`` object so as to be able to work with it in *pyorc*.
Please read the API section on the ``Video`` object to get details on how to create such a ``Video`` object. Some
important properties are described below:

Camera configuration
--------------------

Essential for almost all steps after opening a video is to supply a ``CameraConfig`` with a video. This object contains
all information from the camera's lens characteristics, perspective, geographical awareness as derived from a sample
image or frame, with control point information in view. As soon as you wish to work with the frames in the video
(e.g. by calling the method ``get_frames``) you MUST specify a camera configuration. You can find more information in
:ref:`camera_config` section.

The current water level
-----------------------
If you decide to take multiple videos of the same objective, for instance with a fixed camera rig, or with a smartphone
in a mould, then you can reuse the camera configuration for each video. This is because as long as the perspective does
not change (i.e. you are looking at exactly the same scene from the same location) then also the camera configuration
can remain exactly the same. This is ideal for gathering time series, for instance to interpret river flow during an
event or for permanent video observations of river flow. Per video, there is only one thing that must be provided. This
is the present water level. This can be provided in a locally selected datum (e.g. the level as read from a staff gauge)
and will be related to the water level, as read during the survey, used to construct the camera configuration. The
"current" water level (i.e. commensurate with the situation in the video you are currently processing) can be set with
the option ``h_0``.

.. note::

   To guarantee that the perspective does not change, the following conditions MUST be met:

   * The same lens must be used as used for the control image of the camera configuration. Note that smnartphone often
     have multiple lenses e.g. for wide angle versus close-ups. Ensure you have zoom level at the same level as used
     for the control image and do not use digital zoom! It generally only reduces image quality.
   * The camera must be placed at exactly the same location and oriented to exactly the same objective
   * The camera's resolution and other settings must be exactly the same as during the control image.

Frame range
-----------
You may have recorded a relatively long video and only wish to process a subset of frames. This can be controlled with
the ``start_frame`` and ``end_frame`` options. If you set this to an integer larger than 0 and smaller than the maximum
frames available, then only the frames in between will be processed.

Stabilization
-------------
.. note::

    Video stabilization is still experimental. Please raise an issue on Github with a link to a video if you experience
    issues.

Videos may be taken in unsteady conditions. This may happen e.g. with slight movements of a smartphone, a
drone that has varying air pressure conditions or wind gusts to deal with, or even fixed cameras in strong winds. But
also, someone may have taken an incidental video, that was not originally intended to be used for river flow and velocity
observations, but may render important information about a flood. For this the ``stabilize`` option can be passed
with either the value "fixed" for a video without intentional movements, or "moving" for a video with intentional
movements. With this option, each frame will be stabilized with respect to the first frame chosen by the user
(through the option ``first_frame``). The method works by first finding well traceable points in the first frame,
then tracing where these points move to in the next frames. Of course some of these points may actually be moving water.
That is why you have to define whether the video is taken with or without intentional movements, so that moving water
can be distinguished from much smaller (larger) movements in the fixed (moving) video.

.. table:: Small part of 4K drone footage in Rio Grande - Brazil, showing left: no stabilization applied; right:
           stabilization applied with ``stabilize="fixed"``. The algorithm automatically detects rigid points on river
           banks but also on the debris showed in this subscene.

    +-----------------------------------------------------------+----------------------------------------------------------+
    | Unstable                                                  + Stable                                                   |
    +===========================================================+==========================================================+
    | |videounstab|                                             | |videostab|                                              |
    +-----------------------------------------------------------+----------------------------------------------------------+

A working example to obtain a stabilized video from our example section is provided below.

.. code::

    import pyorc

    # set a video filename below, change to your own local file location
    video_file = "examples/ngwerere/ngwerere_20191103.mp4"
    # point to a file containing the camera configuration
    cam_config = pyorc.load_camera_config("examples/ngwerere/ngwerere.json")
    video = pyorc.Video(
        video_file,
        camera_config=cam_config,
        start_frame=0,
        end_frame=125,
        stabilize="fixed"
    )
    video

.. note::

    If you choose to only treat a very short part of a video such as only one second, then it may be difficult for the
    stabilizing functions to distinguish rigid points from non-rigid. In this case we recommend to set ``start_frame``
    and ``end_frame`` to cover a larger time span, and then make a sub-selection after having retrieved the frames
    from the video. This will not be significantly slower, because *pyorc* utilizes a lazy programming approach and
    will then only load and process the frames you select afterwards.

    .. code ::

        # start with a large frame set for detecting rigid points
        video = pyorc.Video(fn, start_frame=0., end_frame=200)
        # get your frames, and only the first 30
        da_frames = video.get_frames()[0:30]
        # do the rest of your work




.. |videostab| image:: ../_images/video_stable.gif
   :scale: 80%
   :align: middle

.. |videounstab| image:: ../_images/video_unstable.gif
   :scale: 80%
   :align: middle