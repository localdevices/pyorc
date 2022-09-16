.. _manual:

======
Manual
======

.. note::

   This manual is a work in progress.

*pyorc* is a python API that allows you to process a video of a river or stream into surface velocities and river flow
estimates. The entire process, including stabilization, preprocessing, velocity estimation, filtering of spurious
velocities, deriving transect velocities and river flow estimation is supported. Why is *pyorc* attractive?

Solid data model
----------------
We rely on *xarray* for data models. *xarray* provides access and intelligent dimensional computations over
multi-dimensional datasets with coordinate systems that can relate to time, space or other logical dimensions.

Lazy computations
-----------------
The main idea of *pyorc* is that all computations are performed lazily, meaning that as a user, you can first setup and
investigate the entire pipeline of computations, and only at the moment when you request actual numbers from the
pipeline, will *pyorc* perform computations. And then only those strictly needed to provide you the number(s) you want.
This makes any intermediate saves of results to files unnecessary and makes it much easier to test if your pipeline
makes sense before you do any computations.

Open-source
-----------
Our intention is to bring camera-based observations to a wide number of users anywhere in the world. Therefore the
code is entirely open-source. You can therefore:

* always access our latest code and use it within python
* make your own application around *pyorc*. Think of a geospatial front end, a time series database management system
  or other
* we can introduce the latest science in our code. As we continuously develop, new methods will appear in our code which
  you can immediately use. No license costs!

Organisation of API
===================

The API has a number of specific subclass models, which can be accessed as a method to outputs of different steps in the
process. Outputs from steps in *pyorc* are typically a ``xarray.DataArray`` for sets of frames or an ``xarray.Dataset``
for velocimetry results and transects. If you wish to access functionalities that can be applied on frames this is done
by calling the subclass
For instance, if you open a video, and extract frames from it, any functionality that can be applied to a frames set is
accessed as follows (with an example for normalization of frames to reduce background noise):

.. code::

   # open video, see Video section for information
   video = pyorc.Video("river.mp4", end_frame=100, camera_config=cam_config)
   # access grayscale frames
   da_frames = video.get_frames()
   # apply a specific frames functionality, here we access the normalization functionality
   da_norm_frames = da_frames.frames.normalize()

You can see that ``da_frames.frames`` directs to the ``.frames`` functionality of *pyorc*. You will see similar
principles for velocimetry and transect results throughout this manual. We wish you a lot of success with the use of
*pyorc*.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Table of Content

    Videos <video>
    Camera configurations <camera_config>
    Frames <frames>
    Velocimetry <velocimetry>
    Transects <transect>
