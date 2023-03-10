.. _api_ug:

Application Programming Interface
=================================

The API has a number of specific subclass models, which can be accessed as a method to outputs of different steps in the
process. Outputs from steps in *pyorc* are typically a ``xarray.DataArray`` for sets of frames or an ``xarray.Dataset``
for velocimetry results and transects. If you wish to access functionalities that can be applied on frames this is done
by calling the subclass
For instance, if you open a video, and extract frames from it, any functionality that can be applied to a frames set is
accessed as follows (with an example for normalization of frames to reduce background noise):

.. code:: python

   # open video, see Video section for information
   video = pyorc.Video("river.mp4", end_frame=100, camera_config=cam_config)
   # access grayscale frames
   da_frames = video.get_frames()
   # apply a specific frames functionality, here we access the normalization functionality
   da_norm_frames = da_frames.frames.normalize()

You can see that ``da_frames.frames`` directs to the ``.frames`` functionality of *pyorc*. You will see similar
principles for velocimetry and transect results throughout this manual.

If you are a programmer, then we highly recommend to look at the following pages:

* :ref:`Quick start for programmers <quickstart>` which contains full working examples to start programming
* :ref:`API reference <api>`