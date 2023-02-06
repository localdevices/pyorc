.. _manual:

==========
User Guide
==========

.. note::

   This manual is a work in progress.

*pyorc* is a python Application Programming Interface (API) and command-line interface that allows you to process a video of a river or stream into surface
velocities and river flow estimates. The entire process, including stabilization, preprocessing, velocity estimation,
filtering of spurious velocities, deriving transect velocities and river flow estimation is supported. This user guide
provides a full understanding of the approach to analyze videos into surface velocities, transects and river flow
estimates. It is intended for both command-line users, and programmers that like to use the API. The command-line interface
can:

* prepare a camera configuration using a number of smart command-line arguments, and, where needed, a number of visual
  aided point and click windows. A camera configuration is needed to understand the perspective and dimensions of the
  video's field of view.
* analyze one to many videos of a specific field of view, using the aforementioned camera configuration and estimated or
  monitored water level, and produce 2-dimensional surface velocity estimates, surface velocities over a transect
  (sampled from the 2D fields) and river discharge.

Why is *pyorc* attractive to use?

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

This user guide
===============
As this user guide is intended for both command-line users and API users, you will sometimes see that two tabs are
available to select from. You may use the Command-line route, which explains the functionalities of the command-line,
or the API route, which details classes, methods and examples available within the API. In this way, as a user, you are
not bothered by details that you are not interested in. Try it out below!

.. tab-set::

    .. tab-item:: Command-line

        In this tab, you will be able to read and find examples of the usage of the command-line interface

    .. tab-item:: API

        In this tab, you will be able to read and find examples of the usage of the Application Programming Interface (API).

From here onwards, we will dive into the details. You will find out how to establish a "camera configuration", how
to enhance patterns and details that our algorithms may be tracing, how to reproject your field of view into a
orthorectified projection, that even understands geographical projections, how to estimate velocities, and mask out
poorly defined velocities, and how to extract analyzed end products such as velocities of a transect or integrated
river flow over a transect.

Organisation pyorc
==================

.. tab-set::

    .. tab-item:: Command-line

        The command line interface consists of so-called subcommands. You can see the available subcommands by passing
        the ``--help`` option.

        .. code-block:: cmd

            $ pyorc --help

        .. program-output:: pyorc --help

        The meaning and way to use of these subcommands are explained through the rest of this user guide.

    .. tab-item:: API

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
    Camera configurations <camera_config/index>
    Frames <frames/frames.ipynb>
    Velocimetry <velocimetry>
    Transects <transect>
