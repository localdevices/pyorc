.. _manual:

==========
User Guide
==========

*pyorc* processes a video, taken with a smartphone, drone or other camera into surface velocities and river flow
estimates. It can do so either through a Command Line Interface or an Application Programming Interface (API).
The entire process, including stabilization, preprocessing, velocity estimation,
filtering of spurious velocities, deriving transect velocities and river flow estimation is supported. This user guide
aims to provide a full understanding of the approach to analyze videos into surface velocities, transects and river flow
estimates. It is intended for both command-line users, and programmers that like to use the API.
The command-line interface can:

* prepare a camera configuration using a number of smart command-line arguments, and, where needed, a number of visual
  aided point and click windows. A camera configuration is needed to understand the perspective and dimensions of the
  video's field of view.
* analyze one to many videos of a specific field of view, using the aforementioned camera configuration and estimated or
  monitored water level, and produce 2-dimensional surface velocity estimates, surface velocities over a transect
  (sampled from the 2D fields) and river discharge.

The API basically does the same but provides access to all underlying methods through e.g. scripting or through your
own wrapped functionality around it. If you wish to develop your own application around video-based velocity estimation
or river flow estimations, then *pyorc* is the way to go. Think for instance of the following possibilities:

- A Graphical User Interface for estimating river flows from videos
- A data assimilation framework for hydraulic simulations and calibration of surface velocities
- A user-oriented dashboard for doing comparative analyses for pre and post-interventions, erosion sensitivies,
  suitability studies, you name it.

Why is *pyorc* attractive to use?

Solid data model
----------------
We rely on *xarray* for data models. *xarray* provides access and intelligent dimensional computations over
multi-dimensional datasets with coordinate systems that can relate to time, space or other logical dimensions. If you
only use the command-line interface, then you don't have to wrory about this at all, it just means that you can expect
*pyorc* to be easy to maintain for us, easy to extend to exports to other formats you may desire, and therefore used
for a long and sustainable period of time.

Lazy computations
-----------------
The main idea of *pyorc* is that all computations are performed lazily, meaning that as a programmer (again, are you
using the command-line only, then just don't worry about this at all), you can first setup and investigate the entire
pipeline of computations, and only at the moment when you request actual numbers from the pipeline, will *pyorc*
perform computations. And then only those strictly needed to provide you the number(s) you want.
This makes any intermediate saves of results to files unnecessary and makes it much easier to test if your pipeline
makes sense before you do any computations.

Open-source
-----------
Our intention is to bring camera-based observations to a wide number of users anywhere in the world. Therefore the
code is entirely open-source. You can therefore:

* always access our latest code and use it within python
* make your own application around *pyorc*. Think of a geospatial front end, a time series database management system
  or other app.
* we can introduce the latest science in our code. As we continuously develop, new methods will appear in our code which
  you can immediately use. No license costs and full access to the code always!

This user guide
---------------

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

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Table of Content

    Command-line Interface <cli>
    Application Programming Interface <api>
    Camera configurations <camera_config/index>
    Videos <video/index>
    Frames <frames/index>
    Velocimetry <velocimetry/index>
    Transects <transect/index>
    Plotting <plot/index>
