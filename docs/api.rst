.. currentmodule:: pyorc

.. _api:

=============
API reference
=============

**pyorc**'s API consists of several subclasses of the ``xarray.Dataset`` and ``xarray.DataArray`` data models.
In a nutshell, xarray_'s data models are meant to store and analyze scientific datasets with multiple
dimensions. A ``xarray.DataArray`` contains one variable with possibly several dimensions and coordinates
within those dimensions. A ``xarray.Dataset`` may contain multiple ``xarray.DataArray`` objects, with shared 
coordinates. In **pyorc** typically the coordinates are ``time`` for time epochs measured in seconds since
the beginning of a video, ``x`` for horizontal grid spacing (in meters), ``y`` for vertical grid spacing 
(in meters). Operations you can apply on both data models are very comparable with operations you may
already use in pandas_, such as resampling, reindexing, aggregations, and so on.


.. note::
    We highly recommend to use the excellent xarray_ manual side-by-side with ``pyorc`` to understand
    how to effectively work with the xarray_ ecosystem.

In the remaining sections, we describe the API classes, and the functions they are based on.

.. _cameraconfig:

CameraConfig class
==================

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    CameraConfig
    CameraConfig.bbox
    CameraConfig.shape
    CameraConfig.transform

Setting of properties and attributes
------------------------------------

.. autosummary::
    :toctree: _generated

    CameraConfig.set_bbox_from_corners
    CameraConfig.set_gcps
    CameraConfig.set_lens_pars
    CameraConfig.set_lens_calibration
    CameraConfig.set_lens_position

Exporting
---------

.. autosummary::
    :toctree: _generated

    CameraConfig.to_dict
    CameraConfig.to_file
    CameraConfig.to_json


Retrieve geometrical information
--------------------------------

.. autosummary::
    :toctree: _generated

    CameraConfig.get_M
    CameraConfig.get_depth
    CameraConfig.z_to_h

Plotting methods
----------------

.. autosummary::
    :toctree: _generated

    CameraConfig.plot
    CameraConfig.plot_bbox

.. _video:

Video class
===========

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    Video
    Video.camera_config
    Video.fps
    Video.end_frame
    Video.start_frame
    Video.corners

Getting frames from video objects
---------------------------------

.. autosummary::
    :toctree: _generated

    Video.get_frame
    Video.get_frames

.. _frames:

Frames subclass
===============

These methods can be called from an ``xarray.DataArray`` that is generated using ``pyorc.Video.get_frames``.

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    Frames
    Frames.camera_config
    Frames.camera_shape
    Frames.h_a

Enhancing frames
----------------

.. autosummary::
    :toctree: _generated

    Frames.edge_detect
    Frames.normalize
    Frames.time_diff
    Frames.smooth

Projecting frames to planar views
---------------------------------
.. autosummary::
    :toctree: _generated

    Frames.project

Retrieving surface velocities from frames
-----------------------------------------

.. autosummary::
    :toctree: _generated

    Frames.get_piv

Visualizing frames
------------------

.. autosummary::
    :toctree: _generated

    Frames.plot
    Frames.to_ani
    Frames.to_video

.. _velocimetry:

Velocimetry subclass
====================

These methods can be called from an ``xarray.Dataset`` that is generated using ``pyorc.Frames.get_piv``.

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    Velocimetry
    Velocimetry.camera_config
    Velocimetry.camera_shape
    Velocimetry.h_a

.. _masks:

Temporal masking methods
------------------------

The mask methods below either require or may have a dimension ``time`` in the data. Therefore they are most logically
applied before doing any reducing over time.

.. autosummary::
    :toctree: _generated
    :template: accessor_method.rst

    Velocimetry.mask.minmax
    Velocimetry.mask.corr
    Velocimetry.mask.angle
    Velocimetry.mask.rolling
    Velocimetry.mask.outliers
    Velocimetry.mask.variance
    Velocimetry.mask.count

.. _spatial_mask:

Spatial masking methods
-----------------------

The spatial masking methods look at a time reduced representation of the grid results. The resulting mask can be applied
on a full time series and will then mask out grid cells over its full time span if these do not pass the mask.

.. autosummary::
    :toctree: _generated
    :template: accessor_method.rst

    Velocimetry.mask.window_mean
    Velocimetry.mask.window_nan

Data infilling
--------------

.. autosummary::
    :toctree: _generated
    :template: accessor_method.rst

    Velocimetry.mask.window_replace

.. _transects:

Getting data over transects
---------------------------

.. autosummary::
    :toctree: _generated

    Velocimetry.get_transect
    Velocimetry.set_encoding

Plotting methods
----------------

.. autosummary::
    :toctree: _generated
    :template: accessor_method.rst


    Velocimetry.plot
    Velocimetry.plot.pcolormesh
    Velocimetry.plot.scatter
    Velocimetry.plot.streamplot
    Velocimetry.plot.quiver

.. _transect:

Transect subclass
=================

These methods can be called from an ``xarray.Dataset`` that is generated using ``pyorc.Velocimetry.get_transect``.

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    Transect
    Transect.camera_config
    Transect.camera_shape
    Transect.h_a

Derivatives
-----------

.. autosummary::
    :toctree: _generated

    Transect.vector_to_scalar
    Transect.get_xyz_perspective

.. _river_flow:

River flow methods
------------------

.. autosummary::
    :toctree: _generated

    Transect.get_river_flow
    Transect.get_q

Plotting methods
----------------

.. autosummary::
    :toctree: _generated
    :template: accessor_method.rst

    Transect.plot
    Transect.plot.quiver
    Transect.plot.scatter

.. _xarray: https://docs.xarray.dev/
.. _pandas: https://pandas.pydata.org/
