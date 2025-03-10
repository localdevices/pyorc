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
    CameraConfig.set_intrinsic
    CameraConfig.set_lens_calibration
    CameraConfig.set_lens_position

Exporting
---------

.. autosummary::
    :toctree: _generated

    CameraConfig.to_dict
    CameraConfig.to_dict_str
    CameraConfig.to_file
    CameraConfig.to_json


Retrieve geometrical information
--------------------------------

.. autosummary::
    :toctree: _generated

    CameraConfig.estimate_lens_position
    CameraConfig.get_M
    CameraConfig.get_bbox
    CameraConfig.get_dist_shore
    CameraConfig.get_dist_wall
    CameraConfig.get_depth
    CameraConfig.get_z_a
    CameraConfig.project_grid
    CameraConfig.project_points
    CameraConfig.unproject_points
    CameraConfig.z_to_h
    CameraConfig.h_to_z
    CameraConfig.estimate_lens_position

Plotting methods
----------------

.. autosummary::
    :toctree: _generated

    CameraConfig.plot
    CameraConfig.plot_bbox

.. _cross_section:

CrossSection class
==================

The `CrossSection` class is made to facilitate geometrical operations with cross sections.
It can be used to guide water line detection methods, estimate the water level and
possibly other optical methods conceived by the user, that require understanding of the
perspective, jointly with a measured cross section.

To facilitate geometric operations, most of the geometric properties are returned as
`shapely.geometry` objects such as `Point`, `LineString` and `Polygon`.

This makes it easy to derive important flow properties such as wetted surface [m2] and
wetted perimeter [m].

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    CrossSection
    CrossSection.cs_points
    CrossSection.cs_points_sz
    CrossSection.cs_linestring
    CrossSection.cs_linestring_sz
    CrossSection.cs_angle
    CrossSection.idx_closest_point
    CrossSection.idx_farthest_point

Getting cross section geometries
--------------------------------

.. autosummary::
    :toctree: _generated

    CrossSection.get_cs_waterlevel
    CrossSection.get_csl_point
    CrossSection.get_csl_line
    CrossSection.get_csl_pol
    CrossSection.get_bottom_surface
    CrossSection.get_planar_surface
    CrossSection.get_wetted_surface
    CrossSection.get_wetted_surface_sz
    CrossSection.get_line_of_interest

Plotting methods
----------------

The plotting methods consist of a number of smaller methods, as well as one
overarching `CrossSection.plot` method, that combines the smaller methods.
The plotting functions are meant to provide the user insight in the situation
on a site. All methods can be combined with the parallel plotting method
for camera configurations `CameraConfig.plot`.

.. autosummary::
    :toctree: _generated

    CrossSection.plot
    CrossSection.plot_cs
    CrossSection.plot_planar_surface
    CrossSection.plot_bottom_surface
    CrossSection.plot_wetted_surface

Water level detection
---------------------
Combined with a preprocessed image from e.g. a video file, a water level can be detected.

.. autosummary::
    :toctree: _generated

    CrossSection.detect_water_level

.. _video:

Video class
===========

Class and properties
--------------------

.. autosummary::
    :toctree: _generated

    Video
    Video.camera_config
    Video.end_frame
    Video.fps
    Video.frames
    Video.freq
    Video.h_a
    Video.lazy
    Video.mask
    Video.rotation
    Video.stabilize
    Video.start_frame
    Video.corners

Setting properties
------------------

.. autosummary::
    :toctree: _generated

    Video.set_mask_from_exterior


Getting frames from video objects
---------------------------------

.. autosummary::
    :toctree: _generated

    Video.get_frame
    Video.get_frames
    Video.get_frames_chunk
    Video.get_ms

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
    Frames.minmax
    Frames.normalize
    Frames.reduce_rolling
    Frames.smooth
    Frames.time_diff

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
    Velocimetry.is_velocimetry

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
    Transect.get_depth_perspective
    Transect.get_bottom_surface_z_perspective
    Transect.get_transect_perspective
    Transect.get_wetted_perspective

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
