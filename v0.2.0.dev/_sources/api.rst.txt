.. currentmodule:: pyorc

.. _api:

=============
API reference
=============

``pyorc``'s consists of several subclasses of the ``xarray.Dataset`` and ``xarray.DataArray`` data models.
In a nutshell, ``xarray``'s data models are meant to store and analyze scientific datasets with multiple 
dimensions. A ``xarray.DataArray`` contains one variable with possibly several dimensions and coordinates
within those dimensions. A ``xarray.Dataset`` may contain multiple ``xarray.DataArray`` objects, with shared 
coordinates. In ``pyorc`` typically the coordinates are ``time`` for time epochs measured in seconds since
the beginning of a video, ``x`` for horizontal grid spacing (in meters), ``y`` for vertical grid spacing 
(in meters). Operations you can apply on both data models are very comparable with operations you may
already use in ``pandas``, such as resampling, reindexing, aggregations, and so on. 


.. note::
    We highly recommend to use the excellent ``xarray`` manual side-by-side with ``pyorc`` to understand 
    how to effectively work with the ``xarray`` ecosystem. 
    The documentation can be found on https://docs.xarray.dev/.

In the remaining sections, we describe the API classes, and the functions they are based on.

Video class
-----------


.. autosummary::
    :toctree: _generated
    
    Video.camera_config
    Video.fps
    Video.end_frame
    Video.start_frame
    Video.corners
    Video.get_frame
    Video.get_frames
    

Frames subclass
---------------

These methods can be called from an ``xarray.DataArray`` that is generated using ``pyorc.Video.get_frames``.

.. autosummary::
    :toctree: _generated

    Frames.normalize
    Frames.edge_detect
    Frames.project
    Frames.get_piv
    Frames.to_ani
    Frames.plot


Velocimetry subclass
--------------------


These methods can be called from an ``xarray.Dataset`` that is generated using ``pyorc.Frames.get_piv``.

.. autosummary::
    :toctree: _generated

    Velocimetry
    Velocimetry.filter_temporal
    Velocimetry.filter_temporal_angle
    Velocimetry.filter_temporal_neighbour
    Velocimetry.filter_temporal_std
    Velocimetry.filter_temporal_velocity
    Velocimetry.filter_temporal_corr
    Velocimetry.filter_spatial
    Velocimetry.filter_spatial_nan
    Velocimetry.filter_spatial_median
    Velocimetry.replace_outliers
    Velocimetry.get_transect
    Velocimetry.set_encoding
    Velocimetry.plot._pcolormesh

Transect subclass
-----------------

These methods can be called from an ``xarray.Dataset`` that is generated using ``pyorc.Velocimetry.get_transect``.

.. automodule:: pyorc.Transect
    :members: __init__, vector_to_scalar, get_xyz_perspective, get_river_flow, get_q, plot
    :imported-members:
    :undoc-members:
    :show-inheritance:
