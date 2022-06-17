.. _intro:

============
Introduction
============

.. _nutshell:

pyOpenRiverCam in a nutshell
============================

pyOpenRiverCam is a Application Programming Interface (API) to preprocess, reproject, and analyze videos of rivers, in order
to estimate river flows. Below we provide an overview of all functionalities in a nutshell:

+-----------------------+----------------------------------------------------+
| Feature               | Example                                            |
+=======================+====================================================+
| Create geographical   | .. code::                                           |
| awareness of your     |                                                    |
| videos using your     |     import pyorc                                   |
|                       |     # create a camera configuration with ground control points and CRS
| own field             |     cam_config = pyorc.CameraConfig(gcps=gcps, crs=32735, lens_position=[642732.6705, 8304289.010, 1188.5])                                               |
| observations          |                                                    |
+-----------------------+----------------------------------------------------+
| .. image:: _images/wark_cam_config.jpg                                     |
+-----------------------+----------------------------------------------------+


+-----------------------+----------------------------------------------------+
| Feature               | Example                                            |
+=======================+====================================================+
| | Create geographical | .. image:: _images/wark_cam_config.jpg             |
| | awareness of your   |                                                    |
| | videos using your   |                                                    |
| | own field           |                                                    |
| | observations        |                                                    |
+-----------------------+----------------------------------------------------+
| | Extract frames      | .. image:: _images/video_orig.gif                  |
| | your original video |                                                    |
| | into a array-like   |                                                    |
| | manageable format   |                                                    |
| | using the xarray    |                                                    |
| | library.            |                                                    |
+-----------------------+----------------------------------------------------+
| | Enhance frames to   | .. image:: _images/video_norm.gif                  |
| | improve visibility  |                                                    |
| | of tracers          |                                                    |
+-----------------------+----------------------------------------------------+
| | Reproject frames    | .. image:: _images/video_norm_proj.gif             |
| | to meters-distance  |                                                    |
| | planar views        |                                                    |
+-----------------------+----------------------------------------------------+
| | Enhance gradients   | .. image:: _images/video_edge.gif                  |
| | for improved feature|                                                    |
| | detection           |                                                    |
+-----------------------+----------------------------------------------------+
| | Estimate flow       | .. image:: _images/wark_streamplot.jpg             |
| | velocity at the     |                                                    |
| | water surface       |                                                    |
+-----------------------+----------------------------------------------------+
| | Estimate river      | .. image:: _images/wark_discharge.jpg              |
| | discharge over a    |                                                    |
| | supplied cross-     |                                                    |
| | section             |                                                    |
+-----------------------+----------------------------------------------------+
| | Export preprocessed | | Above examples were generated with this export   |
| | frames to animation | | functionality                                    |
+-----------------------+----------------------------------------------------+
| | Plot combined       | .. image:: _images/wark_cam_persp.jpg              |
| | views using         |                                                    |
| | ``matplotlib``      |                                                    |
| | convenience methods |                                                    |
| | in local, camera    |                                                    |
| | or geographical     |                                                    |
| | perspectives        |                                                    |
+-----------------------+----------------------------------------------------+
| | Perform scientific  | .. code::                                          |
| | analysis and store  |                                                    |
| | results to files    |     # perform piv analysis on treated frames       |
| | with ``xarray`` data|     piv = preprocessed.frames.get_piv()            |
| | models              |     # filter piv results with pyorc filters        |
|                       |     piv_filt = piv.velocimetry.filter_temporal()   |
|                       |     # get mean over time using xarray funcs        |
|                       |     piv_mean = piv_filt.mean(                      |
|                       |         dim="time", keep_attrs=True                |
|                       |     )                                              |
|                       |     # store results to netcdf using xarray         |
|                       |     piv._mean.to_netcdf("my_first_piv.nc")         |
+-----------------------+----------------------------------------------------+

.. note:: Documentation is currently being established. Please come back later for more information.

