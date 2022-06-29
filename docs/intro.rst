.. _intro:

============
Introduction
============

.. _nutshell:

pyOpenRiverCam in a nutshell
============================

pyOpenRiverCam is a Application Programming Interface (API) to preprocess, reproject, and analyze videos of rivers, in order
to estimate river flows. Below we provide an overview of all functionalities in a nutshell:

.. tip::

    .. raw:: html

        <div>
            For a fast idea of how <b>pyorc</b> works, try our interactive examples immediately in our binder link.
            <a href="https://mybinder.org/v2/gh/localdevices/pyorc.git/main?labpath=examples{{ docname|e }}" target="_blank" rel="noopener noreferrer"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg"></a>
        </div>



+----------------------------------+-----------------------------------------------------------------------------------+
| Feature                          | Example                                                                           |
+==================================+===================================================================================+
| Create geographical awareness    | .. code::                                                                         |
| of your videos using your own    |                                                                                   |
| field observations               |     import pyorc                                                                  |
|                                  |     # create a camera configuration with ground control points and CRS            |
|                                  |     cam_config = pyorc.CameraConfig(                                              |
|                                  |         gcps=gcps, crs=32735, lens_position=[642732.6705, 8304289.010, 1188.5]    |
|                                  |     )                                                                             |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/wark_cam_config.jpg                                                                               |
+----------------------------------+-----------------------------------------------------------------------------------+
| Extract frames in an array-like  | .. code::                                                                         |
| manageable format using the      |                                                                                   |
| **xarray** library.              |     # open video with geographical awareness using a camera config                |
|                                  |     video = pyorc.Video(                                                          |
|                                  |         video_file, camera_config=cam_config, start_frame=0, end_frame=150        |
|                                  |     )                                                                             |
|                                  |     # retrieve the frames in a xr.DataArray object                                |
|                                  |     da = video.get_frames(method="rgb")                                           |
|                                  |     # plot the first frame (animation also possible with .to_ani)                 |
|                                  |     da[0].frames.plot()                                                           |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/video_orig.gif                                                                                    |
+----------------------------------+-----------------------------------------------------------------------------------+
| Enhance frames to improve        | .. code::                                                                         |
| visibility of tracers            |                                                                                   |
|                                  |     # reduce background noise by subtracting a multi-frame average                |
|                                  |     da_norm = da.frames.normalize()                                               |
|                                  |     # plot the first frame (animation also possible with .to_ani)                 |
|                                  |     da_norm[0].frames.plot(cmap="gray")                                           |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/video_norm.gif                                                                                    |
+----------------------------------+-----------------------------------------------------------------------------------+
| Reproject frames to metres-      | .. code::                                                                         |
| distance planar views            |                                                                                   |
|                                  |     # project                                                                     |
|                                  |     da_norm_proj = da_norm.frames.project()                                       |
|                                  |     # plot the first projected frame (animation also possible with .to_ani)       |
|                                  |     da_norm_proj[0].frames.plot(cmap="gray")                                      |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/video_norm_proj.gif                                                                               |
+----------------------------------+-----------------------------------------------------------------------------------+
| Enhance gradients for improved   | .. code::                                                                         |
| feature detection                |                                                                                   |
|                                  |     # gradient enhancement                                                        |
|                                  |     da_edge = da_norm_proj.frames.edge_detect()                                   |
|                                  |     # plot the first gradient frame (animation also possible with .to_ani)        |
|                                  |     da_edge[0].frames.plot(cmap="RdBu", vmin=-6, vmax=6)                          |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/video_edge.gif                                                                                    |
+----------------------------------+-----------------------------------------------------------------------------------+
| Estimate flow velocity at the    | .. code::                                                                         |
| water surface using Particle     |                                                                                   |
| Image Velocimetry.               |     # surface velocimetry with PIV                                                |
| Plot with ``matplotlib``         |     ds_piv = da_edge.frames.get_piv()                                             |
| convenience                      |     # filter spurious velocities with several filters                             |
|                                  |     ds_piv = ds_piv.velocimetry.filter_temporal().velocimetry.filter_spatial()    |
|                                  |     # plot the median                                                             |
|                                  |     ds_median = ds_piv.median(dim="time", keep_attrs=True)                        |
|                                  |     # combine plots in one axes, including a rgb first frame of video             |
|                                  |     norm = Normalize(vmin=0., vmax=0.5, clip=False)                               |
|                                  |     p = video.get_frames(method="rgb")[0].frames.plot()                           |
|                                  |     p2 = ds_median.velocimetry.plot.streamplot(                                   |
|                                  |         ax=p.axes,                                                                |
|                                  |         alpha=0.9,                                                                |
|                                  |         cmap="rainbow",                                                           |
|                                  |         density=3.,                                                               |
|                                  |         minlength=0.05,                                                           |
|                                  |         norm=norm,                                                                |
|                                  |         linewidth_scale=4,                                                        |
|                                  |         add_colorbar=True                                                         |
|                                  |     )                                                                             |
|                                  |     p.axes.set_aspect("equal")                                                    |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/wark_streamplot.jpg                                                                               |
+----------------------------------+-----------------------------------------------------------------------------------+
| Estimate river discharge over    | .. code::                                                                         |
| a supplied cross-section.        |                                                                                   |
| Use smart functions to fill      |     # x, y, z are coordinates and elevation of cross-section, get a transect      |
| missing data. Plot with          |     ds_points = ds_piv.velocimetry.get_transect(x, y, z, crs=4326)                |
| ``matplotlib`` convenience.      |     # retrieve depth integrated velocity (m2/s) and fill missing with log profile |
|                                  |     ds_points_q = ds_points.transect.get_q(fill_method="log_profile")             |
|                                  |     # combine plots in one axes, including a rgb first frame of video             |
|                                  |     p = video.get_frames(method="rgb")[0].frames.plot()                           |
|                                  |     p2 = ds_median.velocimetry.plot.pcolormesh(                                   |
|                                  |         ax=p.axes,                                                                |
|                                  |         alpha=0.3,                                                                |
|                                  |         cmap="rainbow",                                                           |
|                                  |         norm=norm,                                                                |
|                                  |     )                                                                             |
|                                  |     p.axes.set_aspect("equal")                                                    |
|                                  |     ds_points_q.isel(quantile=2).transect.plot(                                   |
|                                  |         ax=p.axes,                                                                |
|                                  |         cmap="rainbow",                                                           |
|                                  |         norm=norm,                                                                |
|                                  |         scale=5,                                                                  |
|                                  |         width=0.003,                                                              |
|                                  |         add_colorbar=True                                                         |
|                                  |     )                                                                             |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/wark_discharge.jpg                                                                                |
+----------------------------------+-----------------------------------------------------------------------------------+
| Plot combined views in immersive | .. code::                                                                         |
| local, geographical or camera    |                                                                                   |
| perspectives, using              |     # plotting in augmented reality using the camera perspective                  |
| ``matplotlib`` convenience       |     p = video.get_frames(method="rgb")[0].frames.plot(mode="camera")              |
| methods.                         |     # add the surface velocimetry in camera perspective                           |
|                                  |     p1 = ds_median.velocimetry.plot(                                              |
|                                  |         ax=p.axes,                                                                |
|                                  |         mode="camera",                                                            |
|                                  |         alpha=0.5,                                                                |
|                                  |         cmap="rainbow",                                                           |
|                                  |         norm=norm,                                                                |
|                                  |         add_colorbar=True                                                         |
|                                  |     )                                                                             |
|                                  |     p.axes.set_aspect("equal")                                                    |
+----------------------------------+-----------------------------------------------------------------------------------+
| .. image:: _images/wark_cam_persp.jpg                                                                                |
+----------------------------------+-----------------------------------------------------------------------------------+

