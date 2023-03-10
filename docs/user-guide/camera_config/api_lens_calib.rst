.. _camera_config_api_lc:

Once you have your video, the camera calibration is very simple. After creating your camera configuration you can
supply the video in the following manner:

.. code-block:: python

    calib_video = "calib_video.mp4"
    cam_config.set_lens_calibration(calib_video)

When you execute this code, the video will be scanned for suitable images, and will select frames that are relatively
far apart from each other. When a suitable image with patterns is found, the algorithm will show the image and the found
chessboard pattern. There are several options you may supply to the algorithm to influence the amount of internal corner
points of the chessboard (default is 9x6), the maximum frames number that should be used for calibration,
filtering of poorly performing images, switch plotting and writing plots to files (for later checking of the results)
on or off.

.. note::
   the camera calibration is still experimental. If you have comments or issues kindly let us know by making a github
   issue.

