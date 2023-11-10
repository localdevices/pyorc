.. _frames_ug:

Frames
======

The frames of a video can be enhanced before projection to highlight particles and patterns more clearly.
For instance, intensities can be thresholded, edges can be brought forward, and background transparency can be reduced.
The enhancements are automatically applied on all frames from the video or from the user defined frames between start and
end frame. After each enhancement, a new operation can be performed on the result, and even several enhancements can be
repeated. For instance, you may first apply an intensity thresholding to remove intensities above or below a certain
threshold (e.g. solar flare), then apply an edge detection, and then again threshold the result of the edge detection
enhancement.

The methods to perform these image enhancements are described below.

Intensity thresholding
----------------------
To remove effects of noisy dark pixels, or very bright pixels, such as directly reflected sunlight, a simple thresholding
method can be applied.


.. tab-set::

    .. tab-item:: Command-line

        the ``minmax`` flag performs thresholding on the frames. Simply set a minimum and maximum value under ``min``
        and ``max`` to darken pixels that are below or above the set thresholds. An example is provided below.

        .. code-block:: yaml

            frames:
                minmax:
                    min: 150
                    max: 250

    .. tab-item:: API

        Thresholding is performed with the ``minmax`` method, described in the :ref:`frames API section <frames>`:

.. _contrast:

Improving contrast
------------------
There are a number of ways to improve contrast of moving features. These so far include the following methods:

* Normalization: removes the average of a number of frames in time. This then yields a better contrast of moving things
  compared to static things. This is particularly useful to remove visuals of the bottom, when you have very transparent
  water. You can set the amount of frames used for averaging.
* Smoothing: perform a gaussian kernel average over each frame. This removes very small moise features that may be seen
  as moving patterns. It is important to keep the smoothing kernel smaller in size than the typical size of a moving
  feature.
* Differencing over time: this method is extremely effective for distinguishing moving things on the water. It simply
  subtracts frame 1 from frame 2, frame 2 from frame 3 etcetera. Things that are very much the same between subsequent
  frames will disappear. This method only works well when your video is very stable, e.g. taken with a fixed rig.
  In fact, we have seen deteriorating results when applying this filter on highly unstable videos. It is highly
  recommended to apply this method in combination with smoothing when you have a fixed rig setup or are able to keep
  your camera very stable, for instance with a tripod.
* Edge detection: enhances the visibility of strong spatial gradients in the frame, by applying two kernel filters on
  the individual frames with varying window sizes, and returning the difference between the filtered images.

.. tab-set::

    .. tab-item:: Command-line

        The ``normalize`` flag can be used to remove background. By default, 15 frames at equal intervals are extracted
        and averaged to establish an estimate of the background intensities. This is then subtracted from each frame to
        establish a background corrected series. The amount of frames can be controlled with the ``samples`` flag. An
        example, where the default amount of samples is modified to 25 is provided below.

        .. code-block:: yaml

            frames:
                normalize:
                    samples: 25

    .. tab-item:: API

        Normalization is controlled by the ``normalize`` method and described in the :ref:`frames API section <frames>`.


.. tab-set::

    .. tab-item:: Command-line

        The ``time_diff`` method computes the difference between subsequent frames. It is highly recommended to do
        some smoothing before this step to ensure that very small edges are smudged out. It is possible to take the
        absolute value after smoothing (add ``abs: True``) or apply a threshold on pixels that have a very low absolute
        difference in intensity. We have best experience with the default threshold of 2 and no absolute value applied.
        This can be changed as shown below. Simply leave out the line ``thres: 5`` to use defaults:

        .. code-block:: yaml

            frames:
                time_diff:
                    thres: 5

    .. tab-item:: API

        Time differencing is performed with the ``time_diff`` method, described in the :ref:`frames API section <frames>`.


.. tab-set::

    .. tab-item:: Command-line

        The ``smooth`` method smooths each frame a small or large (user-defined) amount. The amount can be controlled
        with the ``wdw`` optional keyword. The default value is 1, which means the smallest kernel possible (3x3) is
        used. When you set it to 2, 3, 4, etc. the kernel size will go to 5x5, 7x7, 9x9, and so on. This filter
        is very efficient when performed *before* the ``time_diff`` filter when you have very stable videos, for
        instance from a fixed rig setup.

        .. code-block:: yaml

            frames:
                smooth:
                    wdw: 2

    .. tab-item:: API

        Smoothing is performed with the ``smooth`` method, described in the :ref:`frames API section <frames>`.


.. tab-set::

    .. tab-item:: Command-line

        The ``edge_detect`` method computes the difference between kernel smoothed frames.
        By default, the kernel window one-sided sizes are 1 (i.e. a 3x3 window) and 2 (a 5x5 window) respectively.
        For very high resolution imagery or very large features to track, these may be enlarged to better encompass the
        edges of the patterns of interest. The kernel sizes can be modified using the flags ``wdw_1`` and ``wdw_2`` for
        instance as follows:

        .. code-block:: yaml

            frames:
                edge_detect:
                    wdw_1: 2
                    wdw_4: 4

    .. tab-item:: API

        Edge detection is performed with the ``edge_detect`` method, described in the :ref:`frames API section <frames>`.

Orthoprojection
---------------
As you supply a camera configuration to the video, *pyorc* is aware how frames must be reprojected to provide an
orthorectified image. Typically orthorectification is the last step before estimating surface velocities from the
frame pairs. Currently we have two approaches for orthoprojection. The first method ``method="cv"`` uses solely OpenCV
functionality. Our new method ``method="numpy"`` relies on numpy and xarray functionalities and is likely to provide
much more stable results in cases where the camera lens has a strong distortion and/or a part of the area of interest
lies outside of the field of view. In these cases we strongly recommend using ``method="numpy"``.

.. tab-set::

    .. tab-item:: Command-line

        In the command-line interface, orthoprojection is performed automatically after all image enhancement steps
        the a user may possible have entered in the recipe. Nonetheless, you can still control the resolution (in meters)
        of the projected end result at this stage. If you leave any specifics about the projection out of your recipe,
        then *pyorc* will assume that you want to use the resolution specified in the camera configuration file. If
        however you wish to manipulate the resolution in the recipe then you can do this by using the following
        keys and values (with an example for 0.1 meter resolution). We here also show how to change the resampling
        method to numpy (instead of the default cv):

        .. code-block:: yaml

            frames:
                ...
                ...
                ...
                project:
                    resolution: 0.1
                    method: numpy

    .. tab-item:: API

        Projection is performed with the ``project`` method, described in the :ref:`frames API section <frames>`.

.. note:: we will likely change our default projection method to numpy in a future release of *pyorc*.

Exporting results to video
--------------------------
If you wish to inspect frames after they have been treated with filters and projected, then you can write the result to
a new video file. This helps to assess if patterns are indeed clearly visible and projected results good enough in
resolution to recognize the features on the water surface.

.. tab-set::

    .. tab-item:: Command-line

        In the recipe, the export to a video can be controlled with the ``to_video`` method and by supplying a
        filename with extension .mp4 or another recognizable video extension. An example of a frames section in which
        enhanced frames (with normalization, edge detection and finally thresholding and projecting) are written to a
        file is given below.

        .. code-block:: yaml

            frames:
                normalize:
                    samples: 25
                edge_detect:
                    wdw_1: 2
                    wdw_4: 4
                minmax:
                    min: 0
                    max: 10

    .. tab-item:: API

        Exporting frames to a video is performed with the ``to_video`` method, described in the
        :ref:`frames API section <frames>`.
