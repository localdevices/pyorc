.. _camera_config_cli:

The command-line interface supports setting up a camera configuration through a subcommand. To see the options
simply call the subcommand on a command prompt with ``--help`` as follows:

.. code-block:: cmd

    $ pyorc camera-config --help

.. program-output:: pyorc camera-config --help

To setup a camera configuration, you will at minimum need to provide the following:

* A video, made with a specific camera, oriented to a fixed point of view, using known and fixed settings. With "fixed"
  we mean here that any additional video supplied to *pyorc* should be taken with the same settings and the same exact
  field of view.
* Ground control points. These are combinations of real-world coordinates (possibly in a geographical coordinate
  reference system) and column, row coordinates in the frames of the video. By assigning where in the world a column,
  row coordinate is, and do this for several locations, the field of view of the camera can be projected into a real-world
  view.
* 4 corner points that approximately indicate the bounding box of your area of interest. These must be provided in
  the order *upstream-left*, *downstream-left*, *downstream-right*, *upstream-right*, where left is the left-bank
  as seen while looking in downstream direction.

There are several ways to assign this information, further explained below.
