.. _cli_ug:

Command line interface
======================

The command line interface is meant to provide an easy, reproducible and streamlined approach to perform the two main
tasks required for performing a surface velocity and discharge analysis being:

* Camera configuration: establishing the relationship between the field of view of the camera and the real world
  coordinates and defining the bounding box of the area of interest within the field of view.
* Velocity and discharge processing: using the camera configuration with one or several (of different moments in time)
  videos of the same field of view (e.g. taken at different times) to estimate surface velocity on the area of interest
  and river flow over bathymetric transects.

The two tasks are available as so-called "subcommands". You can see the available subcommands by passing
the ``--help`` option.

.. code-block:: shell

    $ pyorc --help

.. program-output:: pyorc --help

The meaning and way to use of these subcommands are explained through the rest of this user guide. Currently the
available subcommands are ``camera-config`` and ``velocimetry`` for the two tasks mentioned.
To find out more about them, you can request specific help on them on the command-line as well.

.. code-block:: shell

    $ pyorc camera-config --help

.. program-output:: pyorc camera-config --help

The ``camera-config`` command is meant to create a camera configuration, stored in a file. The camera
configuration details how the camera's perspective relates to real-world coordinates and distances, which area
within the objective should be processed, what the characteristics of the used lens are. All details can be
found in the section :ref:`camera_config_ug`.

.. code-block:: shell

    $ pyorc velocimetry --help

.. program-output:: pyorc velocimetry --help

The ``velocimetry`` command does all the processing of videos into all end products you may want. The inputs
``-V`` (videofile), ``-c`` (camera configuration file) and ``-r`` (recipe file) are all required and in turn contain a
video file compatible with the provided camera configuration, and a recipe-file containing the recipe for the
processing steps. These steps may differ from location to location, or dependent on what details you wish to capture in
your video.

The processing steps that this command goes through are:

* Preprocessing frames to e.g. enhance or sharpen patterns on the water that may be traceable.
* Reproject the frames to a coordinate system (i.e. as if you are looking at the stream from above with known distances
  between pixels).
* estimate surface velocity between groups of pixels.
* mask out spurious velocities (can also be left out, if tracers are very prominent).
* extract velocities over one or more transects and estimate river flow (only if you are interested in this).
* make plots of frames, velocities and transects in the camera perspective, geographical perspective or local
  projection.

With the ``-u`` or ``--update`` flag, you can indicate that you only want to process those parts of your recipe that
you have modified. With this option *pyorc* will check if *any* of your inputs and outputs (including the recipe
components, the input files and output files) have changed or require reprocessing because they were (re)moved. If this
is not the case then that step will be skipped. This is ideal for instance in case you only changed a plot setup or
masking step. The velocimetry part (which is time consuming) will then be skipped entirely.

.. note::

    You can also supply an option ``--lowmem`` which may be useful if you work with a low memory device or very large.
    With this setting, only small portions of data are processed at the same time. This therefore increases processing time,
    while decreasing (sometimes severely!) the amount memory required. If *pyorc* crashes then this is likely due
    to low memory resources. Please then try this option

The recipe file provided with ``-r`` is a so-called Yaml formatted file (extension .yml). This is a text file with
a specific format that defines exactly what steps are being performed, and how the steps are to be performed. For
instance, the preprocessing steps can be done with different techniques, and these can be combined in substeps.
The same is true for the masking step. Several masking strategies can be performed, and this can even be done in parallel
or in series, to improve the results. The Yaml file is referred to as the *recipe* in the remainder of the User Guide.

.. note::

    A recipe file seems a lot of work to write, however, as you get used to *pyorc* you will notice that for many
    use cases, you can simply use exactly the same or almost the same recipe throughout. For instance, for a fixed
    camera, one only needs to supply a new value for ``h_a`` (water level during video) and keep all the rest exactly
    the same.

To give a first sense of a recipe, an example recipe file (also used in our examples) is displayed below.

    .. literalinclude:: ../../examples/ngwerere/ngwerere.yml
        :language: yaml

If you are not used to .yml files, this may seem a little bit abstract. A few rules and hints are explained below.

* A Yaml file is a text file without any formatting. Hence you may not edit it in Word or other word processors. You
  need to use a raw text editor to modify these. A recommended text editor for windows is notepad++_ which you can
  freely download and install. Set it up as default editor for files with the extension ``.yml`` in Windows Explorer
  by right clicking on a ``.yml`` file in Windows Explorer and
* A .yaml file consists of sections. Each section can have one or multiple sub-sections. And below each subsection
  you may define another set of sub-sections below that. This is very similar to numbering of report or book chapters
  with headings and subheadings, like Chapter 1, section 1.1, subsection 1.1.1, 1.1.2, 1.1.3. A section that has
  subsections is defined with a name and double colon ``:``, e.g. ``video:```. Subsections are defined by providing
  indented text below the section. You can also end these with ``:`` and then define subsections under that with a
  deeper indentation level.
* For indentation, you can either use the <TAB> button on your keyboard, or for instance two spaces to
  indent. Both is ok, but ensure you are very consistent with the indentation level. For instance, first indenting
  with two spaces and then with a <TAB> will give an error.
* Anywhere in the file, you can add comments, by typing ``#``. Any text right of the ``#`` will be interpreted as a
  comment. This is very useful to annotate the files and explain choices made in the file, either for yourself for
  later reference, to distinguish different experiments or make a colleague aware of your choices and reasoning.
* In *pyorc* each main section has a specific name that relates to a larger processing steps. The steps that you can
  go through are ``video``, ``frames``, ``velocimetry``, ``mask``, ``transect`` and ``plot``. Any other sections you
  would provide would simply be skipped, so carefully check your spelling if anything seems to be not working.
* The options you may provide under each section, are (of course) different per section.

The details on the different steps and what you may configure are described in all other chapters of this User Guide.
For quick reference you can use the links below:

* How to select start and end frame of the video to work with: :ref:`video_ug`
* Working with frames, preprocessing and reprojection: :ref:`frames_ug`
* Estimate surface velocity and masking: :ref:`velocimetry_ug`
* Extract velocities over transects: :ref:`transect_ug`
* Plotting frames, velocities and transects: :ref:`plot_ug`

.. _yaml: https://yaml.com/
.. _notepad++: https://notepad++.com/
