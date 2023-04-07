.. _quickstart:

===========================
Quick start for programmers
===========================

The best start for *pyorc* is to go through a number of examples that demonstrate how to move from a video
and some field work to estimates of surface velocity and discharge, and insightful plots. If you are not a programmer
then please look at the :ref:`User Guide <manual>` for further information.

.. tip::

    .. raw:: html

        <div>
            All examples can also be performed interactively without installation of *pyorc*. Click here for the interactive
            version.
            <a href="https://mybinder.org/v2/gh/localdevices/pyorc.git/main?labpath=examples{{ docname|e }}" target="_blank" rel="noopener noreferrer"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg"></a>
        </div>

.. _examples:

List of examples
================

For a static (non-interactive) view of the examples follow one of the links below:

* `Setting up a camera configuration <_examples/01_Camera_Configuration_single_video.ipynb>`_
* `Analyze surface velocity with Particle Image Velocimetry <_examples/02_Process_velocimetry.ipynb>`_
* `Masking spurious velocities and plotting <_examples/03_Plotting_and_masking_velocimetry_results.ipynb>`_
* `Extracting cross-sections and discharge estimation <_examples/04_Extracting_crosssection_velocities_and_discharge.ipynb>`_
* `Performing camera calibration with a video of a chessboard pattern <_examples/05_Camera_calibration.ipynb>`_

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Table of Content

    Setup a camera configuration for processing a single video <_examples/01_Camera_Configuration_single_video.ipynb>
    Analyze surface velocities of a video with velocimetry <_examples/02_Process_velocimetry.ipynb>
    Immersive plotting and analyzing results <_examples/03_Plotting_and_masking_velocimetry_results.ipynb>
    Obtain a discharge measurement over a cross section <_examples/04_Extracting_crosssection_velocities_and_discharge.ipynb>
    Camera calibration with chessboard pattern <_examples/05_Camera_calibration.ipynb>