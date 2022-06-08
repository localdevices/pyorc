Installation
============

User install
------------

ODMax can be installed with ``pip install``.

To install ODMax open a console, activate a python virtual environment if you wish and do:

.. code-block:: console

    $ pip install odmax

This will install the library, API and command-line utility of ODMax.

.. note::
    For windows users, it may be difficult to get geographic libraries installed with the normal pip install procedure.
    For windows users we therefore recommend that you follow the Developer install instructions and setup a ``conda``
    environment using the ``environment.yml`` delivered with the code. This environment includes geopandas and cartopy.
    After that follow the instructions above from within the established environment. We will deliver a conda package
    soon so that you can simply install with ``conda install odmax``.

Developer install
------------------
If you want to download ODMax directly from git to easily have access to the latest developments or
make changes to the code you can use the following steps.

First, clone ODMax's ``git`` repo from
`github <https://github.com/localdevices/ODMax.git>`_, then navigate into the
the code folder as follows:

.. code-block:: console

    $ git clone https://github.com/localdevices/ODMax.git
    $ cd ODMax

Then, make and activate a new python environment if you wish, or use your base environment.
After that, build and install odmax for development using ``pip`` as follows:

.. code-block:: console

    $ pip install -e .

Installation of exiftool for metadata extraction
------------------------------------------------

Especially for photogrammetry or 360 streetview applications, it is essential to have time stamps and geographical
coordinates embedded in the extracted stills. ODMax automatically extracts such information from 360-video files if
these are recorded by the device used. In order to do this, ODMax requires ``exiftool`` to be installed and available on
the path. To install ``exiftool`` in Windows, please follow the download and installation instructions for Windows on
https://exiftool.org/install.html. For Linux, you can also follow the download and installation instructions, or simply
acquire a stable version from the package manager of your installed distribution. E.g. for Ubuntu users, you can easily
install ``exiftool`` as follows:

.. code-block:: console

    $ sudo apt update
    $ sudo apt install libimage-exiftool-perl

.. note::
    For windows installation, please make sure to rename ``exiftool(k).exe`` to ``exiftool.exe`` and make sure
    ``exiftool.exe`` is located in a folder that is part of your ``PATH`` environment variable. If you do not know how
    to get a folder added to your ``PATH`` then simply ensure that you copy the exiftool(k).exe to ``C:\WINDOWS``
    and rename it after that to ``exiftool.exe``.

