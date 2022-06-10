Installation
============

User install
------------

To install **pyorc**, you will need a package manager in the Anaconda/Miniconda ecosystem such as **conda** or **mamba**.

We recommend using the Mambaforge_ Python distribution. This installs Python and the mamba package manager. 
Miniforge_ and Miniconda_ will install Python and the conda package manager.

In general, **mamba** is a lot faster than **conda** and requires less memory.

Installation from pypi with pip
-------------------------------

First activate the environment you want **pyorc** to be installed in (if you don't care about virtual environments, then
simply skip this step). Dependencies **rasterio**, **geopandas** and **cartopy** are known to be difficult to install with
**pip**. Therefore, we highly recommend to use a conda environment that already includes these dependencies before
installation. If you don't have them yet, install them as follows:

.. code-block:: console

    $ conda activate <name-of-your-environment>
    $ conda install -c conda-forge cartopy geopandas rasterio

Then get the pyorc library from pypi using

.. code-block:: console

    $ pip install pyopenrivercam

Installation from latest code base
----------------------------------
To install the latest (unreleased) version from github, replace the last command shown in the previous section by:

.. code-block:: console

    $ pip install git+https://github.com/localdevices/pyorc.git

.. note::
    You may have to uninstall **pyorc** first to successfully install from github.


.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Mambaforge: https://github.com/conda-forge/miniforge#mambaforge
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _limitations: https://www.anaconda.com/blog/anaconda-commercial-edition-faq
.. _mamba package manager: https://github.com/mamba-org/mamba
.. _conda package manager: https://docs.conda.io/en/latest/
.. _pip package manager: https://pypi.org/project/pip/
.. _manage environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
