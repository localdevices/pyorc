.. _installation:

============
Installation
============

.. _user_install:

User install
============

To install **pyorc**, you will need a package manager in the Anaconda/Miniconda ecosystem such as **conda** or **mamba**.

We recommend using the Mambaforge_ Python distribution. This installs Python and the mamba package manager.
Miniforge_ and Miniconda_ will install Python and the conda package manager.

In general, **mamba** is a lot faster than **conda** and requires less memory.

Making a virtual environment
============================
To get started with **pyorc**, we recommend to set up a python virtual environment. This ensures that installed libraries
**pyorc** will not conflict with other libraries or library versions which you may need for other projects.

Setting up a virtual environment can be done with:
.. code-block:: console

    $ python -m venv pyorc_env

this creates a new folder `pyorc_env` on your disk which contains your virtual environment.
After activating the environment, any package you install will be installed in this environment only.
Activating in Unix/Linux is done as follows:

.. code-block:: console

    source pyorc_env/bin/activate

In Windows, the activation script is in a different folder. Type the following to activate the environment.

.. code-block:: console

    pyorc_env\Scripts\activate

Alternatively, you can make and activate an environment with mamba and python and pip
installations in that environment as follows:

.. code-block:: console

    mamba create --name pyorc_env
    mamba activate pyorc_env
    mamba install python pip

.. _install_pip:

Installation from pypi with pip
===============================
The most straightforward installation is through ``pip``. This also works in a Raspberry Pi 64-bit OS.
First activate the environment you want **pyorc** to be installed in (if you don't care about virtual environments, then
simply skip this step). Activation of a virtual environment made with ``venv`` or with mamba is
explained in the section above.

.. code-block:: console

    $ pip install pyopenrivercam[extra]

The ``[extra]`` section ensures that also geographical plotting is supported, which we recommend especially for the
set up of a camera configuration.

.. _install_conda-forge:

Upgrading from pypi with pip
============================
Did you read about a new version and you want to upgrade? Simply activate your virtual environment, type

.. code-block:: console

    $ pip install --upgrade pyopenrivercam[extra]

and then enjoy the latest features.

Installation from conda-forge package
=====================================

Activate your mamba created virtual environment with the activate command as follows

Once the environment is prepared, install the latest `conda-forge` package, with all dependencies using the following
command:

.. code-block:: console

    $ mamba install -c conda-forge pyopenrivercam

.. _install_code:

Installation from latest code base
==================================

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
