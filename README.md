<figure>
    <img src="https://raw.githubusercontent.com/localdevices/pyorc/main/docs/_static/orc_logo_color.svg"
width=100 align="right">
</figure>
<br>

# pyOpenRiverCam

[![PyPI](https://badge.fury.io/py/pyopenrivercam.svg)](https://pypi.org/project/pyopenrivercam)
[![Conda-Forge](https://anaconda.org/conda-forge/pyopenrivercam/badges/version.svg)](https://anaconda.org/conda-forge/pyopenrivercam)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=localdevices_pyorc&metric=coverage)](https://sonarcloud.io/summary/new_code?id=localdevices_pyorc)
[![python](https://img.shields.io/pypi/pyversions/pyopenrivercam?color=%2376519B)](https://pypi.org/project/pyopenrivercam/)
[![docs_latest](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://localdevices.github.io/pyorc/latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/localdevices/pyorc.git/main?labpath=examples)
[![License](https://img.shields.io/github/license/localdevices/pyorc?style=flat)](https://github.com/localdevices/pyorc/blob/main/LICENSE)
[![downloads](https://img.shields.io/pypi/dm/pyopenrivercam)](https://pypi.org/project/pyopenrivercam/)

**pyorc**, short for "pyOpenRiverCam" is a fully Open Source library for performing image-based river flow analysis. It is the underlying library for
computations on the fully open software stack OpenRiverCam. **pyorc** can only be successful if the underlying methods
are made available openly for all. Currently **pyorc** implements Large-scale Particle Image Velocimetry (LSPIV) based
flow analysis using the OpenPIV library and reprojections and image pre-processing with OpenCV. We wish to extend this
to Large-scale Particle Tracking Velocimetry (LSPTV) and Space-Time Image Velocimetry (STIV) for conditions that are less favourable for LSPIV using open
libraries or extensions to this code.

![example_image](https://raw.githubusercontent.com/localdevices/pyorc/main/docs/ngwerere.jpg)
Image: Example of pyorc velocimetry over Ngwerere river at the Zambezi Road crossing - Lusaka, Zambia.

Current capabilities are:
* Reading of frames and reprojection to surface
* Velocimetry estimation at user-defined resolution
* Discharge estimation over provided cross-section
* Plotting of velocimetry results and cross-section flows in camera, geographical and orthoprojected perspectives.

We use the well-known **xarray** data models and computation pipelines (with dask) throughout the entire library to
guarantee an easy interoperability with other tools and methods, and allow for lazy computing.

We are seeking funding for the following frequently requested functionalities:
* Exports to simple text formats and GIS-compatible layers
* Exports to augmented reality videos
* Implementation of additional processing algorithms (STIV and LSPTV)
* Improved nighttime / poor weather conditions processing through learning approaches

If you wish to fund this or other work on features, please contact us at info@rainbowsensing.com.

> **_note:_**  For instructions how to get Anaconda (with lots of pre-installed libraries) or Miniconda (light weight) installed, please go to https://docs.conda.io/projects/conda/en/latest/

> **_manual:_** Please go to https://localdevices.github.io/pyorc for the latest documentation

> **_compatibility:_** At this moment **pyorc** works with any video compatible with OpenCV as long as it has proper metadata.

## Installation
You need a python environment. We recommend using the Miniforge project. Download
the latest miniforge installer from https://github.com/conda-forge/miniforge and follow the installation instructions
before continuing installing **pyorc**.

### virtual environment
To get started with **pyorc**, we recommend to set up a python virtual environment. This ensures that installed libraries
**pyorc** will not conflict with other libraries or library versions which you may need for other projects.

Setting up a virtual environment can be done with:
```shell
python -m venv pyorc_env
```
this creates a new folder `pyorc_env` on your disk which contains your virtual environment.
After activating the environment, any package you install will be installed in this environment only.
Activating in Unix/Linux is done as follows:

```shell
source pyorc_env/bin/activate
```
In Windows, the activation script is in a different folder. Type the following to activate the environment.
```shell
pyorc_env\Scripts\activate
```

### Installation for direct use

If you simply want to add **pyorc** to an existing python installation or virtual environment, then follow these
instructions.

First activate the environment you want **pyorc** to be installed in (if you don't care about virtual environments, then
simply skip this step). See the above sub-section for information. You can simply install pyorc with all its
dependencies as follows:

```shell
pip install pyopenrivercam[extra]
```
The `[extra]` section ensures that also geographical plotting is supported, which we recommend especially for the
set up of a camera configuration with RTK-GPS measured control points.

### Upgrading from pypi with pip

Did you read about a new version and you want to upgrade? Simply activate your virtual environment, type

```shell
pip install --upgrade pyopenrivercam[extra]
```

and then enjoy the latest features.

### installation with mamba package manager

If you use `mamba` as a package manager, then the steps are the same, except for the installation step, which is:

```shell
mamba install pyopenrivercam
```

### Installation from latest code base

To install **pyorc** from scratch in a new virtual environment from the code base, go through these steps. Logical cases
when you wish to install from the code base are when you wish to have the very latest non-released version.

First, clone the code with `git` and move into the cloned folder.

```
git clone https://github.com/localdevices/pyorc.git
cd pyorc
```

Set up a virtual environment with all dependencies as follows:
```
conda env create -f envs/pyorc-dev.yml
conda activate pyorc-dev
```
then install **pyorc** from the code base as follows:
```
pip install .
```
> **_note:_** **pyorc** is now installed in a virtual environment called `pyorc-dev`. This means that if you wish to run
python with **pyorc**. You need to always first activate this environment before running python (or jupyter). This is
done with the following command:
```
conda activate pyorc-dev
```
### Installation from latest code base as developer

Clone the repository with ssh and move into the cloned folder.

```
git clone git@github.com:localdevices/pyorc.git
cd pyorc
```

Setup a virtual developers environment and install the package as follows:
```
conda env create -f envs/pyorc-dev.yml
conda activate pyorc-dev
pip install -e .
```

Make sure you install pre-commit hooks so that code is properly linted before pushing.
```
pip install pre-commit
pre-commit install
```

## Using pyorc
To use **pyorc**, you can use the API for processing. A command-line interface is forthcoming pending funding.
A manual is also still in the making.

## Acknowledgement
The first development of pyorc has been supported by the World Meteorological Organisation - HydroHub.

## License
**pyorc** is licensed under AGPL Version 3 (see [LICENSE](./LICENSE) file).

**pyorc** uses the following important libraries and software with said licenses.

| Package    | Version | License                            |
|------------|---------|------------------------------------|
| ffpiv      | 0.1.2   | AGPLv3                             |
| numpy      | 1.26.4  | BSD License                        |
| opencv2    | 4.10.0  | MIT License                        |
| openpiv    | 0.25.3  | GPLv3                              |
| matplotlib | 3.9.2   | Python Software Foundation License |
| geopandas  | 1.0.1   | BSD License                        |
| pandas     | 2.2.2   | BSD License                        |

Project organisation
--------------------

    .
    ├── CHANGELOG.md        <- Version-based changelog documentation
    ├── README.md           <- This file
    ├── LICENSE             <- License file containing AGPLv3.0 license terms
    ├── TRADEMARK.md        <- Trademark guidelines
    ├── pyproject.toml      <- setup pipeline compatible with pip
    ├── environment.yml     <- YML-file for setting up a conda environment with dependencies
    ├── docs                <- Sphinx documentation source code
        ├── ...             <- Sphinx source code files
    ├── examples            <- Jupyter notebooks with examples how to use the API
        ├── ...             <- individual notebooks and folder with example data files
    ├── pyorc               <- pyorc library
        ├── ...             <- pyorc functions and API files
    ├── tests               <- pytest suite
        ├── ...             <- pytest functions on API level
