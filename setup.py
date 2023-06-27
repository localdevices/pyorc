#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pyopenrivercam",
    description="pyopenrivercam (pyorc) is a front and backend to control river camera observation locations",
    version="0.5.1",
    long_description=readme + "\n\n",
    long_description_content_type="text/markdown",
    url="https://github.com/localdevices/pyorc",
    author="Hessel Winsemius",
    author_email="winsemius@rainbowsensing.com",
    packages=find_packages(),
    package_dir={"pyorc": "pyorc"},
    test_suite="tests",
    python_requires=">=3.9",
    install_requires=[
        "click",
        "cython; platform_machine == 'armv7l'",
        "dask",
        "descartes",
        "geojson",
        "geopandas",
        "matplotlib",
        "netCDF4",
        "numpy",
        "opencv-python",
        "openpiv",
        "packaging; platform_machine == 'armv7l'",
        "pip",
        "pyproj",
        "pythran; platform_machine == 'armv7l'",
        "pyyaml",
        "rasterio",
        "scikit-image",
        "scipy",
        "shapely",
        "tqdm",
        "typeguard",
        "xarray"
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "optional": [],
    },
    entry_points={
        "console_scripts": [
            "pyorc = pyorc.cli.main:cli"
        ]
    },
    include_package_data=True,
    license="AGPLv3",
    zip_safe=False,
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="hydrology, hydrometry, river-flow, pyorc",
)
