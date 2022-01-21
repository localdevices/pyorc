#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pyorc",
    version="0.1.0",
    description="pyOpenRiverCam reprojects and analyzes river videos into veclocities and discharge using Particle Image \ Velocimetry. It is the main computation engine of OpenRiverCam",
    long_description=readme + "\n\n",
    long_description_content_type="text/markdown",
    url="https://github.com/localdevices/pyorc",
    author="Hessel Winsemius",
    author_email="winsemius@rainbowsensing.com",
    packages=find_packages(),
    package_dir={"OpenRiverCam": "OpenRiverCam"},
    test_suite="tests",
    python_requires=">=3.8",
    install_requires=[
        "descartes",
        "geos",
        "geojson",
        "matplotlib",
        "netCDF4",
        "numpy",
        "opencv-python-headless",
        "openpiv",
        "pip",
        "pyproj",
        "rasterio",
        "scikit-image",
        "scipy",
        "shapely",
        "xarray",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "optional": [],
    },
    #entry_points="""
    #""",
    include_package_data=True,
    license="GPLv3",
    zip_safe=False,
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="hydrology hydrometry river-flow OpenRiverCam",
)
