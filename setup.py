#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="OpenRiverCam",
    description="OpenRiverCam is a front and backend to control river camera observation locations",
    long_description=readme + "\n\n",
    url="https://github.com/TAHMO/OpenRiverCam",
    author="Hessel Winsemius",
    author_email="winsemius@rainbowsensing.com",
    packages=find_packages(),
    package_dir={"OpenRiverCam": "OpenRiverCam"},
    test_suite="tests",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.8",
    install_requires=[
        "black",
        "numpy",
        "pip",
        "scipy",
        "scikit-image",
        "opencv-python",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black"],
        "optional": [],
    },
    # scripts=["OpenRiverCam"],
    entry_points="""
    """,
    include_package_data=True,
    license="GPLv3",
    zip_safe=False,
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Photogrammetry",
        "License :: OSI Approved :: ???",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="hydrology hydrometry river-flow OpenRiverCam",
)
