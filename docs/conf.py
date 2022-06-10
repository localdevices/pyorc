# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import shutil
import sphinx_autosummary_accessors
import os
import glob
import sys
import pyorc
# from pyorc import Video
# from pyorc import Velocimetry

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

##copy notebooks to inside this
#src_dir = "../notebooks"
#dst_dir = "."
#if not(os.path.isdir(dst_dir)):
    #os.makedirs(dst_dir)
 
#copy relevant notebooks
#shutil.copy(os.path.join(src_dir, "reprojection.ipynb"), dst_dir)
#shutil.copy(os.path.join(src_dir, "geotags.ipynb"), dst_dir)

##also copy the README of the notebook folder
#shutil.copy(os.path.join(src_dir, "README.rst"), dst_dir)


# -- Project information -----------------------------------------------------

project = 'pyorc'
copyright = '2022, Rainbow Sensing'
author = 'Hessel Winsemius'

# The full version, including alpha/beta/rc tags
# TODO: uncomment this as soon as we have a version number on the package within pypi
# release = pkg_resources.get_distribution("ODMax").version
release = '0.2.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "sphinxcontrib.programoutput",
    "sphinx_autosummary_accessors"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ".rst"
master_doc = "index"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

autoclass_content = "both"
autosummary_generate = True
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
print(sys.path)
