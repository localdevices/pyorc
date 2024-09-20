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
import sys
from distutils.dir_util import copy_tree
import pyorc

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

def remove_dir_content(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    if os.path.isdir(path):
        shutil.rmtree(path)


# -- Copy notebooks to include in docs -------
if os.path.isdir("_examples"):
    remove_dir_content("_examples")
os.makedirs("_examples/ngwerere")
copy_tree("../examples/ngwerere", "_examples/ngwerere")
copy_tree("../examples/camera_calib", "_examples/camera_calib")

# copy specific notebooks to include in build
shutil.copy("../examples/01_Camera_Configuration_single_video.ipynb", "_examples")
#
# Notebook 02 requires considerable rendering time. Therefore it is not executed unless a final build is done
shutil.copy("../examples/02_Process_velocimetry.ipynb", "_examples")
shutil.copy("../examples/03_Plotting_and_masking_velocimetry_results.ipynb", "_examples")
shutil.copy("../examples/04_Extracting_crosssection_velocities_and_discharge.ipynb", "_examples")
shutil.copy("../examples/05_Camera_calibration.ipynb", "_examples")

# -- Project information -----------------------------------------------------

project = 'pyorc'
copyright = '2024, Rainbow Sensing'
author = 'Hessel Winsemius'

# The full version, including alpha/beta/rc tags
# release = pkg_resources.get_distribution("pyorc").version
release = pyorc.__version__

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
    "sphinx_autosummary_accessors",
    "sphinx_design"
]
autosummary_generate = True
nbsphinx_allow_errors = True
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
autodoc_member_order = "bysource"
autoclass_content = "both"

html_static_path = ['_static']
html_css_files = ["theme-localdevices.css"]
html_logo = "_static/orc_logo_color.svg"
html_favicon = "_static/orc_favicon.svg"

html_theme_options = {
    "show_nav_level": 2,
    "navbar_align": "content",
    # "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "Local Devices",
            "url": "https://localdevices.org",
            "icon": "_static/logo.svg",
            "type": "local",
        },
    ],
    "logo": {
        "text": f"pyOpenRiverCam {release}"
    }
}

html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise interprise
    "github_user": "localdevices",
    "github_repo": "pyorc",
    "github_version": "docs",
    "doc_path": "docs",
}


remove_from_toctrees = ["_generated/*", "_build/doctrees/*"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
print(sys.path)
