"""Setup Script

C files are contained in ./src, and are installed to the submodule bmcc.core.
Capsule APIs therefore start with bmcc.core (i.e. bmcc.core.ModelMethods, etc).

Usage
-----
Local install:
    pip3 install .
Build:
    python3 setup.py sdist

Note
----
Since bmcc has a C module, and therefore must compile code, bdist_wheel cannot
be used.

Requires
--------
- numpy: general linear algebra framework
- scipy: Normal distribution, Wishart distribution, numerical solver
- sklearn: clustering metrics (rand index, NMI)
- matplotlib: plots
"""

from setuptools import setup, find_packages
from distutils.core import Extension
import numpy as np

import datetime
import os


#
# -- Module Metadata ----------------------------------------------------------
#

# Build Date (used mainly for debugging C module)
BUILD_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Description
MODULE_SHORT_DESC = (
    "Implementation of Markov Chain Bayesian Clustering techniques, including "
    "DPM and MFM, with an abstract Mixture Model and Component Model API.")

# Long description; pulled from README (is included in MANIFEST)
with open("README.md", "r") as f:
    MODULE_LONG_DESC = f.read()

# Tags / Classifiers (https://pypi.org/classifiers/)
CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Information Analysis"
]

# Core Metadata
META = {
    "name": "bmcc",
    "version": "2.0.0",
    "author": "Tianshu Huang",
    "author_email": "thetianshuhuang@gmail.com",
    "description": MODULE_SHORT_DESC,
    "long_description": MODULE_LONG_DESC,
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/thetianshuhuang/bmcc",
    "classifiers": CLASSIFIERS
}


#
# -- Debug Configuration ------------------------------------------------------
#
# Warning:
#
# Enabling these macros will cause the underlying C module to print debug logs.
# This action will greatly slow down the program (mostly due to terminal output
# limitations).
# These logs cannot be silenced through python.
#
# Sampling modes may or may not choose to respond to these macros.

DEBUG_MACROS = [
    # Show message on metropolis-hastings accept
    # ("SHOW_ACCEPT", 0),
    # Show message on metropolis-hastings reject
    # ("SHOW_REJECT", 0),
]


#
# -- API Names ----------------------------------------------------------------
#
# Names used by the C module to label its capsules. These API names must be
# used by extending modules.

API_NAMES = [
    ("COMPONENT_METHODS_API", "\"bmcc.core.ComponentMethods\""),
    ("MODEL_METHODS_API", "\"bmcc.core.ModelMethods\""),
    ("MIXTURE_MODEL_API", "\"bmcc.core.MixtureModel\"")
]


#
# -- Other Configuration Options ----------------------------------------------
#
OTHER_MACROS = [
    # Base size for dynamically sized vectors
    ("BASE_VEC_SIZE", 32),
    # Build Datetime -- used for debug purposes
    # Bound to bmcc.CONFIG["BUILD_DATETIME"]
    ("BUILD_DATETIME", '"' + BUILD_DATETIME + '"'),
]


#
# -- C Extension --------------------------------------------------------------
#
C_EXTENSION = Extension(
    "bmcc.core",

    # List module.c first
    sources=[
        './core/src/module.c'
    ] + [
        './core/src/' + s for s in os.listdir('./core/src') if s != 'module.c'
    ],

    # Libraries
    libraries=["gsl"],

    # Headers; must be in a separate directory from sources since include_dirs
    # only takes directories as an argument
    include_dirs=[np.get_include(), './core/include'],

    # Configuration
    define_macros=DEBUG_MACROS + API_NAMES + OTHER_MACROS,
)


#
# -- Module -------------------------------------------------------------------
#
setup(
    # Requirements
    python_requires='>=3, <4',
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib"
    ],

    # Python Core
    packages=find_packages("."),

    # C Extension
    ext_modules=[C_EXTENSION],
    include_package_data=True,

    # Core Metadata
    **META
)
