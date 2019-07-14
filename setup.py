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


BUILD_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


C_EXTENSION = Extension(
    "bmcc.core",

    # List module.c first
    sources=[
        './src/module.c'
    ] + [
        './src/' + s for s in os.listdir('./src') if s != 'module.c'
    ],

    # Headers; must be in a separate directory from sources since include_dirs
    # only takes directories as an argument
    include_dirs=[np.get_include(), './include'],

    # Configuration
    define_macros=[
        # Base size for dynamically sized vectors
        ("BASE_VEC_SIZE", 32),
        # Build Datetime -- used for debug purposes
        # Bound to bmcc.CONFIG["BUILD_DATETIME"]
        ("BUILD_DATETIME", '"' + BUILD_DATETIME + '"'),
        # API Descriptions; needed to create other extensions that modify
        # core capsules
        ("COMPONENT_METHODS_API", "\"bmcc.core.ComponentMethods\""),
        ("COMPONENT_PARAMS_API", "\"bmcc.core.ComponentParams\""),
        ("MODEL_METHODS_API", "\"bmcc.core.ModelMethods\""),
        ("MODEL_PARAMS_API", "\"bmcc.core.ModelParams\""),
        ("MIXTURE_MODEL_API", "\"bmcc.core.MixtureModel\"")
    ],
)


MODULE_SHORT_DESC = (
    "Implementation of Markov Chain Bayesian Clustering techniques, including "
    "DPM and MFM, with an abstract Mixture Model and Component Model API.")

with open("README.md", "r") as f:
    MODULE_LONG_DESC = f.read()


setup(
    # About
    name='bmcc',
    version='1.0.1',
    author='Tianshu Huang',
    author_email='thetianshuhuang@gmail.com',

    # Description
    description=MODULE_SHORT_DESC,
    long_description=MODULE_LONG_DESC,
    long_description_content_type="text/markdown",
    url="https://github.com/thetianshuhuang/bmcc",

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

    # Classifiers (https://pypi.org/classifiers/)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
