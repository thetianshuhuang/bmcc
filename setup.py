"""Setup Script

C files are contained in ./src, and are installed to the submodule bmcc.core.
Capsule APIs therefore start with bmcc.core (i.e. bmcc.core.ModelMethods, etc).

Usage
-----
Local install:
    python3 setup.py install
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

from distutils.core import setup, Extension
import numpy as np
import os


C_EXTENSION = Extension(
    "bmcc.core",
    ['./src/' + s for s in os.listdir('./src')],
    define_macros=[("BASE_VEC_SIZE", 1024)]
)


MODULE_SHORT_DESC = (
    "Implementation of Markov Chain Bayesian Clustering techniques, including "
    "DPM and MFM, with an abstract Mixture Model and Component Model API.")

try:
    with open("README.md", "r") as f:
        MODULE_LONG_DESC = f.read()
except FileNotFoundError:
    MODULE_LONG_DESC = ""


setup(
    # About
    name='bmcc',
    version='0.2.3',
    author='Tianshu Huang',
    author_email='thetianshuhuang@gmail.com',

    # Description
    description=MODULE_SHORT_DESC,
    long_description=MODULE_LONG_DESC,
    long_description_content_type="text/markdown",
    url="https://github.com/thetianshuhuang/bmcc",

    # Requirements
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib"
    ],

    # Python Core
    packages=['bmcc', 'bmcc.models'],

    # C Extension
    ext_modules=[C_EXTENSION],
    include_dirs=[np.get_include(), './src'],

    # Metadata
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
