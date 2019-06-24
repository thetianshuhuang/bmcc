from distutils.core import setup, Extension
import numpy as np

setup(
    name='cluster_utils', version='1.0',
    ext_modules=[Extension('cluster_utils', ['membership.c'])],
    include_dirs=[np.get_include()])
