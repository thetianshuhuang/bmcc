from distutils.core import setup, Extension
import numpy as np

setup(
    name='cluster_util', version='1.0',
    ext_modules=[Extension('cluster_util', ['util.c'])],
    include_dirs=[np.get_include()])
