from distutils.core import setup, Extension
import numpy as np

setup(
    name='test', version='1.0',
    ext_modules=[Extension('test', ['test.cpp'])],
    include_dirs=[np.get_include()])
