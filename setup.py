

from distutils.core import setup, Extension
import numpy as np

import os

cfiles = ['./src/' + s for s in os.listdir('./src') if s[-2:] == '.c']
print("C Files:")
print(cfiles)

setup(
    name='bayesian_clustering_c', version='0.0.1',
    ext_modules=[Extension('bayesian_clustering_c', cfiles)],
    include_dirs=[np.get_include(), './src'])

