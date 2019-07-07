

from distutils.core import setup, Extension
import numpy as np

import os

cfiles = ['./src/' + s for s in os.listdir('./src') if s[-2:] == '.c']
print("C Files:")
print(cfiles)

setup(
    name='bclust', version='0.0.2',
    packages=['bclust'],
    ext_modules=[Extension('bclust.core', cfiles)],
    include_dirs=[np.get_include(), './src'])

