

from distutils.core import setup, Extension
import numpy as np

import os

cfiles = ['./src/' + s for s in os.listdir('./src') if s[-2:] == '.c']
print("C Files:")
print(cfiles)

setup(
    name='bmcc', version='0.1.0',
    packages=['bmcc', 'bmcc.models'],
    ext_modules=[Extension('bmcc.core', cfiles)],
    include_dirs=[np.get_include(), './src'])

