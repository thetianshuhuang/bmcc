"""Helper functions for R wrapper"""

import numpy as np


def is_np_array(arr):
    return type(arr) == np.ndarray


def is_uint16(arr):
    return arr.dtype == np.uint16


def is_float64(arr):
    return arr.dtype == np.float64


def is_contiguous(arr):
    return arr.flags['C_CONTIGUOUS']
