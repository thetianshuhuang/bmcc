/**
 * Gibbs Sampler
 */

#ifndef GIBBS_H
#define GIBBS_H

#include <Python.h>

#define DOCSTRING_GIBBS_ITER \
    "Run Gibbs sampler for one iteration.\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data : np.array\n" \
    "    Data matrix. Should have type np.float64 (double); row-major order.\n" \
    "assignments : np.array\n" \
    "    Assignment array. Should have type np.uint16.\n" \
    "model : capsule\n" \
    "    Capsule containing model data.\n" \

PyObject *gibbs_iter_py(PyObject *self, PyObject *args);

#endif
