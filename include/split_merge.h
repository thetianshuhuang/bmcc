/**
 * Split Merge Sampler
 */

#ifndef SPLIT_MERGE_H
#define SPLIT_MERGE_H

#include <Python.h>

#define DOCSTRING_SPLIT_MERGE \
    "Run Split Merge sampler for one iteration.\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data : np.array\n" \
    "    Data matrix. Should have type np.float64 (double); row-major order.\n" \
    "assignments : np.array\n" \
    "    Assignment array. Should have type np.uint16.\n" \
    "model : capsule\n" \
    "    Capsule containing model data.Mixture model and component model " \
        "used in\n" \
    "    initialization should support Split Merge sampling."

PyObject *split_merge_py(PyObject *self, PyObject *args);

#endif
