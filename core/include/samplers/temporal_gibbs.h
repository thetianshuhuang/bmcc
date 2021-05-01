/**
 * Gibbs Sampler
 */

#ifndef TEMPORAL_GIBBS_H
#define TEMPORAL_GIBBS_H

#include <Python.h>

#define DOCSTRING_TEMPORAL_GIBBS_ITER \
    "Run Temporal Gibbs sampler for one iteration.\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data : np.array\n" \
    "    Data matrix. Should have type np.float64 (double); row-major order.\n" \
    "assignments : np.array\n" \
    "    Assignment array. Should have type np.uint16.\n" \
    "model : capsule\n" \
    "    Capsule containing model data. Mixture model and component model " \
        "used in\n" \
    "    initialization should support Gibbs sampling." \
    "\n" \
    "Keyword Args\n" \
    "------------\n" \
    "annealing : float\n" \
    "    annealing factor."

PyObject *temporal_gibbs_iter_py(
    PyObject *self, PyObject *args, PyObject *kwargs);

#endif
