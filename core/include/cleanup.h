/**
 * Cleanup gibbs sampler
 */

#ifndef CLEANUP_H
#define CLEANUP_H

#define DOCSTRING_CLEANUP_GIBBS \
	"Clean up clustering by running a single expectation maximization " \
	"iteration.\n" \
	"Runs exactly like the gibbs sampler, except assignments are made with " \
	"maximum\n" \
	"likelihood instead of sampling, and new cluster likelihood is 0." \
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
    "    initialization should support Gibbs sampling."

PyObject *cleanup_iter_py(PyObject *self, PyObject *args, PyObject *kwargs);

#endif
