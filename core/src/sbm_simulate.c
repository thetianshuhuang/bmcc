
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>
#include <stdint.h>

#include "../include/misc_math.h"


PyObject *sbm_simulate_py(PyObject *self, PyObject *args)
{
	PyArrayObject *Q_py;
	PyArrayObject *A_py;
	PyArrayObject *assignments_py;

	bool success = PyArg_ParseTuple(
		args, "O!O!O!",
		&PyArray_Type, &Q_py,
		&PyArray_Type, &A_py,
		&PyArray_Type, &assignments_py);
	if(!success) { Py_RETURN_NONE; }

	uint16_t *asn = (uint16_t *) PyArray_DATA(assignments_py);
	uint8_t *A = (uint8_t *) PyArray_DATA(A_py);
	double *Q = (double *) PyArray_DATA(Q_py);

	int n = PyArray_DIM(assignments_py, 0);
	int k = PyArray_DIM(Q_py, 0);

	// Generate bernoulli off-diagonal links
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < i; j++) {
			double p = Q[asn[i] * k + asn[j]];
			uint8_t val = rand_double() > p ? 0 : 1;
			A[i * n + j] = val;
			A[j * n + i] = val;
		}
	}

	// Each point is connected to itself
	for(int i = 0; i < n; i++) { A[i * n + i] = 1; }

	Py_RETURN_NONE;
}
