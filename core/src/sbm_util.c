/**
 * SBM utility functions
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>
#include <stdint.h>

#include "../include/misc_math.h"


/**
 * C-accelerated implementation of SBM simulation
 */
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
	if(!success) { return NULL; }

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


/**
 * Run update for SBM Q array
 * Returns new Q array; DOES NOT MODIFY.
 * @param data : data array
 * @param asn : assignment array
 * @param n : number of points (size of data, asn)
 * @param k : number of clusters (size of Q)
 * @param a : Q beta prior parameter
 * @param b : Q beta prior parameter
 */
double *sbm_update(
	uint8_t *data, uint16_t *asn, int n, int k, double a, double b)
{
	// Connection matrix (a and b matrices)
	double *Q_a = malloc(sizeof(double) * k * k);
	double *Q_b = malloc(sizeof(double) * k * k);

	// Populate with prior
	for(int i = 0; i < k * k; i++) {
		Q_a[i] = a;
		Q_b[i] = b;
	}

	// For each pair of points...
	for(int i = 0; i < n; i++) { for(int j = 0; j < n; j++) {
		if(i != j) {
			if(data[i * n + j]) {
				Q_a[asn[i] * k + asn[j]] += 1;
				Q_a[asn[j] * k + asn[i]] += 1;
			}
			else {
				Q_b[asn[i] * k + asn[j]] += 1;
				Q_b[asn[j] * k + asn[i]] += 1;
			}
		}
	} }

	// Generate new Q
	double *Q_new = malloc(sizeof(double) * k * k);
	for(int i = 0; i < k * k; i++) {
		Q_new[i] = rand_beta(Q_a[i], Q_b[i]);
	}

	// Caller takes ownership of Q_new
	free(Q_a);
	free(Q_b);
	return Q_new;
}


/**
 * Resample Q array for SBM
 */
PyObject *sbm_update_py(PyObject *self, PyObject *args)
{
	PyObject *data_py;
	PyObject *asn_py;
	int k;
	double a;
	double b;

	bool success = PyArg_ParseTuple(
		args, "O!O!idd",
		&PyArray_Type, &data_py,
		&PyArray_Type, &asn_py,
		&k, &a, &b);
	if(!success) { return NULL; }

	// Get new Q
	double *Q_new = sbm_update(
		(uint8_t *) PyArray_DATA((PyArrayObject *) data_py),
		(uint16_t *) PyArray_DATA((PyArrayObject *) asn_py),
		PyArray_DIM((PyArrayObject *) asn_py, 0),
		k, a, b);

	// Create wrapper
	npy_intp dims[2] = {k, k};
	PyObject *Q_new_py = PyArray_SimpleNewFromData(
		2, &dims, NPY_FLOAT64, Q_new);
	PyArray_ENABLEFLAGS(Q_new_py, NPY_ARRAY_OWNDATA);

	return Q_new_py;
}
