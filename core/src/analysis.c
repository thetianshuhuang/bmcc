/** 
 * MCMC Output Analysis
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>

#include "../include/type_check.h"


/**
 * Get aggregation score.
 */
PyObject *aggregation_score_py(PyObject *self, PyObject *args)
{
	PyArrayObject *actual_py;
	PyArrayObject *asn_py;

	bool success = PyArg_ParseTuple(
		args, "O!O!", &PyArray_Type, &actual_py, &PyArray_Type, &asn_py);
	if(!success) { return NULL; }
	if(!type_check_assignments(actual_py, asn_py)) { return NULL; }

	uint16_t *actual = (uint16_t *) PyArray_DATA(actual_py);
	uint16_t *asn = (uint16_t *) PyArray_DATA(asn_py);

	int dim = PyArray_DIM(actual_py, 0);

	double total = 0;
	double score = 0;
	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			if(actual[i] == actual[j]) {
				total += 1;
				if(asn[i] == asn[j]) {
					score += 1;
				}
			}
		}
	}
	return Py_BuildValue("f", score / total);
}


/**
 * Get segregation score.
 */
PyObject *segregation_score_py(PyObject *self, PyObject *args)
{
	PyArrayObject *actual_py;
	PyArrayObject *asn_py;

	bool success = PyArg_ParseTuple(
		args, "O!O!", &PyArray_Type, &actual_py, &PyArray_Type, &asn_py);
	if(!success) { return NULL; }
	if(!type_check_assignments(actual_py, asn_py)) { return NULL; }

	uint16_t *actual = (uint16_t *) PyArray_DATA(actual_py);
	uint16_t *asn = (uint16_t *) PyArray_DATA(asn_py);

	int dim = PyArray_DIM(actual_py, 0);

	double total = 0;
	double score = 0;
	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			if(actual[i] != actual[j]) {
				total += 1;
				if(asn[i] != asn[j]) {
					score += 1;
				}
			}
		}
	}
	return Py_BuildValue("f", score / total);
}


/** 
 * Mixture of Gaussian Oracle Pairwise Probability Matrix
 */
PyObject *oracle_matrix_py(PyObject *self, PyObject *args)
{
	PyArrayObject *likelihoods_py;

	bool success = PyArg_ParseTuple(
		args, "O!", &PyArray_Type, &likelihoods_py);
	if(!success) { return NULL; }

	int size = PyArray_DIM(likelihoods_py, 0);
	int clusters = PyArray_DIM(likelihoods_py, 1);
	double *likelihoods = PyArray_DATA(likelihoods_py);

	npy_intp dims[2] = {size, size};
	PyArrayObject *prob_matrix_py = (
		(PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64));
	double *prob_matrix = (double *) PyArray_DATA(prob_matrix_py);

	// For assignment likelihoods L_k(i) of assigning point i to cluster k:
	// Pairwise Probability [i, j] = Sum_k (L_k(i) * L_k(j))
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			double pairwise = 0;
			for(int k = 0; k < clusters; k++) {
				pairwise += (
					likelihoods[i * clusters + k] *
					likelihoods[j * clusters + k]);
			}
			prob_matrix[i * size + j] = pairwise;
		}
	}	

	return (PyObject *) prob_matrix_py;
}

