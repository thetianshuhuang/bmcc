/**
 * MFM methods
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <math.h>

#include "mixture.h"


/**
 * MFM parameters
 */
struct mfm_params_t {
	double gamma;
	double *v_n;
	PyArrayObject *v_n_py;
};


/**
 * Create MFM parameters struct.
 * @param dict Python dictionary containing hyperparameters. Prior on K should
 * 		be precomputed and used to compute log(V_n(t)) in Python, before
 *		passing the result into C.
 * @return allocated mfm_params_t struct.
 */
void *mfm_create(PyObject *dict) {

	// Unpack dictionary; fetch V_n numpy array
	PyArrayObject *log_vn_py = (
		(PyArrayObject *) PyDict_GetItemString(dict, "V_n"));
	if(log_vn_py == NULL) {
		PyErr_SetString(
			PyExc_KeyError,
			"MFM requires pre-computed V_n coefficients.");
		return NULL;
	}
	double *log_vn = PyArray_DATA(log_vn_py);

	// Allocate
	struct mfm_params_t *params = (
		(struct mfm_params_t *) malloc(sizeof(struct mfm_params_t)));

	// Bind V_n(t); INCREF to prevent garbage collection; destructor DECREFs
	params->v_n = log_vn;
	params->v_n_py = log_vn_py;
	Py_INCREF(log_vn_py);

	// Set gamma
	PyObject *gamma_py = PyDict_GetItemString(dict, "gamma");
	if(gamma_py == NULL) {
		PyErr_SetString(
			PyExc_KeyError,
			"MFM requires mixing parameter gamma (python float).");
		return NULL;
	}
	params->gamma = PyFloat_AsDouble(gamma_py);

	return params;
}


/**
 * Destroy MFM parameters
 * @param params struct to destroy
 */
void mfm_destroy(void *params) {
	Py_DECREF(((struct mfm_params_t *) params)->v_n_py);
	free(params);
}


/**
 * Update MFM parameters
 * @param params parameters to update
 * @param update python dictionary containing new values
 */
void mfm_update(void *params, PyObject *update) {
	struct mfm_params_t *params_tc = (struct mfm_params_t *) params;
	params_tc->gamma = PyFloat_AsDouble(PyDict_GetItemString(update, "gamma"));

	PyArrayObject *log_vn_py = (
		(PyArrayObject *) PyDict_GetItemString(update, "V_n"));
	double *log_vn = PyArray_DATA(log_vn_py);
	int size = PyArray_DIM(log_vn_py, 0);
	for(int i = 0; i < size; i++) { params_tc->v_n[i] = log_vn[i]; }
}


/**
 * Log coefficients for MFM: |c_i| + gamma
 * @param params MFM hyperparameters
 * @param size size of cluster
 * @param nc number of clusters
 * @return log(|c_i| + gamma)
 */
double mfm_log_coef(void *params, int size, int nc) {
	return log(size + ((struct mfm_params_t *) params)->gamma);
}


/**
 * Log coefficients for new cluster: gamma * V_n(t + 1) / V_n(t)
 * @param params MFM hyperparameters
 * @param nc number of clusters
 * @return log(gamma) + log(V_n(t + 1)) - log(V_n(t))
 */
double mfm_log_coef_new(void *params, int nc) {
	struct mfm_params_t *params_tc = (struct mfm_params_t *) params;
	return log(params_tc->gamma) + params_tc->v_n[nc + 1] - params_tc->v_n[nc];
}


/**
 * mfm_methods package
 */
ModelMethods MFM_METHODS = {
	&mfm_create,
	&mfm_destroy,
	&mfm_update,
	&mfm_log_coef,
	&mfm_log_coef_new
};
