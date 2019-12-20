/**
 * DPM methods
 */

#include <Python.h>
#include <math.h>

#include "../include/mixture.h"


/**
 * DPM Parameters
 */
struct dpm_params_t {
	double alpha;
};


/**
 * Create dpm parameters struct.
 * @param dict Python dictionary containing hyperparameter alpha
 * @return allocated dpm_params_t struct
 */
void *dpm_create(PyObject *dict)
{
	struct dpm_params_t *params = (
		(struct dpm_params_t *) malloc(sizeof(struct dpm_params_t)));

	PyObject *alpha_py = PyDict_GetItemString(dict, "alpha");
	if((alpha_py == NULL) || !PyFloat_Check(alpha_py)) {
		PyErr_SetString(
			PyExc_KeyError,
			"DPM model requires hyperparameter 'alpha' (python float).");
		return NULL;
	}
	params->alpha = PyFloat_AsDouble(alpha_py);
	return (void *) params;
}


/**
 * Destroy DPM parameters struct
 * @param params struct to destroy
 */
void dpm_destroy(void *params)
{
	free(params);
}


/**
 * Update DPM parameters
 * @param params parameters to update
 * @param update python dictionary containing new values
 */
void dpm_update(void *params, PyObject *update)
{
	((struct dpm_params_t *) params)->alpha = PyFloat_AsDouble(
		PyDict_GetItemString(update, "alpha"));
}


/**
 * Log coefficients for DPM (simply proportional to cluster size)
 * @param params model hyperparameters (just alpha; unused for this)
 * @param size size of cluster
 * @param nc number of clusters
 * @return |c_i|
 */
double dpm_log_coef(void *params, int size, int nc)
{
	return log(size);
}


/**
 * Log coefficients for new cluster
 * @param params model hyperparameters (simply returns log(alpha))
 * @return alpha
 */
double dpm_log_coef_new(void *params, int nc)
{
	return log(((struct dpm_params_t *) params)->alpha);
}


/**
 * Log coefficients for split procedure
 * @param params model hyperparameters
 * @param nc Number of clusters
 * @param n1 Number of points in first proposed split cluster
 * @param n2 Number of points in second proposed split cluster
 * @return P(c_split) / P(c_nosplit)
 */
double dpm_log_split(void *params, int nc, int n1, int n2)
{
	struct dpm_params_t *params_tc = (struct dpm_params_t *) params;
	return (
		log(params_tc->alpha)
		+ lgamma(n1) + lgamma(n2)
		- lgamma(n1 + n2));
}


/**
 * Log coefficients for merge procedure
 * @param params model hyperparameters
 * @param nc Number of clusters
 * @param n1 Number of points in first proposed split cluster
 * @param n2 Number of points in second proposed split cluster
 * @return P(c_merge) / P(c_nomerge)
 */
double dpm_log_merge(void *params, int nc, int n1, int n2)
{
	return -1 * dpm_log_split(params, nc, n1, n2);
}


/**
 * dpm_methods package
 */
ModelMethods DPM_METHODS = {
	&dpm_create,
	&dpm_destroy,
	&dpm_update,
	&dpm_log_coef,
	&dpm_log_coef_new,
	&dpm_log_split,
	&dpm_log_merge
};
