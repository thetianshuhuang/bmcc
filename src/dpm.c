/**
 * DPM methods
 */

#include <math.h>


/**
 * DPM Parameters
 */
struct dpm_params_t {
	double alpha;
}


/**
 * Create dpm parameters struct.
 * @param dict Python dictionary containing hyperparameter alpha
 * @return allocated dpm_params_t struct
 */
void *dpm_create(PyObject *dict) {
	struct dpm_params_t *params = (
		(struct dpm_params_t *) malloc(sizeof(struct dp_params_t)));
	params->alpha = PyFloat_AsDouble(PyDict_GetItemString(dict, "alpha"));
	return params;
}


/**
 * Destroy dpm parameters struct
 * @param params struct to destroy
 */
void dpm_destroy(void *params) {
	free(params);
}


/**
 * Log coefficients for DPM (simply proportional to cluster size)
 * @param params model hyperparameters (just alpha; unused for this)
 * @param size size of cluster
 * @return |c_i|
 */
double (*dpm_log_coef)(void *params, int size) {
	return log(size);
}


/**
 * Log coefficients for new cluster
 * @param params model hyperparameters (simply returns log(alpha))
 * @return alpha
 */
double (*dpm_log_coef_new)(void *params) {
	return log(params->alpha);
}


/**
 * dpm_methods package
 */
const ModelMethods DPM_METHODS {
	&dpm_create_params,
	&dpm_destroy_params,
	&dpm_log_coef,
	&dpm_log_coef_new
}
