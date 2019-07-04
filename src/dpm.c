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
	return (void *) params;
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
 * @param nc number of clusters
 * @return |c_i|
 */
double dpm_log_coef(void *params, int size, int nc) {
	return log(size);
}


/**
 * Log coefficients for new cluster
 * @param params model hyperparameters (simply returns log(alpha))
 * @return alpha
 */
double dpm_log_coef_new(void *params, int nc) {
	return log(((struct dpm_params_t *) params)->alpha);
}


/**
 * dpm_methods package
 */
const ModelMethods DPM_METHODS {
	&dpm_create,
	&dpm_destroy,
	&dpm_log_coef,
	&dpm_log_coef_new
}
