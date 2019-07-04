/**
 * MFM methods
 */

#include <Python.h>
#include <math.h>


/**
 * MFM parameters
 */
struct mfm_params_t {
	double gamma;
	double *v_n;
}


/**
 * Create MFM parameters struct.
 * @param dict Python dictionary containing hyperparameters. Prior on K should
 * 		be precomputed and used to compute log(V_n(t)) in Python, before
 *		passing the result into C.
 * @return allocated mfm_params_t struct.
 */
void *mfm_create(PyObject *dict) {

	// Unpack dictionary; fetch V_n numpy array
	PyArrayObject *log_vn_py = PyDict_GetItemString(dict, "V_n");
	double *log_vn = PyArray_DATA(log_vn_py);
	int size = PyArray_DIM(log_vn_py, 0)

	// Allocate memory
	struct mfm_params_t *params = (
		(struct mfm_params_t) *) malloc(sizeof(struct mfm_params_t));
	params->v_n = (double *) malloc(sizeof(double) * size);

	// Copy V_n(t)
	for(int i = 0; i < size; i++) { params->v_n[i] = log_vn[i]; }

	// Set gamma
	params->gamma = PyFloat_AsDouble(PyDict_GetItemString(dict, "gamma"));

	return params;
}


/**
 * Destroy MFM parameters
 * @param params struct to destroy
 */
void mfm_destroy(void *params) {
	free(params->v_n);
	free(params);
}


/**
 * Log coefficients for MFM: |c_i| + gamma
 * @param params MFM hyperparameters
 * @param size size of cluster
 * @param nc number of clusters
 * @return log(|c_i| + gamma)
 */
void mfm_log_coef(void *params, int size, int nc) {
	return log(size + ((struct mfm_params_t *) params)->gamma);
}


/**
 * Log coefficients for new cluster: gamma * V_n(t + 1) / V_n(t)
 * @param params MFM hyperparameters
 * @param nc number of clusters
 * @return log(gamma) + log(V_n(t + 1)) - log(V_n(t))
 */
void mfm_log_coef_new(void *params, int nc) {
	struct mfm_params_t *params_tc = (struct mfm_params_t *) params;
	return log(params_tc->gamma) + params_tc->v_n[nc + 1] - params_tc->v_n[nc];
}


/**
 * mfm_methods package
 */
const ModelMethods MFM_METHODS {
	&mfm_create,
	&mfm_destroy,
	&mfm_log_coef,
	&mfm_log_coef_new
}
