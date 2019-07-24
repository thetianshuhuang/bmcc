/**
 * Hybrid DPM, MFM model
 */

#include <Python.h>
#include <stdbool.h>

#include "../include/mixture.h"
#include "../include/dpm.h"
#include "../include/mfm.h"

/**
 * Parameters
 */
struct hybrid_params_t {
	void *dpm;
	void *mfm;
	bool is_mfm;
}


/**
 * Create hybrid parameters struct.
 * @param dict Python dictionary containing DPM and MFM hyperparameters
 * @return allocated hybrid_params_t struct
 */
void *hybrid_create(PyObject *dict)
{
	struct hybrid_params_t *params = (
		(struct hybrid_params_t *) malloc(sizeof(struct hybrid_params_t)));

	params->dpm = DPM_METHODS->create(dict);
	params->mfm = MFM_METHODS->create(dict);
	params->is_mfm = (bool) PyObject_IsTrue(
		PyDict_GetItemString(dict, "is_mfm"));

	return (void *) params;
}


/**
 * Destroy hybrid parameters struct
 * @param params struct to destroy
 */
void hybrid_destroy(void *params)
{
	DPM_METHODS->destroy(params->dpm);
	MFM_METHODS->destroy(params->mfm);
	free(params);
}


/**
 * Update hybrid parameters
 */
void hybrid_update(void *params, PyObject *update) 
{
	struct hybrid_params_t *params_tc = (struct hybrid_params_t *) params;

	DPM_METHODS->update(params->dpm, update);
	MFM_METHODS->update(params->mfm, update);
	params->bool = (bool) PyObject_IsTrue(
		PyDict_GetItemString(dict, "is_mfm"));
}


/**
 * Log coefficients for hybrid model
 */
void hybrid_log_coef(void *params, int size, int nc)
{
	if(params->is_mfm) {
		return MFM_METHODS->log_coef(params->mfm, size, nc);
	}
	else {
		return DPM_METHODS->log_coef(params->dpm, size, nc);
	}
}


/**
 * Log coefficient for new cluster
 */
void hybrid_log_coef_new(void *params, int nc)
{
	if(params->is_mfm) {
		return MFM_METHODS->log_coef_new(params, nc);
	}
	else {
		return DPM_METHODS->log_coef(params, nc);
	}
}


/**
 * hybrid_methods package
 */
ModelMethods HYBRID_METHODS = {
	&hybrid_create,
	&hybrid_destroy,
	&hybrid_update,
	&hybrid_log_coef,
	&hybrid_log_coef_new
}
