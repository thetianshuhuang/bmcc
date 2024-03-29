/**
 * Identity-distributed multivariate gaussian components
 */

#include <Python.h>

#include <stdint.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../include/models/symmetric_normal.h"
#include "../include/mixture/mixture.h"


// ----------------------------------------------------------------------------
//
//                            Component Management
//
// ----------------------------------------------------------------------------

/**
 * Create symmetric normal object
 * @param params : model hyperparameters
 * @return Allocated component structure
 */
void *sn_create(void *params)
{
	struct sn_component_t *component = (
		(struct sn_component_t *) malloc(sizeof(struct sn_component_t)));

	struct sn_params_t *params_tc = (struct sn_params_t *) params;
	uint32_t dim = params_tc->dim;

	// Total = 0
	component->total = (double *) malloc(sizeof(double) * dim);
	for(uint32_t i = 0; i < dim; i++) { component->total[i] = 0; }

	return component;
}


/**
 * Destroy symmetric normal object
 */
void sn_destroy(Component *component)
{
	free(((struct sn_component_t *) component->data)->total);
	free(component->data);
}


// ----------------------------------------------------------------------------
//
//                            Parameters Management
//
// ----------------------------------------------------------------------------

/**
 * Create struct sn_params_t from python dictionary
 */
void *sn_params_create(PyObject *dict)
{
	// Check keys
	PyObject *dim_py = PyDict_GetItemString(dict, "dim");
	PyObject *scale_py = PyDict_GetItemString(dict, "scale");
	PyObject *scale_all_py = PyDict_GetItemString(dict, "scale_all");
	if((dim_py == NULL) || (scale_py == NULL) || (scale_all_py == NULL) ||
			(!PyFloat_Check(scale_py)) ||
			(!PyLong_Check(dim_py)) ||
			(!PyFloat_Check(scale_all_py))) {
		PyErr_SetString(
			PyExc_KeyError,
			"Symmetric normal requires 'scale' (symmetric normal scale; "
			"float), 'scale_all' (aggregate scale for all points; float), "
			"and 'dim' (data dimensions; int) arguments.");
		return NULL;
	}

	// Unpack dict
	uint32_t dim = (int) PyLong_AsLong(dim_py);
	double scale = PyFloat_AsDouble(scale_py);
	double scale_all = PyFloat_AsDouble(scale_all_py);

	// Allocate parameters
	struct sn_params_t *params = (
		(struct sn_params_t *) malloc(sizeof(struct sn_params_t)));
	params->dim = dim;
	params->scale = scale;
	params->scale_all = scale_all;

	return (void *) params;
}


/**
 * Destroy struct sn_params_t
 */
void sn_params_destroy(void *params)
{
	free(params);
}


// ----------------------------------------------------------------------------
//
//                     Component Likelihoods & Information
//
// ----------------------------------------------------------------------------


/**
 * Add point to symmetric normal object
 * @param component : component to add
 * @param point : data point
 */
void sn_add(Component *component, void *params, void *point)
{
    struct sn_component_t *comp_tc = (struct sn_component_t *) component->data;
    struct sn_params_t *params_tc = (struct sn_params_t *) params;

    // Update mean
    for(uint32_t i = 0; i < params_tc->dim; i++) {
    	comp_tc->total[i] += ((double *) point)[i];
    }
}


/**
 * Remove point from symmetric normal object
 * @param component : component to remove
 * @param point : data point
 */
void sn_remove(Component *component, void *params, void *point)
{
    struct sn_component_t *comp_tc = (struct sn_component_t *) component->data;
    struct sn_params_t *params_tc = (struct sn_params_t *) params;

    // Downdate mean
    for(uint32_t i = 0; i < params_tc->dim; i++) {
    	comp_tc->total[i] -= ((double *) point)[i];
    }
}


/**
 * Get Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
 * @param component : component c
 * @param params : model hyperparameters
 * @param point : data point
 */
double sn_loglik_ratio(Component *component, void *params, void *point)
{
	struct sn_component_t *cpt = (struct sn_component_t *) component->data;
	struct sn_params_t *params_tc = (struct sn_params_t *) params;

	uint32_t dim = params_tc->dim;
	double scale = params_tc->scale;

	// (x-mu)^T(x-mu)
	double acc = 0;
	for(uint32_t i = 0; i < dim; i++) {
		double centered = (
			((double *) point)[i] - (cpt->total[i] / component->size));
		acc += centered * centered;
	}

	// -k/2 log(2pi) - 0.5 dim * log(scale) - 0.5 (x-mu)^T(x-mu)
	return (
		- log(2 * M_PI) * (dim / 2)
		- 0.5 * dim * log(scale)
		- 0.5 * acc / scale);
}


/**
 * Get Log Likelihood of unconditional assignment for new cluster probability
 * m(x_j)
 * Assumes points are centered.
 * @param params : model hyperparameters
 * @param point : data point
 */
double sn_loglik_new(void *params, void *point)
{
	struct sn_params_t *params_tc = (struct sn_params_t *) params;

	uint32_t dim = params_tc->dim;
	double scale_all = params_tc->scale_all;

	// (x-0)^T(x-0)
	double acc = 0;
	for(uint32_t i = 0; i < dim; i++) {
		acc += ((double *) point)[i] * ((double *) point)[i];
	}

	return (
		- log(2 * M_PI) * (dim / 2)
		- 0.5 * dim * log(scale_all)
		- 0.5 * acc / scale_all);
}


/**
 * Extern for symmetric_normal methods
 */
ComponentMethods SYMMETRIC_NORMAL = {
	// Hyperparameters
	&sn_params_create,
	&sn_params_destroy,
	NULL,  	// No update

	// Component Management
	&sn_create,
	&sn_destroy,
	&sn_add,
	&sn_remove,

	// Component Likelihoods
	&sn_loglik_ratio,
	&sn_loglik_new,
	NULL,  	// Does not support split merge

	// Debug
	NULL,	// Nothing to inspect
};
