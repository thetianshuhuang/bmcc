/** 
 * Stochastic Block Model
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdlib.h>

#include "../include/models/sbm.h"
#include "../include/mixture.h"
#include "../include/type_check.h"
#include "../include/misc_math.h"


// ----------------------------------------------------------------------------
//
//                            Component Management
//
// ----------------------------------------------------------------------------

void *sbm_create(void *params)
{
	// Memory
	struct sbm_component_t *component = (
		(struct sbm_component_t *) malloc(sizeof(struct sbm_component_t)));
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params;
	component->params = params_tc;

	// Number of clusters (alias)
	int k = params_tc->k;
	params_tc->k += 1;

	double *Q_new = malloc(sizeof(double) * (k + 1) * (k + 1));

	// Copy all but new row and column
	for(int i = 0; i < k; i++) {
		for(int j = 0; j < k; j++) {
			Q_new[i * (k + 1) + j] = params_tc->Q[i * k + j];
		}
	}
	// Write in new row and column
	for(int i = 0; i < (k + 1); i++) {
		double rbeta = rand_beta(params_tc->alpha, params_tc->beta);
		Q_new[(k + 1) * (k + 1) + i] = rbeta;
		Q_new[i * (k + 1) + (k + 1)] = rbeta;
	}

	// Swap Q
	free(params_tc->Q);
	params_tc->Q = Q_new;

	return component;
}


void sbm_destroy(Component *component)
{
	struct sbm_params_t *params = (
		((struct sbm_component_t *) component->data)->params);

	int k = params->k;
	params->k -= 1;

	double *Q_new = malloc(sizeof(double) * (k - 1) * (k - 1));

	// Copy all but current index
	for(int i = 0; i < component->idx; i++) {
		for(int j = 0; j < component->idx; j++) {
			Q_new[i * (k - 1) + j] = params->Q[i * (k - 1) + j];
		}
		for(int j = component->idx + 1; j < k; j++) {
			Q_new[i * (k - 1) + j - 1] = params->Q[i * (k - 1) + j];
		}
	}
	for(int i = component->idx + 1; i < k; i++) {
		for(int j = 0; j < component->idx; j++) {
			Q_new[(i - 1) * (k - 1) + j] = params->Q[i * (k - 1) + j];
		}
		for(int j = component->idx + 1; j < k; j++) {
			Q_new[(i - 1) * (k - 1) + j - 1] = params->Q[i * (k - 1) + j];
		}
	}

	// Swap Q
	free(params->Q);
	params->Q = Q_new;

	// Free component
	free(component->data);
}

// ----------------------------------------------------------------------------
//
//                            Parameters Management
//
// ----------------------------------------------------------------------------

/**
 * Create struct sbm_params_t from python dictionary
 */
void *sbm_params_create(PyObject *dict)
{
	// Check Keys
	PyArrayObject *Q_py = (PyArrayObject *) PyDict_GetItemString(dict, "Q");
	PyObject *n_py = (PyObject *) PyDict_GetItemString(dict, "n");
	PyObject *alpha_py = (PyObject *) PyDict_GetItemString(dict, "alpha");
	PyObject *beta_py = (PyObject *) PyDict_GetItemString(dict, "beta");

	bool check = (
		(Q_py != NULL) &&
		(n_py != NULL) && PyLong_Check(n_py) &&
		(alpha_py != NULL) && PyFloat_Check(alpha_py) &&
		(beta_py != NULL) && PyFloat_Check(beta_py)
	);

	if(!check) {
		PyErr_SetString(
			PyExc_KeyError,
			"SBM requires Q array (SPM likelihood array), n (number of "
			"points), alpha, beta (SBM prior parameters).");
		return NULL;
	}

	// Allocate Parameters
	struct sbm_params_t *params = (
		(struct sbm_params_t *) malloc(sizeof(struct sbm_params_t)));

	// Parameters
	params->n = (int) PyLong_AsLong(n_py);
	params->k = PyArray_DIM(Q_py, 0);
	params->alpha = PyFloat_AsDouble(alpha_py);
	params->beta = PyFloat_AsDouble(beta_py);

	// Allocate and copy Q
	double *Q = PyArray_DATA(Q_py);
	params->Q = malloc(sizeof(double) * params->k * params->k);
	for(int i = 0; i < params->k * params->k; i++) { params->Q[i] = Q[i]; }

	// TODO: link assignments

	return (void *) params;
}


/**
 * Destroy struct sbm_params_t
 */
void sbm_params_destroy(void *params)
{
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params_tc;
	free(params_tc->Q);
	free(params_tc);
}


/**
 * Update with new Q array
 */
void sbm_params_update(void *params, PyObject *dict)
{
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params;

	// Q present?
	PyArrayObject *Q_py = (PyArrayObject *) PyDict_GetItemString(dict, "Q");
	if(Q_py != NULL) {
		int k = params_tc->k;

		// Make sure Q is valid
		if(!type_check_square(Q_py, k)) { return; }
		// Copy Q
		double *Q = (double *) PyArray_DATA(Q_py);
		for(int i = 0; i < k * k; i++) { params_tc->Q[i] = Q[i]; }
	}
}


// ----------------------------------------------------------------------------
//
//                     Component Likelihoods & Information
//
// ----------------------------------------------------------------------------

/**
 * Add point to SBM object
 * @param component : component to add
 * @param point : data point
 */
void sbm_add(Component *component, void *params, void *point)
{
	// No update
}


/**
 * Remove point from SBM object
 * @param component : component to remove
 * @param point : data point
 */
void sbm_remove(Component *component, void *params, void *point)
{
	// No update
}


/**
 * Get Log Likelihood of unconditional assignment for new cluster probability
 * m(x_j)
 * @param params : model hyperparameters
 * @param point : data point
 */

double sbm_loglik_new(void *params, void *point)
{
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params;
	uint8_t *point_tc = (uint8_t *) point;

	// Count number of connections to each cluster
	uint32_t *connected = malloc(sizeof(uint32_t) * params_tc->k);
	uint32_t *unconnected = malloc(sizeof(uint32_t) * params_tc->k);
	for(int i = 0; i < params_tc->n; i++) {
		int idx = params_tc->assignments[i];
		if(point_tc[i]) { connected[idx] += 1; }
		else { unconnected[idx] += 1; }
	}

	// Compute loglik
	double loglik = 0;
	for(int i = 0; i < params_tc->k; i++) {
		loglik -= log(rand_beta(
			params_tc->alpha, params_tc->beta));
		loglik += log(rand_beta(
			connected[i] + params_tc->alpha,
			unconnected[i] + params_tc->beta));
	}

	return loglik;
}


/**
 * Get Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
 * @param component : component c
 * @param params : model hyperparameters
 * @param point : data point
 */
double sbm_loglik_ratio(Component *component, void *params, void *point)
{
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params;
	uint8_t *point_tc = (uint8_t *) point;

	double loglik = 0;
	for(int i = 0; i < params_tc->n; i++) {
		double q = params_tc->Q[
			component->idx * params_tc->k + params_tc->assignments[i]];
		if(point_tc[i]) { loglik += log(q); }
		else { loglik += log(1 - q); }
	}
	return loglik;
}


// ----------------------------------------------------------------------------
//
//                                  Utilities
//
// ----------------------------------------------------------------------------

/** 
 * Get Q Array
 * @return PyArrayObject containing *copy* of Q
 */
PyObject *sbm_inspect(void *params) {
	/*
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params;

	PyArrayObject *Q_py = PyArray_SimpleNew(
		2, {params_tc->k, params_tc->k}, NPY_FLOAT64);
	double *Q = PyArray_DATA(Q_py);
	for(int i = 0; i < params_tc->k * params_tc->k; i++) {
		Q[i] = params_tc->Q[i];
	}

	return Q_py;
	*/
	return NULL;
}


ComponentMethods STOCHASTIC_BLOCK_MODEL = {
	// Hyperparameters
	&sbm_params_create,
	&sbm_params_destroy,
	&sbm_params_update,

	// Component Management
	&sbm_create,
	&sbm_destroy,
	&sbm_add,
	&sbm_remove,

	// Component Likelihoods
	&sbm_loglik_ratio,
	&sbm_loglik_new,
	NULL,

	// Debug
	&sbm_inspect
};
