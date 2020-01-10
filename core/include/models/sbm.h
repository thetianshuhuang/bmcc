/** 
 * Stochastic Block Model
 */

#ifndef SBM_H

#include <stdint.h>
#include <Python.h>
#include "../include/mixture/mixture.h"


// SBM component struct
// Literally empty, except for pointer to parameters
struct sbm_component_t {
	// Pointer to config
	struct sbm_params_t *params;
};

// Hyperparameters struct
struct sbm_params_t {
	// Q array
	double *Q;
	// Number of points
	uint32_t n;
	// Number of clusters
	uint32_t k;
	// SBM Prior parameters
	double a;
	double b;
	// Pointer to assignment array (for refcounting)
	PyArrayObject *assignments;
	// Pointer to data array (for refcounting)
	PyArrayObject *data;
};

// Component methods; only the struct is exposed!
ComponentMethods STOCHASTIC_BLOCK_MODEL;

#endif
