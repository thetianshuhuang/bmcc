
#ifndef SBM_H

#include <stdint.h>
#include <Python.h>
#include "../mixture.h"


struct sbm_component_t {
	// Pointer to config
	struct sbm_params_t *params;
};

struct sbm_params_t {
	// Q array
	double *Q;
	// Number of points
	uint32_t n;
	// Number of clusters
	uint32_t k;
	// SBM Prior parameters
	double alpha;
	double beta;
	// Pointer to assignments
	uint16_t *assignments;
};

ComponentMethods STOCHASTIC_BLOCK_MODEL;

#endif