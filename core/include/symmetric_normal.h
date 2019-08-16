/**
 * Symmetric normal components
 */

#ifndef SYMMETRIC_NORMAL_H
#define SYMMETRIC_NORMAL_H

#include <Python.h>
#include "mixture.h"


// Symmetric normal struct
struct sn_component_t {
	// Total vector; dimensions [dim]
	double *total;
	// Number of points
	uint32_t n;
};


// Hyperparameters struct
struct sn_params_t {
	// Dimensions
	uint32_t dim;
	// Scale
	double scale;
	// Scale of all points
	double scale_all;
};

// Component methods
ComponentMethods SYMMETRIC_NORMAL;

#endif
