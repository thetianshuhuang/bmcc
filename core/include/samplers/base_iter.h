/** 
 * Base Python MCMC iteration wrapper
 */

#ifndef BASE_ITER_H
#define BASE_ITER_H

#include <Python.h>

#include "../include/mixture/mixture.h"

PyObject *base_iter(
	PyObject *self, PyObject *args, PyObject *kwargs,
	bool (*error_check)(struct mixture_model_t *),
	bool (*iter)(void *, uint16_t *, struct mixture_model_t *, double));

#endif
