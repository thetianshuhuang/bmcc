
#ifndef SBM_SIMULATE_H
#define SBM_SIMULATE_H

#include <Python.h>

double *sbm_update(
	uint8_t *data, uint16_t *asn, int n, int k, double alpha, double beta);


#define DOCSTRING_SBM_SIMULATE \
	"todo"

PyObject *sbm_simulate_py(PyObject *self, PyObject *args);

#define DOCSTRING_SBM_UPDATE \
	"todo"

PyObject *sbm_update_py(PyObject *self, PyObject *args);

#endif
