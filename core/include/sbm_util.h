/**
 * SBM utility functions
 */

#ifndef SBM_SIMULATE_H
#define SBM_SIMULATE_H

#include <Python.h>


// Update SBM (C function; wrapped by sbm_update_py)
double *sbm_update(
	uint8_t *data, uint16_t *asn, int n, int k, double a, double b);


#define DOCSTRING_SBM_SIMULATE \
	"Generate simulated SBM data (C-accelerated)\n" \
	"\n" \
	"For each pair of points (i,j) with clusters (r,s), populates A[i,j] " \
		"with an \n" \
	"independent Bernoulli(Q[r,s]) random variable.\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"Q : np.array\n" \
	"    Q array to generate SBM from\n" \
	"A : np.array\n" \
	"    Output array. Supply with np.zeros() or similar.\n" \
	"assignments : np.array\n" \
	"    Assignment array. Should be pre-generated; is not modified."

PyObject *sbm_simulate_py(PyObject *self, PyObject *args);

#define DOCSTRING_SBM_UPDATE \
	"Sample new Q array for SBM\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"data : np.array\n" \
	"    Data array. Must have type uint8.\n" \
	"assignments : np.array\n" \
	"    Assignment array. Must have type uint16.\n" \
	"k : int\n" \
	"    Number of clusters\n" \
	"a, b : float\n" \
	"    Q beta prior parameters\n" \
	"\n" \
	"Returns\n" \
	"-------\n" \
	"np.array\n" \
	"    Resampled Q array; dimensions (k,k) and data type float64."

PyObject *sbm_update_py(PyObject *self, PyObject *args);

#endif
