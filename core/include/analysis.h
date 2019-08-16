/** 
 * MCMC Output Analysis
 */

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <Python.h>

#define DOCSTRING_AGGREGATION_SCORE \
	"Get Aggregation Score: \n" \
	"P[x_i, x_j assigned to same cluster | x_i, x_j in same cluster]\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"actual : np.array\n" \
	"    Actual assignments. Must have type np.uint16." \
	"asn : np.array\n" \
	"    Computed assignments. Must have type np.uint16; must match " \
		"dimensions of 'actual'."

PyObject *aggregation_score_py(PyObject *self, PyObject *args);


#define DOCSTRING_SEGREGATION_SCORE \
	"Get Segregation Score: \n" \
	"P[x_i, x_j assigned to different cluster | x_i, x_j in different " \
	"clusters]\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"actual : np.array\n" \
	"    Actual assignments. Must have type np.uint16." \
	"asn : np.array\n" \
	"    Computed assignments. Must have type np.uint16; must match " \
		"dimensions of 'actual'."

PyObject *segregation_score_py(PyObject *self, PyObject *args);


#define DOCSTRING_ORACLE_MATRIX \
	"Get Oracle Pairwise Probability Matrix:\n" \
	"A[i, j] = P[x_i, x_j assigned to same cluster| Oracle Information]\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"likelihoods : np.array\n" \
	"    Numpy array containing assignment likelihoods for each point. each " \
		"row must have\n" \
	"    likelihoods normalized to 1.\n" \
	"\n" \
	"Returns\n" \
	"-------\n" \
	"np.array\n" \
	"    Pairwise probability matrix"

PyObject *oracle_matrix_py(PyObject *self, PyObject *args);

#endif
