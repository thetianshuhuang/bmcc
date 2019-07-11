/**
 *
 */


#ifndef SELECT_H
#define SELECT_H

#include <Python.h>


#define DOCSTRING_PAIRWISE_PROBABILITY \
	"Get Pairwise Probability Matrix and resulting Membership Matrix " \
		"residuals.\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"hist : np.array\n" \
	"    Assignment history array; row-major, uint16.\n" \
	"burn_in : int\n" \
	"    Burn in duration; must be larger than the number of samples.\n" \
	"\n" \
	"Returns\n" \
	"-------\n" \
	"(np.array, np.array)\n" \
	"    [0] : Pairwise Probability Matrix; dim = [N * N]\n" \
	"    [1] : Residuals; dim = [N]"

PyObject *pairwise_probability_py(PyObject *self, PyObject *args);

#endif
