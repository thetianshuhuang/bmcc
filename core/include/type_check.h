/**
 * Routines to check Numpy Array types and model support
 *  - Assignment arrays are uint16.
 *    (It's assumed that there will be <<65536 clusters)
 *  - Data arrays are float64 (double).
 *  - Data arrays must be C-style contiguous (arr[y][x] = y * xdim + x)
 *  - PyArray_FLAGS (not documented in numpy C api) -- each flag set is a
 *    single bit. Presence of the bit indicates pass (flags & NPY_ARRAY_...)
 *  - Data array should have the same first dimension as the assignment array
 */

#ifndef TYPE_CHECK_H
#define TYPE_CHECK_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdbool.h>


// Check type of data and assignments
bool type_check(PyArrayObject *data_py, PyArrayObject *assignments_py);
// Check type of data against target dimensions
bool type_check_square(PyArrayObject *data_py, int dim);
// Type check two different assignment arrays
bool type_check_assignments(PyArrayObject *arr1, PyArrayObject *arr2);
// Check for gibbs support
bool supports_gibbs(struct mixture_model_t *model);
// Check for split merge support
bool supports_split_merge(struct mixture_model_t *model);


#define DOCSTRING_GET_CAPSULE_NAME \
	"Get name of python capsule.\n" \
	"\n" \
	"Parameters\n" \
	"----------\n" \
	"capsule : Python Capsule\n" \
	"    Capsule to fetch name of\n" \
	"\n" \
	"Returns\n" \
	"-------\n" \
	"str or None\n" \
	"    str if object is capsule and has a name. None on error."

PyObject *get_capsule_name_py(PyObject *self, PyObject *args);

#endif
