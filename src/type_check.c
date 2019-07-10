/**
 * Routines to check Numpy Array types.
 *  - Assignment arrays are uint16.
 *    (It's assumed that there will be <<65536 clusters)
 *  - Data arrays are float64 (double).
 *  - Data arrays must be C-style contiguous (arr[y][x] = y * xdim + x)
 *  - PyArray_FLAGS (not documented in numpy C api) -- each flag set is a
 *    single bit. Presence of the bit indicates pass (flags & NPY_ARRAY_...)
 *  - Data array should have the same first dimension as the assignment array
 */

#include <stdbool.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>


/**
 * Run type checks on data and assignment arrays
 * @param data_py : data numpy array
 * @param assignments_py : assignments numpy array
 * @return true if types and dimensions are correct
 */  
bool type_check(PyArrayObject *data_py, PyArrayObject *assignments_py)
{
    // Check types
    if(PyArray_TYPE(data_py) != NPY_FLOAT64) {
        PyErr_SetString(
            PyExc_TypeError, "Data must have type float64 (double).");
        return false;
    }
    if(PyArray_TYPE(assignments_py) != NPY_UINT16) {
        PyErr_SetString(
            PyExc_TypeError, "Assignments must have type uint16.");
        return false;
    }

    // Check contiguous
    if(!(PyArray_FLAGS(data_py) & NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data must be a contiguous C-style array (np.ascontiguousarray)");
        return false;
    }
    if(!(PyArray_FLAGS(assignments_py) & NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignmnets must be a contiguous C-style array "
            "(np.ascontiguousarray)");
        return false;
    }

    // Check dimensions
    if(PyArray_NDIM(data_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError, "Data must have 2 dimensions.");
        return false;
    }
    if(PyArray_DIM(data_py, 0) != PyArray_DIM(assignments_py, 0)) {
        PyErr_SetString(
            PyExc_TypeError, "Assignments vector length does not match data.");
        return false;
    }

    return true;
}


/**
 * Run type checks on square matrices (such as covariance matrices)
 * @param data_py : numpy array to check
 * @param dim : target dimensions
 * @return true if type is float64, is square, and matches dim
 */
bool type_check_square(PyArrayObject *data_py, int dim)
{
    // Check type
    if(PyArray_TYPE(data_py) != NPY_FLOAT64) {
        PyErr_SetString(
            PyExc_TypeError, "Array must have type float64 (double).");
        return false;
    }

    // Check contiguous
    if(!(PyArray_FLAGS(data_py) & NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Array must be a contiguous C-style array (np.ascontiguousarray)");
        return false;
    }

    // Check size
    if(PyArray_NDIM(data_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError, "Array must have 2 dimensions.");
        return false;        
    }
    if(PyArray_DIM(data_py, 0) != PyArray_DIM(data_py, 1)) {
        PyErr_SetString(
            PyExc_TypeError, "Array must be square.");
        return false;
    }
    if(PyArray_DIM(data_py, 0) != dim) {
        PyErr_SetString(
            PyExc_TypeError, "Array does not match data dimensions.");
        return false;
    }
    return true;
}


/**
 * Run type checks on assignment vectors
 * @param arr1 : first array
 * @param arr2 : second array
 * @return true if arr1, arr2 have same shape, 1 dimension, have type np.uint16
 */ 
bool type_check_assignments(PyArrayObject *arr1, PyArrayObject *arr2)
{
    if((PyArray_NDIM(arr1) != 1) || (PyArray_NDIM(arr2) != 1)) {
        PyErr_SetString(
            PyExc_TypeError, "Vectors must have 1 dimension.");
        return false;
    }
    if(PyArray_DIM(arr1, 0) != PyArray_DIM(arr2, 0)) {
        PyErr_SetString(
            PyExc_TypeError, "Vectors have different length.");
        return false;
    }
    if((PyArray_TYPE(arr1) != NPY_UINT16) || (PyArray_TYPE(arr2) != NPY_UINT16)) {
        PyErr_SetString(
            PyExc_TypeError, "Vectors must have type uint16.");
        return false;
    }
    return true;
}
