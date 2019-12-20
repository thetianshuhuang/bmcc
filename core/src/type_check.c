/**
 * Routines to check Numpy Array types.
 *  - Assignment arrays are uint16.
 *    (It's assumed that there will be <<65536 clusters)
 *  - Data arrays have some unspecified data type.
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

#include "../include/mixture.h"


/**
 * Utility function to get size of numpy type
 * @param type : numpy enumerated type
 * @return int : size of type
 */
int type_get_size(int type) {
    if(
        (type == NPY_INT8) ||
        (type == NPY_UINT8)) { return 1; }
    if(
        (type == NPY_INT16) ||
        (type == NPY_UINT16) ||
        (type == NPY_FLOAT16)) { return 2; }
    if(
        (type == NPY_INT32) ||
        (type == NPY_UINT32) ||
        (type == NPY_FLOAT32)) { return 4; }
    if(
        (type == NPY_INT64) ||
        (type == NPY_UINT64) ||
        (type == NPY_FLOAT64)) { return 8; }
    printf("Type not recognized.\n");
    return 4;
}


/**
 * Run type checks on data and assignment arrays
 * @param data_py : data numpy array
 * @param assignments_py : assignments numpy array
 * @return true if types and dimensions are correct
 */  
bool type_check(
    PyArrayObject *data_py, PyArrayObject *assignments_py)
{
    // Check types
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


/**
 * Check that models support gibbs
 */
bool supports_gibbs(struct mixture_model_t *model)
{
    // Mixture Methods
    bool mixture_is_gibbs = (
        (model->model_methods->log_coef != NULL) &&
        (model->model_methods->log_coef_new != NULL));
    if(!mixture_is_gibbs) {
        PyErr_SetString(
            PyExc_TypeError,
            "Selected mixture model does not support gibbs sampling (log_coef "
            "and log_coef_new methods must not be NULL)");
        return false;
    }

    // Component Methods
    bool component_is_gibbs = (
        (model->comp_methods->loglik_ratio != NULL) &&
        (model->comp_methods->loglik_new != NULL));
    if(!component_is_gibbs) {
        PyErr_SetString(
            PyExc_TypeError,
            "Selected component model does not support gibbs sampling "
            "(loglik_ratio and loglik_new methods must not be NULL)");
        return false;
    }

    return true;
}


/**
 * Check that models support split merge
 */
bool supports_split_merge(struct mixture_model_t *model)
{
    // Mixture methods
    bool mixture_is_sm = (
        (model->model_methods->log_split != NULL) &&
        (model->model_methods->log_merge != NULL));
    if(!mixture_is_sm) {
        PyErr_SetString(
            PyExc_TypeError,
            "Selected mixture model does not support split merge sampling "
            "(log_split and log_merge methods must not be NULL)");
        return false;
    }

    // Component methods
    bool component_is_sm = (
        (model->comp_methods->split_merge != NULL));
    if(!component_is_sm) {
        PyErr_SetString(
            PyExc_TypeError,
            "Selected component model does not support split merge sampling "
            "(split_merge method must not be NULL)");
        return false;
    }

    return true;
}


/**
 * Get Capsule Name
 */
PyObject *get_capsule_name_py(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    if(!PyArg_ParseTuple(args, "O", &capsule)) { return NULL; }

    const char *name = PyCapsule_GetName(capsule);
    return Py_BuildValue("s", name);
}
