

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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
    if(PyArray_TYPE(data_py) != NPY_FLOAT32) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data must have type float32.");
        return false;
    }
    if(PyArray_TYPE(assignments_py) != NPY_UINT16) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignments must have type uint16.");
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
