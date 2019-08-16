/**
 *
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>


/**
 * Compute Pairwise Probability Matrix
 * @param mat buffer to write result to
 * @param hist assignment history array; row-major
 * @param size number of data points
 * @param iterations number of iterations
 */
void pairwise_probability_matrix(
    double *mat, uint16_t *hist, int size, int iterations, int burn_in)
{
    // Clear buffer
    for(int i = 0; i < size * size; i++) { mat[i] = 0; }

    // Count up values
    for(int idx = burn_in; idx < iterations; idx++) {
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                if(hist[idx * size + i] == hist[idx * size + j]) {
                    mat[i * size + j] += 1;
                }
            }
        }
    }

    // Divide
    for(int i = 0; i < size * size; i++) {
        mat[i] = mat[i] / (iterations - burn_in);
    }
}


/**
 * Get residual between a probability matrix and an assignment vector.
 * @param prob : pairwise probability matrix P
 * @param asn : assignment vector A
 * @param size : number of data points
 * @param res : ||P - M(A)||_F^2 where M(A) is the membership matrix induced
 *      by A.
 */
double membership_residual(double *prob, uint16_t *asn, int size)
{
    double res = 0;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            double pairwise = (asn[i] == asn[j] ? 1 : 0);
            double diff = prob[i * size + j] - pairwise;
            res += diff * diff;
        }
    }
    return res;
}


/** 
 * Get pairwise probability matrix and residuals
 */
PyObject *pairwise_probability_py(PyObject *self, PyObject *args)
{
    PyArrayObject *hist_py;
    int burn_in;
    if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &hist_py, &burn_in)) {
        return NULL;
    }
    bool pass = (
        PyArray_Check(hist_py) &&
        (PyArray_TYPE(hist_py) == NPY_UINT16) &&
        (PyArray_NDIM(hist_py) == 2));
    if(!pass) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignment history must be a numpy array with 2 dimensions, " \
            "type uint16.");
        return NULL;
    }

    int size = PyArray_DIM(hist_py, 1);
    int iterations = PyArray_DIM(hist_py, 0);
    uint16_t *hist = PyArray_DATA(hist_py);

    // Create pairwise probability matrix
    npy_intp dims[2] = {size, size};
    PyArrayObject *prob_matrix_py = (
        (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_FLOAT64));
    double *prob_matrix = (double *) PyArray_DATA(prob_matrix_py);
    pairwise_probability_matrix(prob_matrix, hist, size, iterations, burn_in);

    // Compute least squares array
    npy_intp dims_res[1] = {iterations};
    PyArrayObject *residuals_py = (
        (PyArrayObject *) PyArray_SimpleNew(1, dims_res, NPY_FLOAT64));
    double *residuals = (double *) PyArray_DATA(residuals_py);
    for(int idx = 0; idx < iterations; idx++) {
        residuals[idx] = membership_residual(
            prob_matrix, &hist[size * idx], size);
    }

    return Py_BuildValue("OO", prob_matrix_py, residuals_py);
}
