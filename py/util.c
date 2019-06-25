/**
 * Clustering Utilities
 */

#include <stdint.h>

// Python includes
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


/**
 * Compute pairwise membership matrix
 * @param assignments : cluster assignments for each point
 * @param len : number of data points
 * @return 
 */
void make_pairwise_matrix(uint16_t *assignments, int len, uint8_t *res)
{
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < len; j++) {
            res[i * len + j] = (assignments[i] == assignments[j] ? 1 : 0);
        }
    }
}


/**
 * Compute mean and covariance matrix
 * @param data_py : data points
 * @param mean_py : output array for mean value
 * @param cov_py : output array for covariance matrix
 * @param cluster : PySet object containing indices
 */
void get_cov_by_index(
    PyArrayObject *data_py,
    PyArrayObject *mean_py, PyArrayObject *cov_py,
    PyObject *cluster)
{
    // Set up iterator for PySet
    PyObject *item;
    PyObject *cluster_iter = PyObject_GetIter(cluster);

    // Fetch objects
    double *data = PyArray_DATA(data_py);
    double *mean = PyArray_DATA(mean_py);
    double *cov = PyArray_DATA(cov_py);

    int dim = PyArray_DIM(data_py, 1);
    int n = 0;

    // Calculate sums
    while((item = PyIter_Next(cluster_iter))) {
        // Fetch reference
        long idx = PyLong_AsLong(item);
        n += 1;
        // Add to E[X] and E[XX^T]
        for(int i = 0; i < dim; i++) {
            // Update mean vector E[X]
            mean[i] += data[dim * idx + i];
            // Update covariance vector E[XX^T]
            for(int j = 0; j < dim; j++) {
                cov[dim * i + j] += (data[dim * idx + i] * data[dim * idx + j]);
            }
        }
        // Clean up reference
        Py_DECREF(item);
    }

    // Divide
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) { cov[dim * i + j] /= n; }
        mean[i] /= n;
    }

    // Subtract
    for(int i = 0; i < dim; i++) {
        for(int j = 0; j < dim; j++) { cov[dim * i + j] -= mean[i] * mean[j]; }
    }
}


//
// -- Exposed Functions -------------------------------------------------------
//

#define DOCSTRING_MAKE_PAIRWISE_MATRIX \
    "Compute pairwise assignment matrix for an assignment vector.\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "assignments : np.array\n" \
    "    Assignment vector. Should have data type NPY_UINT8; each unique " \
        "value should\n" \
    "    correspond to a different cluster assignment.\n" \
    "\n" \
    "Returns\n" \
    "-------\n" \
    "np.array\n" \
    "    Pairwise assignment matrix where element (i,j) = 1 if data[i] and " \
        "data[j] are\n" \
    "    in the same cluster."

static PyObject *make_pairwise_matrix_py(PyObject *self, PyObject *args)
{

    // Get args and check for type error
    PyArrayObject *assn;
    // Is np.array
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &assn)) {
        return NULL;
    }
    // 1 dimension
    if(PyArray_NDIM(assn) != 1) {
        PyErr_SetString(
            PyExc_TypeError, "Assignments must have 1 dimension.");
        return NULL;
    }
    // Type=uint8
    if(PyArray_TYPE(assn) != NPY_UINT16) {
        PyErr_SetString(
            PyExc_TypeError, "Assignment vector must have type uint8_t.");
        return NULL;
    }

    // Compute pairwise membership matrix

    // Make numpy array
    int size = PyArray_DIM(assn, 0);
    npy_intp dims[2] = {size, size};
    PyObject *res = PyArray_SimpleNew(2, dims, NPY_UINT8);

    // Compute pairwise membership matrix
    make_pairwise_matrix(
        (uint16_t *) PyArray_DATA(assn), size,
        (uint8_t *) PyArray_DATA((PyArrayObject *) res));

    // Return
    Py_INCREF(res);
    return res;
}


#define DOCSTRING_GET_COV_BY_INDEX \
    "Get covariance matrix from data array indexed by set.\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data : np.array\n" \
    "    Data matrix. Should have type np.float64.\n" \
    "cluster : set[int]\n" \
    "    Set of integers containing indices to compute covariance for.\n" \
    "\n" \
    "Returns\n" \
    "-------\n" \
    "[np.array, np.array]\n" \
    "    [0] Mean vector\n" \
    "    [1] Covariance matrix\n"

static PyObject *get_cov_by_index_py(PyObject *self, PyObject *args)
{
    // Get args and check for type error
    PyArrayObject *data;
    PyObject *cluster;
    // [0] is np.array
    if(!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &data, &cluster)) {
        return NULL;
    }
    // 2 dimensions
    if(PyArray_NDIM(data) != 2) {
        PyErr_SetString(PyExc_TypeError, "Data must have 2 dimensions.");
    }
    // Type=double
    if(PyArray_TYPE(data) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "Data must have type float64");
    }

    // Allocate results
    int size = PyArray_DIM(data, 1);
    npy_intp dims_mean[1] = {size};
    npy_intp dims_cov[2] = {size, size};
    PyObject *mean = PyArray_SimpleNew(1, dims_mean, NPY_FLOAT64);
    PyObject *cov = PyArray_SimpleNew(2, dims_cov, NPY_FLOAT64);

    // Compute
    get_cov_by_index(
        data, (PyArrayObject *) mean, (PyArrayObject *) cov, cluster);

    return Py_BuildValue("OO", mean, cov);

}


//
// -- Python Module Configuration ---------------------------------------------
//

/**
 * ModuleMethods -- Python Module methods list
 */
static PyMethodDef ModuleMethods[] = {
    {
        "pairwise_matrix",
        (PyCFunction) make_pairwise_matrix_py,
        METH_VARARGS,
        DOCSTRING_MAKE_PAIRWISE_MATRIX
    },
    {
        "cov_by_index",
        (PyCFunction) get_cov_by_index_py,
        METH_VARARGS,
        DOCSTRING_GET_COV_BY_INDEX
    },
    {NULL, NULL, 0, NULL}
};

/**
 * ModuleDef -- Python Module Definitions
 */
static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "cluster_util",
    "Utilities for bayesian clustering algorithms",
    -1,
    ModuleMethods
};

/**
 * PyMODINIT_FUNC -- Module initialization
 */
PyMODINIT_FUNC PyInit_cluster_util() {
    import_array();
    return PyModule_Create(&ModuleDef);
}
