/**
 * Create membership matrix
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
void make_pairwise_matrix(uint8_t *assignments, int len, uint8_t *res)
{
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < len; j++) {
            res[i * len + j] = (assignments[i] == assignments[j] ? 1 : 0);
        }
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
    if(PyArray_TYPE(assn) != NPY_UINT8) {
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
        (uint8_t *) PyArray_DATA(assn), size,
        (uint8_t *) PyArray_DATA((PyArrayObject *) res));

    // Return
    Py_INCREF(res);
    return res;
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
    {NULL, NULL, 0, NULL}
};

/**
 * ModuleDef -- Python Module Definitions
 */
static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "clustering_util",
    "Utilities for bayesian clustering algorithms",
    -1,
    ModuleMethods
};

/**
 * PyMODINIT_FUNC -- Module initialization
 */
PyMODINIT_FUNC PyInit_clustering_util() {
    import_array();
    return PyModule_Create(&ModuleDef);
}
