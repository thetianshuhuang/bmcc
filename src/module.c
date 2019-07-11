/**
 * Bayesian Clustering Python C API Core Module
 */

#include <Python.h>

/**
 * Core Numpy Import
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#ifndef BASE_VEC_SIZE
#define BASE_VEC_SIZE 32
#endif

#include "../include/gibbs.h"
#include "../include/dpm.h"
#include "../include/mfm.h"
#include "../include/normal_wishart.h"
#include "../include/mixture.h"
#include "../include/select.h"
#include "../include/analysis.h"


/**
 * Module Methods
 */
static PyMethodDef ModuleMethods[] = {
    {
        "gibbs_iter",
        (PyCFunction) gibbs_iter_py,
        METH_VARARGS,
        DOCSTRING_GIBBS_ITER
    },
    {
        "init_model",
        (PyCFunction) init_model_capsules_py,
        METH_VARARGS,
        DOCSTRING_INIT_MODEL_CAPSULES
    },
    {
        "update_mixture",
        (PyCFunction) update_mixture_py,
        METH_VARARGS,
        DOCSTRING_UPDATE_MIXTURE
    },
    {
        "update_components",
        (PyCFunction) update_components_py,
        METH_VARARGS,
        DOCSTRING_UPDATE_COMPONENTS
    },
    {
        "pairwise_probability",
        (PyCFunction) pairwise_probability_py,
        METH_VARARGS,
        DOCSTRING_PAIRWISE_PROBABILITY
    },
    {
        "aggregation_score",
        (PyCFunction) aggregation_score_py,
        METH_VARARGS,
        DOCSTRING_AGGREGATION_SCORE
    },
    {
        "segregation_score",
        (PyCFunction) segregation_score_py,
        METH_VARARGS,
        DOCSTRING_SEGREGATION_SCORE
    },
    {NULL, NULL, 0, NULL}
};


/**
 * Module Definitions
 */
static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "core",
    "C accelerator functions for bayesian clustering algorithms",
    -1,
    ModuleMethods
};


/**
 * Module Initialization
 * Capsules are loaded separately from function definitions
 */
PyMODINIT_FUNC PyInit_core()
{
    import_array();

    PyObject *mod = PyModule_Create(&ModuleDef);
    if(mod == NULL) { return NULL; }

    // -- Capsules - Models ---------------------------------------------------
    PyModule_AddObject(
        mod, "MODEL_DPM", PyCapsule_New(
            &DPM_METHODS, MODEL_METHODS_API, NULL));
    PyModule_AddObject(
        mod, "MODEL_MFM", PyCapsule_New(
            &MFM_METHODS, MODEL_METHODS_API, NULL));

    // -- Capsules - Components -----------------------------------------------    
    PyModule_AddObject(
        mod, "COMPONENT_NORMAL_WISHART", PyCapsule_New(
            &NORMAL_WISHART, COMPONENT_METHODS_API, NULL));

    // -- Module Constants ----------------------------------------------------
    PyModule_AddIntConstant(
        mod, "BASE_VEC_SIZE", BASE_VEC_SIZE);
    PyModule_AddStringConstant(
        mod, "COMPONENT_METHODS_API", COMPONENT_METHODS_API);
    PyModule_AddStringConstant(
        mod, "COMPONENT_PARAMS_API", COMPONENT_PARAMS_API);
    PyModule_AddStringConstant(
        mod, "MODEL_METHODS_API", MODEL_METHODS_API);
    PyModule_AddStringConstant(
        mod, "MODEL_PARAMS_API", MODEL_PARAMS_API);
    PyModule_AddStringConstant(
        mod, "MIXTURE_MODEL_API", MIXTURE_MODEL_API);

    return mod;
}

