/**
 * Bayesian Clustering Python C API Core Module
 */

#include <Python.h>


// ----------------------------------------------------------------------------
//
//                              Core Numpy Import
//
// ----------------------------------------------------------------------------
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>


// ----------------------------------------------------------------------------
//
//                            Configurable Defines
//
// ----------------------------------------------------------------------------
#ifndef BASE_VEC_SIZE
#define BASE_VEC_SIZE 32
#endif

#ifndef BUILD_DATETIME
#define BUILD_DATETIME "NOT AVAILABLE"
#endif


// ----------------------------------------------------------------------------
//
//                           Capsule API Definitions
//
// ----------------------------------------------------------------------------
#if \
    !defined(COMPONENT_METHODS_API) || \
    !defined(MODEL_METHODS_API) || \
    !defined(MIXTURE_MODEL_API)
#error \
    API Names (COMPONENT_METHODS_API, MODEL_METHODS_API, MIXTURE_MODEL_API) \
    must be #defined on compilation.
#endif


// ----------------------------------------------------------------------------
//
//                               Module Includes
//
// ----------------------------------------------------------------------------

// Core
#include "../include/samplers/gibbs.h"
#include "../include/samplers/temporal_gibbs.h"
#include "../include/samplers/split_merge.h"
#include "../include/mixture/mixture_capsules.h"
#include "../include/cleanup.h"

// Analysis and other non-sampler helpers
#include "../include/select.h"
#include "../include/analysis.h"
#include "../include/type_check.h"
#include "../include/sbm_util.h"

// Mixture Models
#include "../include/models/dpm.h"
#include "../include/models/mfm.h"
#include "../include/models/hybrid.h"

// Component Models
#include "../include/models/normal_wishart.h"
#include "../include/models/symmetric_normal.h"
#include "../include/models/sbm.h"


// ----------------------------------------------------------------------------
//
//                               Module Methods
//
// ----------------------------------------------------------------------------

static PyMethodDef ModuleMethods[] = {
    {
        "gibbs",
        (PyCFunction) gibbs_iter_py,
        METH_VARARGS | METH_KEYWORDS,
        DOCSTRING_GIBBS_ITER
    },
    {
        "temporal_gibbs",
        (PyCFunction) temporal_gibbs_iter_py,
        METH_VARARGS | METH_KEYWORDS,
        DOCSTRING_TEMPORAL_GIBBS_ITER
    },
    {
        "split_merge",
        (PyCFunction) split_merge_py,
        METH_VARARGS | METH_KEYWORDS,
        DOCSTRING_SPLIT_MERGE
    },
    {
        "cleanup_gibbs",
        (PyCFunction) cleanup_iter_py,
        METH_VARARGS | METH_KEYWORDS,
        DOCSTRING_CLEANUP_GIBBS
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
        "inspect_mixture",
        (PyCFunction) inspect_mixture_py,
        METH_VARARGS,
        DOCSTRING_INSPECT_MIXTURE
    },
    {
        "count_clusters",
        (PyCFunction) count_clusters_py,
        METH_VARARGS,
        DOCSTRING_COUNT_CLUSTERS
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
    {
        "oracle_matrix",
        (PyCFunction) oracle_matrix_py,
        METH_VARARGS,
        DOCSTRING_ORACLE_MATRIX
    },
    {
        "sbm_simulate",
        (PyCFunction) sbm_simulate_py,
        METH_VARARGS,
        DOCSTRING_SBM_SIMULATE
    },
    {
        "sbm_update",
        (PyCFunction) sbm_update_py,
        METH_VARARGS,
        DOCSTRING_SBM_UPDATE
    },
    {
        "get_capsule_name",
        (PyCFunction) get_capsule_name_py,
        METH_VARARGS,
        DOCSTRING_GET_CAPSULE_NAME
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "core",
    "C accelerator functions for bayesian clustering algorithms",
    -1,
    ModuleMethods,
    NULL, NULL, NULL, NULL
};


// ----------------------------------------------------------------------------
//
//                           Module Initialization
//
// ----------------------------------------------------------------------------

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
    PyModule_AddObject(
        mod, "MODEL_HYBRID", PyCapsule_New(
            &HYBRID_METHODS, MODEL_METHODS_API, NULL));

    // -- Capsules - Components -----------------------------------------------    
    PyModule_AddObject(
        mod, "COMPONENT_NORMAL_WISHART", PyCapsule_New(
            &NORMAL_WISHART, COMPONENT_METHODS_API, NULL));
    PyModule_AddObject(
        mod, "COMPONENT_SYMMETRIC_NORMAL", PyCapsule_New(
            &SYMMETRIC_NORMAL, COMPONENT_METHODS_API, NULL));
    PyModule_AddObject(
        mod, "COMPONENT_STOCHASTIC_BLOCK_MODEL", PyCapsule_New(
            &STOCHASTIC_BLOCK_MODEL, COMPONENT_METHODS_API, NULL));

    // -- Build Constants & Metadata ------------------------------------------
    PyModule_AddIntMacro(mod, BASE_VEC_SIZE);
    PyModule_AddStringMacro(mod, BUILD_DATETIME);

    // -- Module Python C Capsule API Identifiers -----------------------------
    PyModule_AddStringMacro(mod, COMPONENT_METHODS_API);
    PyModule_AddStringMacro(mod, MODEL_METHODS_API);
    PyModule_AddStringMacro(mod, MIXTURE_MODEL_API);

    return mod;
}
