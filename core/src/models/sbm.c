/** 
 * Stochastic Block Model
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef ASSERT
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "../include/models/sbm.h"
#include "../include/mixture/mixture.h"
#include "../include/type_check.h"
#include "../include/sbm_util.h"
#include "../include/misc_math.h"


// ----------------------------------------------------------------------------
//
//                            Component Management
//
// ----------------------------------------------------------------------------

/**
 * Create SBM object
 * @param params : model hyperparameters
 * @return Allocated component structure
 */
void *sbm_create(void *params)
{
    // Memory
    struct sbm_component_t *component = (
        (struct sbm_component_t *) malloc(sizeof(struct sbm_component_t)));
    struct sbm_params_t *params_tc = (struct sbm_params_t *) params;
    component->params = params_tc;

    params_tc->k += 1;
    double *Q_new = sbm_update(
        PyArray_DATA(params_tc->data),
        PyArray_DATA(params_tc->assignments),
        params_tc->n, params_tc->k, params_tc->a, params_tc->b);

    // Swap Q
    free(params_tc->Q);
    params_tc->Q = Q_new;

    return component;
}


/**
 * Destroy SBM object
 * @param component : component to destroy
 */
void sbm_destroy(Component *component)
{
    struct sbm_params_t *params = (
        ((struct sbm_component_t *) component->data)->params);

    uint32_t k = params->k;
    params->k -= 1;

    double *Q_new = malloc(sizeof(double) * (k - 1) * (k - 1));

    // Copy all but current index
    for(uint32_t i = 0; i < component->idx; i++) {
        for(uint32_t j = 0; j < component->idx; j++) {
            Q_new[i * (k - 1) + j] = params->Q[i * k + j];
        }
        for(uint32_t j = component->idx + 1; j < k; j++) {
            Q_new[i * (k - 1) + j - 1] = params->Q[i * k + j];
        }
    }
    for(uint32_t i = component->idx + 1; i < k; i++) {
        for(uint32_t j = 0; j < component->idx; j++) {
            Q_new[(i - 1) * (k - 1) + j] = params->Q[i * k + j];
        }
        for(uint32_t j = component->idx + 1; j < k; j++) {
            Q_new[(i - 1) * (k - 1) + j - 1] = params->Q[i * k + j];
        }
    }

    // Swap Q
    free(params->Q);
    params->Q = Q_new;

    // Free component
    free(component->data);
}

// ----------------------------------------------------------------------------
//
//                            Parameters Management
//
// ----------------------------------------------------------------------------

/**
 * Create struct sbm_params_t from python dictionary
 */
void *sbm_params_create(PyObject *dict)
{
    // Check Keys
    PyObject *a_py = (PyObject *) PyDict_GetItemString(dict, "a");
    PyObject *b_py = (PyObject *) PyDict_GetItemString(dict, "b");
    PyObject *asn_py = (PyObject *) PyDict_GetItemString(dict, "asn");
    PyObject *data_py = (PyObject *) PyDict_GetItemString(dict, "data");

    bool check = (
        (a_py != NULL) && PyFloat_Check(a_py) &&
        (a_py != NULL) && PyFloat_Check(a_py) &&
        (asn_py != NULL) &&
        (data_py != NULL)
    );

    if(!check) {
        PyErr_SetString(
            PyExc_KeyError,
            "SBM requires n (number of points), a, b (SBM prior parameters), "
            "asn (reference to assignment array, data (reference to data "
            "matrix).");
        return NULL;
    }

    // Allocate Parameters
    struct sbm_params_t *params = (
        (struct sbm_params_t *) malloc(sizeof(struct sbm_params_t)));

    // Parameters
    params->n = PyArray_DIM((PyArrayObject *) asn_py, 0);
    params->k = 0;
    params->a = PyFloat_AsDouble(a_py);
    params->b = PyFloat_AsDouble(b_py);

    // At this stage, no components have been added yet, so Q is a null
    // pointer.
    params->Q = NULL;

    // Link assignments & data
    params->assignments = (PyArrayObject *) asn_py;
    params->data = (PyArrayObject *) data_py;
    Py_INCREF(asn_py);
    Py_INCREF(data_py);

    return (void *) params;
}


/**
 * Destroy struct sbm_params_t
 */
void sbm_params_destroy(void *params)
{
    struct sbm_params_t *params_tc = (struct sbm_params_t *) params;

    Py_DECREF(params_tc->assignments);
    Py_DECREF(params_tc->data);

    free(params_tc->Q);
    free(params_tc);
}


/**
 * Update with new Q array
 */
void sbm_params_update(void *params, PyObject *dict)
{
    struct sbm_params_t *params_tc = (struct sbm_params_t *) params;

    // Q present?
    PyArrayObject *Q_py = (PyArrayObject *) PyDict_GetItemString(dict, "Q");
    if(Q_py == NULL) {
        printf("Warning: SBM did not update Q array.\n");
    }
    else {
        uint32_t k = params_tc->k;

        // Make sure Q is valid
        if(!type_check_square(Q_py, k)) { return; }
        // Copy Q
        double *Q = (double *) PyArray_DATA(Q_py);
        for(uint32_t i = 0; i < k * k; i++) { params_tc->Q[i] = Q[i]; }
    }
}


// ----------------------------------------------------------------------------
//
//                     Component Likelihoods & Information
//
// ----------------------------------------------------------------------------

/**
 * Add point to SBM object
 * @param component : component to add
 * @param point : data point
 */
void sbm_add(Component *component, void *params, void *point)
{
    // No update
    (void) component;
    (void) params;
    (void) point;
}


/**
 * Remove point from SBM object
 * @param component : component to remove
 * @param point : data point
 */
void sbm_remove(Component *component, void *params, void *point)
{
    // No update
    (void) component;
    (void) params;
    (void) point;
}


/**
 * Get Log Likelihood of unconditional assignment for new cluster probability
 * m(x_j)
 * @param params : model hyperparameters
 * @param point : data point
 */

double sbm_loglik_new(void *params, void *point)
{
    struct sbm_params_t *params_tc = (struct sbm_params_t *) params;
    uint8_t *point_tc = (uint8_t *) point;
    uint16_t *asn = (uint16_t *) PyArray_DATA(params_tc->assignments);

    // Count number of connections to each cluster
    uint32_t *connected = malloc(sizeof(uint32_t) * (params_tc->k));
    uint32_t *unconnected = malloc(sizeof(uint32_t) * (params_tc->k));
    for(uint32_t i = 0; i < params_tc->k; i++) {
        connected[i] = 0;
        unconnected[i] = 0;
    }

    for(uint32_t i = 0; i < params_tc->n; i++) {
        assert(asn[i] < params_tc->k);
        if(point_tc[i]) { connected[asn[i]] += 1; }
        else { unconnected[asn[i]] += 1; }
    }

    // Compute loglik
    double loglik = (
        -1 * params_tc->k * log_beta(params_tc->a, params_tc->b));
    for(uint32_t i = 0; i < params_tc->k; i++) {
        loglik += log_beta(
            connected[i] + params_tc->a,
            unconnected[i] + params_tc->b);
    }

    free(connected);
    free(unconnected);
    return loglik;
}


/**
 * Get Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
 * @param component : component c
 * @param params : model hyperparameters
 * @param point : data point
 */
double sbm_loglik_ratio(Component *component, void *params, void *point)
{
    struct sbm_params_t *params_tc = (struct sbm_params_t *) params;
    uint8_t *point_tc = (uint8_t *) point;
    uint16_t *asn = (uint16_t *) PyArray_DATA(params_tc->assignments);

    double loglik = 0;
    for(uint32_t i = 0; i < params_tc->n; i++) {
        double q = params_tc->Q[component->idx * params_tc->k + asn[i]];
        if(point_tc[i]) { loglik += log(q); }
        else { loglik += log(1 - q); }
    }
    return loglik;
}


// ----------------------------------------------------------------------------
//
//                                  Utilities
//
// ----------------------------------------------------------------------------

/** 
 * Get Q Array, and sample new array
 * @return PyArrayObject containing *copy* of Q
 */
PyObject *sbm_inspect(void *params) {

    struct sbm_params_t *params_tc = (struct sbm_params_t *) params;

    // Copy of Q
    const npy_intp dims[] = {params_tc->k, params_tc->k};
    PyArrayObject *Q_old_py = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    double *Q_old = PyArray_DATA(Q_old_py);
    for(uint32_t i = 0; i < params_tc->k * params_tc->k; i++) {
        Q_old[i] = params_tc->Q[i];
    }

    return (PyObject *) Q_old_py;
}


ComponentMethods STOCHASTIC_BLOCK_MODEL = {
    // Hyperparameters
    &sbm_params_create,
    &sbm_params_destroy,
    &sbm_params_update,

    // Component Management
    &sbm_create,
    &sbm_destroy,
    &sbm_add,
    &sbm_remove,

    // Component Likelihoods
    &sbm_loglik_ratio,
    &sbm_loglik_new,
    NULL,

    // Get Q matrix
    &sbm_inspect
};
