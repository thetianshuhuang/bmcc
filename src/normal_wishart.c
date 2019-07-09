/**
 * Wishart-distributed Multivariate Gaussian Components
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdlib.h>

#include <stdio.h>

#include "cholesky.h"
#include "misc_math.h"
#include "normal_wishart.h"
#include "mixture.h"
#include "type_check.h"


// ----------------------------------------------------------------------------
//
//                            Component Management
//
// ----------------------------------------------------------------------------

/**
 * Create normal wishart object
 * @param params : model hyperparameters
 * @return Allocated component structure
 */
void *nw_create(void *params)
{
    // Allocate memory
    struct nw_component_t *component = (
        (struct nw_component_t *) malloc(sizeof(struct nw_component_t)));
    
    struct nw_params_t *params_tc = (struct nw_params_t *) params;
    int dim = params_tc->dim;

    // Total = [0]
    component->total = (double *) malloc(sizeof(double) * dim);
    for(int i= 0; i < dim; i++) { component->total[i] = 0; }

    // # points = 0
    component->n = 0;

    // Copy over starting value [S + XX^T] = [S]
    component->chol_decomp = (double *) malloc(sizeof(double) * dim * dim);
    double *chol_src = params_tc->s_chol;
    for(int i = 0; i < dim * dim; i++) {
        component->chol_decomp[i] = chol_src[i];
    }

    return component;
}


/**
 * Destroy normal wishart object
 * @param component : component to destroy
 */
void nw_destroy(void *component)
{
    struct nw_component_t *component_tc = (struct nw_component_t *) component;
    // Free arrays
    free(component_tc->total);
    free(component_tc->chol_decomp);
}


// ----------------------------------------------------------------------------
//
//                            Parameters Management
//
// ----------------------------------------------------------------------------

/**
 * Create struct nw_params_t from python dictionary
 */
void *nw_params_create(PyObject *dict)
{
    // Check keys
    PyObject *df_py = PyDict_GetItemString(dict, "df");
    PyObject *dim_py = PyDict_GetItemString(dict, "dim");
    if((df_py == NULL) || (dim_py == NULL) ||
            (!PyFloat_Check(df_py)) || (!PyLong_Check(dim_py))) {
        PyErr_SetString(
            PyExc_KeyError,
            "Normal Wishart requres 'df' (wishart degrees of freedom) and " \
            "'dim' (data dimensions) arguments.");
        return NULL;            
    }

    // Unpack dict
    double df = PyFloat_AsDouble(df_py);
    int dim = (int) PyLong_AsLong(dim_py);
    // Check cholesky size
    PyObject *data_py = PyDict_GetItemString(dict, "s_chol");
    if((data_py == NULL) || !PyArray_Check(data_py)) {
        PyErr_SetString(
            PyExc_KeyError,
            "Normal Wishart requires 's_chol' (cholesky decomposition of " \
            "scale matrix) to be passed as a numpy array with type float64 " \
            "(double).");
        return NULL;
    }
    if(!type_check_square((PyArrayObject *) data_py, dim)) { return NULL; }

    // Allocate parameters
    struct nw_params_t *params = (
        (struct nw_params_t *) malloc(sizeof(struct nw_params_t)));
    params->df = df;
    params->dim = dim;

    // Copy cholesky decomp
    params->s_chol = (double *) malloc(sizeof(double) * dim * dim);
    double *data = PyArray_DATA((PyArrayObject *) data_py);
    for(int i = 0; i < dim * dim; i++) { params->s_chol[i] = data[i]; }

    return (void *) params;
}


/**
 * Destroy struct nw_params_t
 */
void nw_params_destroy(void *params)
{
    struct nw_params_t *params_tc = (struct nw_params_t *) params;
    free(params_tc->s_chol);
    free(params_tc);
}


// ----------------------------------------------------------------------------
//
//                     Component Likelihoods & Information
//
// ----------------------------------------------------------------------------

/**
 * Get size of component
 * @param component : component to get size for
 * @return number of points associated with the component
 */
int nw_get_size(void *component)
{
    return ((struct nw_component_t *) component)->n;
}


/**
 * Add point to normal wishart object
 * @param component : component to add
 * @param point : data point
 */
void nw_add(void *component, void *params, double *point)
{
    struct nw_component_t *comp_tc = (struct nw_component_t *) component;
    struct nw_params_t *params_tc = (struct nw_params_t *) params;

    // Update Cholesky decomposition, mean, # of points
    cholesky_update(comp_tc->chol_decomp, point, 1, params_tc->dim);
    for(int i = 0; i < params_tc->dim; i++) { comp_tc->total[i] += point[i]; }
    comp_tc->n += 1;
}


/**
 * Remove point from normal wishart object
 * @param component : component to remove
 * @param point : data point
 */
void nw_remove(void *component, void *params, double *point)
{
    struct nw_component_t *comp_tc = (struct nw_component_t *) component;
    struct nw_params_t *params_tc = (struct nw_params_t *) params;

    // Downdate Cholesky decomposition, total, # of points
    cholesky_downdate(comp_tc->chol_decomp, point, 1, params_tc->dim);
    for(int i = 0; i < params_tc->dim; i++) { comp_tc->total[i] -= point[i]; }
    comp_tc->n -= 1;
}


/**
 * Get Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
 * @param component : component c
 * @param params : model hyperparameters
 * @param point : data point
 */
double nw_loglik_ratio(void *component, void *params, double *point)
{
    struct nw_component_t *cpt = (struct nw_component_t *) component;
    struct nw_params_t *params_tc = (struct nw_params_t *) params;

    int dim = params_tc->dim;
    double df = params_tc->df;

    // Scratch arrays
    double *chol_up = (double *) malloc(sizeof(double) * dim * dim);
    double *chol_center = (double *) malloc(sizeof(double) * dim * dim);
    double *total_up = (double *) malloc(sizeof(double) * dim);

    // Copy cholesky decomposition
    for(int i = 0; i < dim * dim; i++) { chol_up[i] = cpt->chol_decomp[i]; }
    for(int i = 0; i < dim * dim; i++) { chol_center[i] = cpt->chol_decomp[i]; }
    
    // Update point
    for(int i = 0; i < dim; i++) { total_up[i] = cpt->total[i] + point[i]; }
    cholesky_update(chol_up, point, 1, dim);

    // Center cholesky decompositions using downdate procedure
    cholesky_downdate(chol_center, cpt->total, 1 / sqrt(cpt->n), dim);
    cholesky_downdate(chol_up, total_up, 1 / sqrt(cpt->n + 1), dim);

    double res = (
        - log(M_PI) * (dim / 2)
        + log_mv_gamma(dim, (df + cpt->n + 1) / 2)
        - log_mv_gamma(dim, (df + cpt->n) / 2)
        + cholesky_logdet(chol_center, dim) * (df + cpt->n) / 2
        - cholesky_logdet(chol_up, dim) * (df + cpt->n + 1) / 2
    );

    // Clean up
    free(chol_up);
    free(chol_center);
    free(total_up);
    return res;
}


/**
 * Get Log Likelihood of unconditional assignment for new cluster probability
 * m(x_j)
 * @param params : model hyperparameters
 * @param point : data point
 */
double nw_loglik_new(void *params, double *point)
{
    struct nw_params_t *params_tc = (struct nw_params_t *) params;
    int dim = params_tc->dim;
    double df = params_tc->df;

    double *chol_up = (double *) malloc(sizeof(double) * dim * dim);
    for(int i = 0; i < dim * dim; i++) { chol_up[i] = params_tc->s_chol[i]; }
    cholesky_update(chol_up, point, 1, dim);

    double res = (
        - log(M_PI) * (dim / 2)
        + log_mv_gamma(dim, (df + 1) / 2)
        - log_mv_gamma(dim, df / 2)
        + cholesky_logdet(params_tc->s_chol, dim) * df / 2
        - cholesky_logdet(chol_up, dim) * (df + 1) / 2
    );

    free(chol_up);

    return res;
}


/**
 * Extern for normal_wishart methods
 */
ComponentMethods NORMAL_WISHART = {
    &nw_params_create,
    &nw_params_destroy,
    NULL,  // No update
    &nw_create,
    &nw_destroy,
    &nw_get_size,
    &nw_add,
    &nw_remove,
    &nw_loglik_ratio,
    &nw_loglik_new
};
