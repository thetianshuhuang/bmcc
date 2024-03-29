/**
 * Wishart-distributed Multivariate Gaussian Components
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "../include/cholesky.h"
#include "../include/misc_math.h"
#include "../include/models/normal_wishart.h"
#include "../include/mixture/mixture.h"
#include "../include/type_check.h"


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
    uint32_t dim = params_tc->dim;

    // Total = [0]
    component->total = (double *) malloc(sizeof(double) * dim);
    for(uint32_t i = 0; i < dim; i++) { component->total[i] = 0; }

    // Copy over starting value [S + XX^T] = [S]
    component->chol_decomp = (double *) malloc(sizeof(double) * dim * dim);
    double *chol_src = params_tc->s_chol;
    for(uint32_t i = 0; i < dim * dim; i++) {
        component->chol_decomp[i] = chol_src[i];
    }

    return component;
}


/**
 * Destroy normal wishart object
 * @param component : component to destroy
 */
void nw_destroy(Component *component)
{
    struct nw_component_t *component_tc = (
        (struct nw_component_t *) component->data);
    // Free arrays
    free(component_tc->total);
    free(component_tc->chol_decomp);
    // Free nw_component_t
    free(component_tc);
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
    uint32_t dim = (int) PyLong_AsLong(dim_py);
    // Check cholesky size
    PyObject *data_py = PyDict_GetItemString(dict, "s_chol");
    if((data_py == NULL) || !PyArray_Check(data_py) ||
            !type_check_square((PyArrayObject *) data_py, dim)) {
        PyErr_SetString(
            PyExc_KeyError,
            "Normal Wishart requires 's_chol' (cholesky decomposition of " \
            "scale matrix) to be passed as a numpy array with type float64 " \
            "(double). s_chol should be square, and have dimensions dim x " \
            "dim.");
        return NULL;
    }

    // Allocate parameters
    struct nw_params_t *params = (
        (struct nw_params_t *) malloc(sizeof(struct nw_params_t)));
    params->df = df;
    params->dim = dim;

    // Copy cholesky decomp
    params->s_chol = (double *) malloc(sizeof(double) * dim * dim);
    double *data = PyArray_DATA((PyArrayObject *) data_py);
    for(uint32_t i = 0; i < dim * dim; i++) { params->s_chol[i] = data[i]; }

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
 * Add point to normal wishart object
 * @param component : component to add
 * @param point : data point
 */
void nw_add(Component *component, void *params, void *point)
{
    struct nw_component_t *comp_tc = (struct nw_component_t *) component->data;
    struct nw_params_t *params_tc = (struct nw_params_t *) params;

    // Update Cholesky decomposition, mean, # of points
    cholesky_update(comp_tc->chol_decomp, point, 1, params_tc->dim);
    for(uint32_t i = 0; i < params_tc->dim; i++) {
        comp_tc->total[i] += ((double *) point)[i];
    }
}


/**
 * Remove point from normal wishart object
 * @param component : component to remove
 * @param point : data point
 */
void nw_remove(Component *component, void *params, void *point)
{
    struct nw_component_t *comp_tc = (struct nw_component_t *) component->data;
    struct nw_params_t *params_tc = (struct nw_params_t *) params;

    // Downdate Cholesky decomposition, total, # of points
    cholesky_downdate(
        comp_tc->chol_decomp, ((double *) point), 1, params_tc->dim);
    for(uint32_t i = 0; i < params_tc->dim; i++) {
        comp_tc->total[i] -= ((double *) point)[i];
    }
}


/**
 * Get Log Likelihood of unconditional assignment for new cluster probability
 * m(x_j)
 * @param params : model hyperparameters
 * @param point : data point
 */
double nw_loglik_new(void *params, void *point)
{
    struct nw_params_t *params_tc = (struct nw_params_t *) params;
    uint32_t dim = params_tc->dim;
    double df = params_tc->df;

    double *chol_up = (double *) malloc(sizeof(double) * dim * dim);
    for(uint32_t i = 0; i < dim * dim; i++) {
        chol_up[i] = params_tc->s_chol[i];
    }
    cholesky_update(chol_up, ((double *) point), 1, dim);

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
 * Get Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
 * @param component : component c
 * @param params : model hyperparameters
 * @param point : data point
 */
double nw_loglik_ratio(Component *component, void *params, void *point)
{
    // Unpack
    struct nw_component_t *cpt = (struct nw_component_t *) component->data;
    struct nw_params_t *params_tc = (struct nw_params_t *) params;
    uint32_t dim = params_tc->dim;
    uint32_t size = component->size;
    double df = params_tc->df;

    // Deal with empty component separately
    if(size == 0) { return nw_loglik_new(params, ((double *) point)); }

    // |S + X'X'^T| (centered)
    // Updated Total
    double *total_up = (double *) malloc(sizeof(double) * dim);
    for(uint32_t i = 0; i < dim; i++) {
        total_up[i] = cpt->total[i] + ((double *) point)[i];
    }

    // Updated Cholesky(S + X'X'^T)
    double *chol_up = (double *) malloc(sizeof(double) * dim * dim);
    for(uint32_t i = 0; i < dim * dim; i++) {
        chol_up[i] = cpt->chol_decomp[i];
    }
    cholesky_update(chol_up, ((double *) point), 1, dim);

    // Centered
    cholesky_downdate(chol_up, total_up, 1 / sqrt(size + 1), dim);
    double logdet_new = cholesky_logdet(chol_up, dim);

    // |S + XX^T| (centered)
    double logdet = centered_logdet(cpt->chol_decomp, cpt->total, dim, size);

    double res = (
        - log(M_PI) * (dim / 2)
        + log_mv_gamma(dim, (df + size + 1) / 2)
        - log_mv_gamma(dim, (df + size) / 2)
        + logdet * (df + size) / 2
        - logdet_new * (df + size + 1) / 2
    );

    // Clean up
    free(chol_up);
    free(total_up);
    return res;
}


/**
 * Get split merge likelihood m(X_A)m(X_B) / m(X_A+B)
 * @param params : model hyperparameters
 * @param merged : component A+B
 * @param c1 : component A
 * @param c2 : component B
 */
double nw_split_merge(
    void *params, Component *merged, Component *c1, Component *c2)
{
    // Unpack params
    struct nw_params_t *params_tc = (struct nw_params_t *) params;
    int dim = params_tc->dim;
    double df = params_tc->df;

    // Type cast
    struct nw_component_t *cpt_merged = (struct nw_component_t *) merged->data;
    struct nw_component_t *cpt1 = (struct nw_component_t *) c1->data;
    struct nw_component_t *cpt2 = (struct nw_component_t *) c2->data;

    // Get centered log determinants
    double merged_logdet = centered_logdet(
        cpt_merged->chol_decomp, cpt_merged->total, dim, merged->size);
    double cpt1_logdet = centered_logdet(
        cpt1->chol_decomp, cpt1->total, dim, c1->size);
    double cpt2_logdet = centered_logdet(
        cpt2->chol_decomp, cpt2->total, dim, c2->size);

    return (
        // Gamma_p((df + |c1|) / 2) * Gamma_p((df + |c2|) / 2)
        + log_mv_gamma(dim, (df + c1->size) / 2)
        + log_mv_gamma(dim, (df + c2->size) / 2)
        // Gamma_p(df / 2) * Gamma_p((df + |c1| + |c2) / 2)
        - log_mv_gamma(dim, df / 2)
        - log_mv_gamma(dim, (df + merged->size) / 2)
        // |S + XX^T|^{(df + |c1| + |c2|) / 2} * |S|^{df / 2}
        + merged_logdet * (merged->size + df) / 2
        + cholesky_logdet(params_tc->s_chol, dim) * df / 2
        // |S + XX^T|^{(df + |c1|) / 2} * |S + XX^T|^{(df + |c2|) / 2}
        - cpt1_logdet * (c1->size + df) / 2
        - cpt2_logdet * (c2->size + df) / 2
    );
}


/**
 * Extern for normal_wishart methods
 */
ComponentMethods NORMAL_WISHART = {
    // Hyperparameters
    &nw_params_create,
    &nw_params_destroy,
    NULL,   // No update

    // Component Management
    &nw_create,
    &nw_destroy,
    &nw_add,
    &nw_remove,

    // Component Likelihoods
    &nw_loglik_ratio,
    &nw_loglik_new,
    &nw_split_merge,

    // Debug
    NULL,   // Nothing to inspect
};
