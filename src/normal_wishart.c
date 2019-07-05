/**
 *
 *
 */

#include <math.h>
#include "cholesky.h"
#include "misc_math.h"

#include "normal_wishart.h"


/**
 * Create struct nw_params_t from python dictionary
 */
void *nw_params_create(PyObject *dict)
{
    // Allocate parameters
    struct nw_params_t *params = (
        (struct nw_params_t *) malloc(sizeof(nw_params_t)));
    params->df = (float) PyFloat_AsDouble(PyDict_GetItemString(dict, "df"));
    params->dim = (int) PyLong_AsLong(PyDict_GetItemString(dict, "dim"));
    params->s_chol = (float *) malloc(sizeof(float) * dim * dim);

    // Copy cholesky decomp
    float *data = PyArray_DATA(PyDict_GetItemString(dict, "s_chol"));
    for(int i = 0; i < dim * dim; i++) { s_chol[i] = data[i]; }

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
    
    struct nw_params *params_tc = (struct nw_params *) params;

    // Total = [0]
    component->total = malloc(sizeof(float) * dim);
    for(int i= 0; i < dim; i++) { component->total[i] = 0; }

    // # points = 0
    component->n = 0;

    // Copy over starting value [S + XX^T] = [S]
    component->chol_decomp = malloc(sizeof(float) * dim * dim);
    float *chol_src = params_tc->s_chol;
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
    // Free arrays
    free(component_tc->total);
    free(component_tc->chol_decomp);
}


/**
 * Get size of component
 * @param component : component to get size for
 * @return number of points associated with the component
 */
void get_size(void *component)
{
    return component->n;
}


/**
 * Add point to normal wishart object
 * @param component : component to add
 * @param point : data point
 */
void nw_add(void *component, float *point)
{
    struct nw_component_t *comp_inner = (struct nw_component_t *) component;

    // Update Cholesky decomposition and total
    cholesky_update(comp_inner->chol_decomp, point, comp_inner->dim);
    for(int i = 0; i < comp_inner->dim; i++) {
        comp_inner->total[i] += point[i];
    }

    // Update # points
    comp_inner->n += 1;
}


/**
 * Remove point from normal wishart object
 * @param component : component to remove
 * @param point : data point
 */
void nw_remove(void *component, float *point)
{
    struct nw_component_t *comp_inner = (struct nw_component_t *) component;

    // Downdate Cholesky decomposition, total, # of points
    cholesky_downdate(comp_inner->chol_decomp, point, comp_inner->dim);
    for(int i = 0; i < comp_inner->dim; i++) {
        comp_inner->total[i] -= point[i];
    }
    comp_inner->n -= 1;
}


/**
 * Get Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
 * @param component : component c
 * @param params : model hyperparameters
 * @param point : data point
 */
double nw_loglik_ratio(void *component, void *params, float *point)
{
    struct nw_component_t *cpt = (struct nw_component_t *) component;

    int dim = params->dim;
    
    // Cholesky update on a copy
    float *chol_cpy = (float *) malloc(sizeof(float) * dim * dim);
    for(int i = 0; i < dim * dim; i++) {
        chol_cpy[i] = cpt->chol_decomp[i];
    }
    cholesky_update(chol_cpy, point, dim);

    return (
        - log(M_PI) * (dim / 2)
        + log_mv_gamma((cpt->df + cpt->n + 1) / 2)
        - log_mv_gamma((cpt->df + cpt->n))
        + cholesky_logdet(cpt->chol_decomp, dim) * (cpt->df + cpt->n) / 2
        - cholesky_logdet(chol_cpy, dim) * (cpt->df + cpt->n + 1) / 2
    );
}


/**
 * Get Log Likelihood of unconditional assignment for new cluster probability
 * @param params : model hyperparameters
 * @param point : data point
 */
double nw_loglik_new(void *params, float *point)
{
    // todo
    return 0;
}


/**
 * Extern for normal_wishart methods
 */
const extern ComponentMethods normal_wishart = {
    &nw_params_create,
    &nw_params_destroy,
    &nw_create,
    &nw_destroy,
    &nw_add,
    &nw_remove,
    &nw_loglik_ratio,
    &nw_loglik_new
};
