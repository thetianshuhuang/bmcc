/** 
 * Core Mixture Struct shortcuts
 */

#include "../include/mixture.h"


/** 
 * Get Marginal Log Likelihood for assignment of a single point
 * @param model parent model
 * @param cluster cluster to add to
 * @param point to get marginal log likelihood for
 * @return component_loglik_ratio(pt) * model_coefficient(pt)
 */
double marginal_loglik(
    struct mixture_model_t *model, void *cluster, double *point)
{
    return (
        model->comp_methods->loglik_ratio(
            cluster,
            model->comp_params,
            point) +
        model->model_methods->log_coef(
            model->model_params,
            model->comp_methods->get_size(cluster),
            model->num_clusters));
}


/** 
 * Get New Cluster Likelihood
 * @param model parent model
 * @param point point to evaluate
 * @return component_loglik_new(point) * model_log_coef_new()
 */
double new_cluster_loglik(struct mixture_model_t *model, double *point)
{
    return (
        model->comp_methods->loglik_new(
            model->comp_params,
            point) +
        model->model_methods->log_coef_new(
            model->model_params,
            model->num_clusters));
}


/**
 * Add point
 * @param model parent model
 * @param cluster cluster to add to
 * @param point point to add
 */
void *add_point(struct mixture_model_t *model, void *cluster, double *point)
{
    model->comp_methods->add(cluster, model->comp_params, point);
}


/**
 * Delete cluster
 * @param model parent model
 * @param cluster cluster to delete
 */
void destroy(struct mixture_model_t *model, void *cluster)
{
    model->comp_methods->destroy(cluster);
    free(cluster);
}
