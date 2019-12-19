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
 * Get index of cluster
 * @param model parent model
 * @param cluster cluster to delete
 * @return current index of cluster
 */
int indexof(struct mixture_model_t *model, void *cluster) {
    for(int i = 0; i < model->num_clusters; i++) {
        if(model->clusters[i] == cluster) { return i; }
    }
    return 0;
}


/**
 * Delete cluster
 * @param model parent model
 * @param cluster cluster to delete
 */
void destroy(struct mixture_model_t *model, void *cluster)
{
    model->comp_methods->destroy(cluster, indexof(model, cluster));
    free(cluster);
}


/**
 * Get cluster at index (safely)
 * @param model : mixture_model_t struct to fetch from
 * @param idx index to fetch
 * @return fetched component; NULL if unsuccessful
 */
void *get_cluster(struct mixture_model_t *model, int idx)
{
    if(idx >= model->num_clusters) {
        printf(
            "[C BACKEND ERROR] Invalid cluster: %d [total=%d]\n",
            idx, model->num_clusters);
        return NULL;
    }
    else {
        return model->clusters[idx];
    }
}
