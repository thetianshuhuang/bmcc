/**
 * Split Merge Sampler
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../include/mixture/mixture.h"
#include "../include/misc_math.h"
#include "../include/type_check.h"
#include "../include/samplers/base_iter.h"

#include <stdio.h>


/**
 * Reconstruct proposal probability for merge operation
 * @param c1 first cluster index
 * @param c2 second cluster index
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 * @return proposal probability
 */
double merge_propose_prob(
    int c1, int c2,
    void *data, uint16_t *assignments,
    struct mixture_model_t *model)
{
    // Temporary clusters
    // void *c1_cpy = model->comp_methods->create(model->comp_params);
    // void *c2_cpy = model->comp_methods->create(model->comp_params);

    // Iterate over clusters c1, c2
    double res = 0;
    for(uint32_t i = 0; i < model->size; i++) {
        if(assignments[i] == c1 || assignments[i] == c2) {
            
            void *point = data + i * model->dim * model->stride;

            // Likelihoods
            double asn1 = marginal_loglik(
                model, get_cluster(model, c1), point);
            double asn2 = marginal_loglik(
                model, get_cluster(model, c2), point);
            res += (assignments[i] == c1) ? asn1 : asn2;

            // Normalization Constant
            res -= log(exp(asn1) + exp(asn2));
        }
    }

    // Clean up
    // destroy(model, c1_cpy);
    // destroy(model, c2_cpy);
    return res;
}


/**
 * Merge Update
 * @param c1 first cluster index
 * @param c2 second cluster index
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 * @return true if returned without error
 */
bool merge(
    int c1, int c2,
    void *data, uint16_t *assignments,
    struct mixture_model_t *model)
{
    // Modify copy of assignments
    uint8_t *asn_tmp = malloc(sizeof(uint8_t) * model->size);

    // Create merge component; index becomes [size-2]
    Component *new_component = create_component(model);
    for(uint32_t i = 0; i < model->size; i++) {
        if(assignments[i] == c1 || assignments[i] == c2) {
            void *point = data + i * model->dim * model->stride;
            add_point(model, new_component, point);
            asn_tmp[i] = 1;
        }
        else {
            asn_tmp[i] = 0;
        }
    }

    // Compute acceptance probability
    double mixture_ratio = model->model_methods->log_merge(
        model->model_params,
        model->num_clusters,
        get_cluster(model, c1)->size,
        get_cluster(model, c2)->size);
    double component_ratio = model->comp_methods->split_merge(
        model->comp_params,
        new_component,
        get_cluster(model, c1),
        get_cluster(model, c2));
    double propose_ratio = merge_propose_prob(
        c1, c2, data, assignments, model);
    double p_accept = exp(propose_ratio + mixture_ratio - component_ratio);

    // Accept -> remove old components, add new components
    if(p_accept > ((double) rand_45_bit() / (double) RAND_MAX_45)) {

        #ifdef SHOW_ACCEPT
        printf(
            "[accepted][merge] "
            "propose: %f mixture: %f component: %f p_accept: %f\n",
            propose_ratio, mixture_ratio, - component_ratio, log(p_accept));
        #endif

        // Remove larger index first to avoid mangling index linkage
        if(c1 > c2) {
            remove_component(model, assignments, c1);
            remove_component(model, assignments, c2);
        }
        else {
            remove_component(model, assignments, c2);
            remove_component(model, assignments, c1);
        }
        bool success = add_component(model, new_component);
        if(!success) { return false; }

        // Update assignments
        for(uint32_t i = 0; i < model->size; i++) {
            if(asn_tmp[i] != 0) {
                assignments[i] = model->num_clusters - 1;
            }
        }
    }
    // Reject -> free new component
    else {

        #ifdef SHOW_REJECT
        printf(
            "          [merge] "
            "propose: %f mixture: %f component: %f p_accept: %f\n",
            propose_ratio, mixture_ratio, - component_ratio, log(p_accept));
        #endif

        destroy(model, new_component);
    }

    free(asn_tmp);
    return true;
}


/**
 * Merge Update
 * @param cluster cluster to split
 * @param p1 first point index
 * @param p2 second point index
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 * @return true if returned without error
 */
bool split(
    uint32_t cluster, uint32_t p1, uint32_t p2,
    void *data, uint16_t *assignments,
    struct mixture_model_t *model)
{
    // Modify copy of assignments
    uint8_t *asn_tmp = malloc(sizeof(uint8_t) * model->size);

    // Create new clusters
    // If accepted, new1 becomes index [size - 1], new2 becomes [size]
    Component *new1 = create_component(model);
    add_point(model, new1, data + p1 * model->dim * model->stride);
    asn_tmp[p1] = 2;
    Component *new2 = create_component(model);
    add_point(model, new2, data + p2 * model->dim * model->stride);
    asn_tmp[p2] = 1;

    // Track propose likelihood
    double propose_ratio = 0;

    // Split points randomly, with equal probability for points in splitting
    // cluster
    for(uint32_t i = 0; i < model->size; i++) {
        bool not_original = (i != p1) && (i != p2);
        bool in_merge = (assignments[i] == cluster);

        if(in_merge && not_original) {

            void *point = data + i * model->dim * model->stride;

            // Get assignment likelihoods
            double asn1 = marginal_loglik(model, new1, point);
            double asn2 = marginal_loglik(model, new2, point);

            // Normalize using logsumexp
            double denom_log = log(exp(asn1) + exp(asn2));
            double likelihood = exp(asn1 - denom_log);

            // Assign randomly; track propose ratio taken
            if(likelihood > (double) rand_45_bit() / (double) RAND_MAX_45) {
                add_point(model, new1, point);
                asn_tmp[i] = 2;
                propose_ratio += asn1 - denom_log;
            }
            else {
                add_point(model, new2, point);
                asn_tmp[i] = 1;
                propose_ratio += asn2 - denom_log;
            }
        }
        else if(!in_merge) {
            asn_tmp[i] = 0;
        }
    }

    // Compute acceptance probability
    double mixture_ratio = model->model_methods->log_split(
        model->model_params,
        model->num_clusters,
        new1->size,
        new2->size);
    double component_ratio = model->comp_methods->split_merge(
        model->comp_params,
        get_cluster(model, cluster),
        new1, new2);

    double p_accept = exp(- propose_ratio + mixture_ratio + component_ratio);

    // Accept -> remove old component, add new components
    if(p_accept > ((double) rand_45_bit() / (double) RAND_MAX_45)) {

        #ifdef SHOW_ACCEPT
        printf(
            "[accepted][split] "
            "propose: %f n1: %d n2: %d mixture: %f component: %f p_accept: %f\n",
            - propose_ratio,
            model->comp_methods->get_size(new1),
            model->comp_methods->get_size(new2),
            mixture_ratio, component_ratio, log(p_accept));
        #endif

        // Remove old
        remove_component(model, assignments, cluster);
        // Add new components
        add_component(model, new1);  // -2
        add_component(model, new2);  // -1

        // Copy assignments
        // asn_tmp index [0] becomes [size]; [1] becomes [size + 1]
        // After removal, these become [size - 1], [size]
        for(uint32_t i = 0; i < model->size; i++) {
            if(asn_tmp[i] != 0) {
                assignments[i] = model->num_clusters - asn_tmp[i];
            }
        }
    }
    // Reject -> free new components
    else {

        #ifdef SHOW_REJECT
        printf(
            "          [split] "
            "propose: %f n1: %d n2: %d mixture: %f component: %f p_accept: %f\n",
            - propose_ratio,
            model->comp_methods->get_size(new1),
            model->comp_methods->get_size(new2),
            mixture_ratio, component_ratio, log(p_accept));
        #endif

        destroy(model, new1);
        destroy(model, new2);
    }

    free(asn_tmp);
    return true;
}


/** 
 * Main Split Merge Function; samples source points and decides whether to
 * call "split" or "merge"
 * @param data data points
 * @param assignments assignment vector
 * @param model mixture model
 * @param annealing annealing factor
 * @return bool true on success; false on failure
 */
bool split_merge(
    void *data,
    uint16_t *assignments,
    struct mixture_model_t *model,
    double annealing)
{
    // Does not support annealing
    (void) annealing;

    // Get pivot elements i, j
    uint32_t i = (uint32_t) (rand_45_bit() % model->size);
    uint32_t j = i;
    while(i == j) { i = (uint32_t) (rand_45_bit() % model->size); }
    if(assignments[i] != assignments[j]) {
        return merge(assignments[i], assignments[j], data, assignments, model);
    }
    else {
        return split(assignments[i], i, j, data, assignments, model);
    }
}


/**
 * Run Split Merge Iteration. See docstring (sourced from split_merge.h) for
 * details on Python calling.
 */
PyObject *split_merge_py(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return base_iter(self, args, kwargs, &supports_split_merge, &split_merge);
}
