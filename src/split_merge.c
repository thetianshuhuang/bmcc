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

#include "../include/mixture.h"
#include "../include/misc_math.h"
#include "../include/type_check.h"
#include "../include/base_iter.h"

#include <stdio.h>
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
    double *data, uint16_t *assignments,
    struct mixture_model_t *model)
{
    // Create merge component; index becomes [size-2]
    void *new_component = model->comp_methods->create(model->comp_params);

    // Modify copy of assignments
    uint16_t *asn_tmp = malloc(sizeof(uint8_t) * model->size);

    // Add points in clusters c1, c2 to new_component
    double propose_ratio = 0;
    for(int i = 0; i < model->size; i++) {
        if(i == c1 || i == c2) {
            double *point = &(data[i * model->dim]);
            propose_ratio += model->comp_methods->loglik_ratio(
                new_component, model->comp_params, point);
            model->comp_methods->add(
                new_component, model->comp_params, point);
            asn_tmp[i] = 1;
        }
        else {
            asn_tmp[i] = 0;
        }
    }

    // Compute acceptance probability
    double mixture_ratio = model->model_methods->log_merge(
        model->model_params, model->num_clusters,
        model->comp_methods->get_size(get_cluster(model, c1)),
        model->comp_methods->get_size(get_cluster(model, c2)));
    double component_ratio = model->comp_methods->split_merge(
        model->comp_params,
        new_component,
        get_cluster(model, c1),
        get_cluster(model, c2));
    double p_accept = exp(propose_ratio + mixture_ratio - component_ratio);

    printf(
        "[merge] propose: %f mixture: %f component: %f p_accept: %f\n",
        propose_ratio, mixture_ratio, - component_ratio, p_accept);

    // Accept -> remove old components, add new components
    if(p_accept > ((double) rand_45_bit() / (double) RAND_MAX_45)) {
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
        for(int i = 0; i < model->size; i++) {
            if(asn_tmp[i] != 0) {
                assignments[i] = model->num_clusters - 1;
            }
        }
    }
    // Reject -> free new component
    else {
        model->comp_methods->destroy(new_component);
    }

    free(asn_tmp);
    return true;
}


/**
 * Merge Update
 * @param cluster cluster to split
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 * @return true if returned without error
 */
bool split(
    int cluster,
    double *data, uint16_t *assignments,
    struct mixture_model_t *model)
{
    // Create new clusters
    // If accepted, new1 becomes index [size - 1], new2 becomes [size]
    void *new1 = model->comp_methods->create(model->comp_params);
    void *new2 = model->comp_methods->create(model->comp_params);

    // Track propose likelihood
    double propose_ratio = 0;

    // Modify copy of assignments
    uint16_t *asn_tmp = malloc(sizeof(uint8_t) * model->size);

    // Split points randomly, with equal probability for points in splitting
    // cluster
    for(int i = 0; i < model->size; i++) {
        if(assignments[i] == cluster) {
            double *point = &(data[model->dim * i]);
  
            // Get assignment likelihoods
            double asn1 = model->comp_methods->loglik_ratio(
                new1, model->comp_params, point);
            double asn2 = model->comp_methods->loglik_ratio(
                new2, model->comp_params, point);
            double likelihood = exp(asn1) / (exp(asn1) + exp(asn2));

            // Assign randomly
            if(likelihood > (double) rand_45_bit() / (double) RAND_MAX_45) {
                propose_ratio += asn1;
                model->comp_methods->add(new1, model->comp_params, point);
                asn_tmp[i] = 2;
            }
            else {
                propose_ratio += asn2;
                model->comp_methods->add(new2, model->comp_params, point);
                asn_tmp[i] = 1;
            }
        }
        else {
            asn_tmp[i] = 0;
        }
    }

    // Compute acceptance probability
    double mixture_ratio = model->model_methods->log_split(
        model->model_params, model->num_clusters,
        model->comp_methods->get_size(new1),
        model->comp_methods->get_size(new2));
    double component_ratio = model->comp_methods->split_merge(
        model->comp_params,
        get_cluster(model, cluster),
        new1, new2);

    double p_accept = exp(- propose_ratio + mixture_ratio + component_ratio);

    printf(
        "[split] propose: %f mixture: %f component: %f p_accept: %f\n",
        - propose_ratio, mixture_ratio, component_ratio, p_accept);

    // Accept -> remove old component, add new components
    if(p_accept > ((double) rand_45_bit() / (double) RAND_MAX_45)) {

        // Remove old
        remove_component(model, assignments, cluster);
        // Add new components
        add_component(model, new1);
        add_component(model, new2);

        // Copy assignments
        // asn_tmp index [0] becomes [size]; [1] becomes [size + 1]
        // After removal, these become [size - 1], [size]
        for(int i = 0; i < model->size; i++) {
            if(asn_tmp[i] != 0) {
                assignments[i] = model->num_clusters - asn_tmp[i];
            }
        }
    }
    // Reject -> free new components
    else {
        model->comp_methods->destroy(new1);
        model->comp_methods->destroy(new2);
    }

    free(asn_tmp);
    return true;
}


bool split_merge(
    double *data, uint16_t *assignments, struct mixture_model_t *model)
{
    // Get pivot elements i, j
    int i = (int) (rand_45_bit() % model->size);
    int j = i;
    while(i == j) { i = (int) (rand_45_bit() % model->size); }

    if(assignments[i] != assignments[j]) {
        return merge(assignments[i], assignments[j], data, assignments, model);
    }
    else {
        return split(assignments[i], data, assignments, model);
    }
}


/**
 * Run Split Merge Iteration. See docstring (sourced from split_merge.h) for
 * details on Python calling.
 */
PyObject *split_merge_py(PyObject *self, PyObject *args)
{
    return base_iter(self, args, &supports_split_merge, &split_merge);
}

