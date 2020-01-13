/**
 * Core Mixture Structs and Routines
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef ASSERT
#undef NDEBUG
#endif
#include "assert.h"
#include <stdint.h>
#include <stdbool.h>

#include "../include/type_check.h"
#include "../include/mixture/mixture.h"


#include <stdio.h>


/**
 * Create new component (not bound)
 * @param model mixture_model_t struct to create component from
 * @return Created Component * container
 */
Component *create_component(struct mixture_model_t *model)
{
    #ifdef SHOW_TRACE
    printf("    create_component\n");
    #endif

    Component *container = (Component *) malloc(sizeof(Component));
    container->size = 0;
    container->data = model->comp_methods->create(model->comp_params);
    return container;
}


/**
 * Delete cluster
 * @param model parent model
 * @param cluster cluster to delete
 */
void destroy(struct mixture_model_t *model, Component *cluster)
{
    #ifdef SHOW_TRACE
    printf("    destroy\n");
    #endif

    if(cluster == NULL) {
        printf("[C BACKEND ERROR] Tried to destroy NULL cluster\n");
    }
    else {
        model->comp_methods->destroy(cluster);
        free(cluster);        
    }
}


/**
 * Add Component: allocates new component, and appends to components capsule
 * @param model mixture_model_t struct containing components
 * @param component component to add; if NULL, allocates a new component
 * @return true if addition successful
 */
bool add_component(struct mixture_model_t *model, Component *component)
{
    #ifdef SHOW_TRACE
    printf("    add_component\n");
    #endif

    // Handle exponential over-allocation
    if(model->mem_size <= model->num_clusters) {
        model->mem_size *= 2;
        Component **clusters_new = (Component **) realloc(
            model->clusters, sizeof(Component *) * model->mem_size);

        if(clusters_new == NULL) {
            PyErr_SetString(
                PyExc_MemoryError, "Failed to allocate new component");
            return false;
        }
        else { model->clusters = clusters_new; }
    }

    // Allocate new only if passed component is NULL
    if(component == NULL) { component = create_component(model); }

    // Bind & update
    model->clusters[model->num_clusters] = component;
    component->idx = model->num_clusters;
    model->num_clusters += 1;

    return true;
}


/**
 * Remove component from component vector.
 * @param components : component vector struct
 * @param assignments : if not NULL, updates assignment indices
 * @param idx : index to remove
 */
void remove_component(
    struct mixture_model_t *model, uint16_t *assignments, uint32_t idx)
{
    assert(idx < model->num_clusters);

    #ifdef SHOW_TRACE
    printf("    remove_component\n");
    #endif

    // Optionally update assignments
    if(assignments != NULL) {
        for(uint32_t i = 0; i < model->size; i++) {
            if(assignments[i] > idx) { assignments[i] -= 1; }
            else if(assignments[i] == idx) { assignments[i] = 0; }
        }
    }

    // Update components
    destroy(model, model->clusters[idx]);
    for(uint32_t i = idx; i < (model->num_clusters - 1); i++) {
        model->clusters[i] = model->clusters[i + 1];
        model->clusters[i]->idx = i;
    }
    model->num_clusters -= 1;
}


/**
 * Remove empty components.
 * @param components components_t vector to remove from
 * @param assignments assignment vector; indices in assignments greater than
 *      the empty vector index are decremented.
 * @return true if a empty component was removed.
 */
bool remove_empty(struct mixture_model_t *model, uint16_t *assignments)
{
    #ifdef SHOW_TRACE
    printf("    remove_empty\n");
    #endif

    // Search for empty
    for(uint32_t i = 0; i < model->num_clusters; i++) {
        if((model->clusters)[i]->size == 0) {
            // Deallocate component; remove component from vector; update
            // assignments
            remove_component(model, assignments, i);
            return true;
        }
    }
    return false;
}



/** 
 * Get Marginal Log Likelihood for assignment of a single point
 * @param model parent model
 * @param cluster cluster to add to
 * @param point to get marginal log likelihood for
 * @return component_loglik_ratio(pt) * model_coefficient(pt)
 */
double marginal_loglik(
    struct mixture_model_t *model, Component *cluster, void *point)
{
    #ifdef SHOW_TRACE
    printf("    marginal_loglik\n");
    #endif

    return (
        model->comp_methods->loglik_ratio(
            cluster, model->comp_params, point) +
        model->model_methods->log_coef(
            model->model_params, cluster->size, model->num_clusters));
}


/** 
 * Get New Cluster Likelihood
 * @param model parent model
 * @param point point to evaluate
 * @return component_loglik_new(point) * model_log_coef_new()
 */
double new_cluster_loglik(struct mixture_model_t *model, void *point)
{
    #ifdef SHOW_TRACE
    printf("    new_cluster_loglik\n");
    #endif

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
void add_point(struct mixture_model_t *model, Component *cluster, void *point)
{
    #ifdef SHOW_TRACE
    printf("    add_point\n");
    #endif

    cluster->size += 1;
    model->comp_methods->add(cluster, model->comp_params, point);
}


/** 
 * Remove point
 * @param model parent model
 * @param cluster cluster to remove from
 * @param point point to remove
 */
void remove_point(
    struct mixture_model_t *model, Component *cluster, void *point)
{
    #ifdef SHOW_TRACE
    printf("    remove_point\n");
    #endif

    cluster->size -= 1;
    model->comp_methods->remove(cluster, model->comp_params, point);
}


/**
 * Get cluster at index (safely)
 * @param model : mixture_model_t struct to fetch from
 * @param idx index to fetch
 * @return fetched component; NULL if unsuccessful
 */
Component *get_cluster(struct mixture_model_t *model, uint32_t idx)
{
    assert(idx < model->num_clusters);
    return model->clusters[idx];
}
