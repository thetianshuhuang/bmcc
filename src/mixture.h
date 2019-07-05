

#ifndef MIXTURE_H
#define MIXTURE_H


#include <stdint.h>
#include <Python.h>


#define COMPONENT_METHODS_API "bayesian_clustering_c.ComponentMethods"
#define COMPONENT_PARAMS_API "bayesian_clustering_c.ComponentParams"

#define MODEL_METHODS_API "bayesian_clustering_c.ModelMethods"
#define MODEL_PARAMS_API "bayesian_clustering_c.ModelParams"

#define MIXTURE_MODEL_API "bayesian_clustering_c.MixtureModel"

// ----------------------------------------------------------------------------
//
//                             Struct Definitions
//
// ----------------------------------------------------------------------------

/**
 * Component Methods
 * Definitions for within-component prior and posterior
 */
typedef struct component_methods_t {

    /* Hyperparameters */
    // Convert python dictionary to hyperparameters struct
    void* (*params_create)(PyObject *dict);
    // Destroy hyperparameters struct
    void (*params_destroy)(void *params);

    /* Component management */
    // Allocate component capsule
    void* (*create)(void *params);
    // Destroy component capsule
    void (*destroy)(void *component);
    // Get size
    void get_size(void *component);
    // Add point
    void add(void *component, float *point);

    /* Component Likelihoods */
    // Remove point
    void remove(void *component, float *point);
    // Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
    double loglik_ratio(void *component, void *params, float *point);
    // Unconditional Log Likelihood log(m(x_j))
    double loglik_new(void *params, float *point);

} ComponentMethods;


/**
 * Model Methods
 * Definitions for cluster mixing model prior and posterior
 */
typedef struct model_methods_t {

    // Create hyperparameters from dictionary
    void* (*create)(PyObject *dict);
    // Destroy hyperparameters
    void (*destroy)(void *parameters);
    // Log coefficients for existing clusters
    double (*log_coef)(void *params, int size);
    // Log coefficient for new cluster
    double (*log_coef_new)(void *params);

} ModelMethods;


/**
 * Complete Mixture Model
 */
struct mixture_model_t {

    /* Component parameters and methods */
    // Methods
    ComponentMethods *comp_methods;
    // Hyperparameters
    void *comp_params;

    /* Model parameters and methods */
    // Methods
    ModelMethods *model_methods;
    // Hyperparameters
    void *model_params;

    /* Data */
    uint32_t size;
    uint32_t dim;

    /* Clusters */
    // Cluster vector size
    uint32_t mem_size;
    // Number of clusters
    uint32_t num_clusters;
    // Clusters
    void *clusters;
}


// ----------------------------------------------------------------------------
//
//                           Mixture Model Routines
//
// ----------------------------------------------------------------------------

// Create mixture model struct
struct mixture_model_t *create_components();
// Destroy mixture model struct
void destroy_components(void *model);
// Add component to model
void add_component(struct mixture_model_t *model);
// Remove component from model
void remove_component(struct mixture_model_t *model, int idx);
// Remove empty component
bool remove_empty(struct mixture_model_t *model, uint16_t *assignments);


#endif
