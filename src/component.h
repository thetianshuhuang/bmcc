
#ifndef COMPONENT_H
#define COMPONENT_H


#include <stdint.h>
#include <Python.h>


#define COMPONENT_METHODS_API "bayesian_clustering_c.ComponentMethods"
#define COMPONENT_PARAMS_API "bayesian_clustering_c.ComponentParams"
#define COMPONENT_DATA_API "bayseian_clustering_c.Clusters"

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


#endif
