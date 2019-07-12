/**
 * Core Mixture Structs and Routines
 */

#ifndef MIXTURE_H
#define MIXTURE_H


#include <stdint.h>
#include <stdbool.h>
#include <Python.h>


// ----------------------------------------------------------------------------
//
//                             Struct Definitions
//
// ----------------------------------------------------------------------------

/**
 * Component Methods
 * Definitions for within-component prior and posterior
 */
typedef struct {

    /* Hyperparameters */
    // Convert python dictionary to hyperparameters struct
    void* (*params_create)(PyObject *dict);
    // Destroy hyperparameters struct
    void (*params_destroy)(void *params);
    // Update hyperparameters
    void (*update)(void *params, PyObject *dict);

    /* Component management */
    // Allocate component capsule
    void* (*create)(void *params);
    // Destroy component capsule
    void (*destroy)(void *component);
    // Get size
    int (*get_size)(void *component);
    // Add point
    void (*add)(void *component, void *params, double *point);

    /* Component Likelihoods */
    // Remove point
    void (*remove)(void *component, void *params, double *point);
    // Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
    double (*loglik_ratio)(void *component, void *params, double *point);
    // Unconditional Log Likelihood log(m(x_j))
    double (*loglik_new)(void *params, double *point);

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
    // Update hyperparameters
    void (*update)(void *parameters, PyObject *dict);

    // Log coefficients for existing clusters
    double (*log_coef)(void *params, int size, int nc);
    // Log coefficient for new cluster
    double (*log_coef_new)(void *params, int nc);

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
    void **clusters;
};


// ----------------------------------------------------------------------------
//
//                           Mixture Model Routines
//
// ----------------------------------------------------------------------------

// Create mixture model struct
struct mixture_model_t *create_mixture(
    ComponentMethods *comp_methods,
    ModelMethods *model_methods,
    PyObject *params,
    uint32_t size, uint32_t dim);
// Destroy mixture model struct
void destroy_components(void *model);
// Add component to model
bool add_component(struct mixture_model_t *model);
// Remove component from model
void remove_component(struct mixture_model_t *model, int idx);
// Remove empty component
bool remove_empty(struct mixture_model_t *model, uint16_t *assignments);


// ----------------------------------------------------------------------------
//
//                               Python Exports
//
// ----------------------------------------------------------------------------

#define DOCSTRING_INIT_MODEL_CAPSULES \
    "Initialize model capsules\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data_py : np.array\n" \
    "    Data array\n" \
    "assignments_py : np.array\n" \
    "    Assignment array\n" \
    "comp_methods : capsule\n" \
    "    Capsule containing ComponentMethods struct\n" \
    "model_methods : capsule\n" \
    "    Capsule containing ModelMethods struct\n" \
    "params : dict\n" \
    "    Dictionary containing hyperparameters.\n" \
    "\n" \
    "Returns\n" \
    "-------\n" \
    "capsule\n" \
    "    Capsule containing the created struct mixture_model_t"

PyObject *init_model_capsules_py(PyObject *self, PyObject *args);


#define DOCSTRING_UPDATE_MIXTURE \
    "Update mixture model hyperparameters\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "mixture : capsule\n" \
    "    Capsule containing mixture struct to update hyperparameters for\n" \
    "update : dict\n" \
    "    Dictionary to update values with"

PyObject *update_mixture_py(PyObject *self, PyObject *args);


#define DOCSTRING_UPDATE_COMPONENTS \
    "Update component hyperparameters\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "mixture : capsule\n" \
    "    Capsule containing mixture struct to update hyperparameters for\n" \
    "update : dict\n" \
    "    Dictionary to update values with"

PyObject *update_components_py(PyObject *self, PyObject *args);


#endif
