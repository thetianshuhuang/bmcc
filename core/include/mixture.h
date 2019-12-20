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
 * Component Wrapper
 */
typedef struct {
    // Index of component
    uint32_t idx;
    // Number of points
    uint32_t size;
    // Inner Data
    void *data;
} Component;


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
    void (*destroy)(Component *component);
    // Add point
    void (*add)(Component *component, void *params, void *point);
    // Remove point
    void (*remove)(Component *component, void *params, void *point);

    /* Component Likelihoods */
    // Marginal Log Likelihood Ratio log(m(x_c+j)/m(x_c))
    double (*loglik_ratio)(Component *component, void *params, void *point);
    // Unconditional Log Likelihood log(m(x_j))
    double (*loglik_new)(void *params, void *point);
    // Split merge likelihood ratio P(c1)P(c2) / P(merged)
    double (*split_merge)(
        void *params, Component *merged, Component *c1, Component *c2);

    /* Debug and Utility */
    // Inspect current state (mostly for debug purposes)
    PyObject (*inspect)(void *params);

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

    // Log coefficient for split
    double (*log_split)(void *params, int nc, int n1, int n2);
    // Log coefficient for merge
    double (*log_merge)(void *params, int nc, int n1, int n2);

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
    // Number of points
    uint32_t size;
    // Number of dimemsions
    uint32_t dim;
    // Data Type; Numpy enumerated type
    int npy_type;
    // Stride (size of data type)
    int stride;
 
    /* Clusters */
    // Cluster vector size
    uint32_t mem_size;
    // Number of clusters
    uint32_t num_clusters;
    // Clusters
    Component **clusters;
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
    uint32_t size, uint32_t dim, int type);

// Destroy mixture model struct
void destroy_components(void *model);

// Create component (but do not bind)
Component *create_component(struct mixture_model_t *model);

// Add component to model
bool add_component(struct mixture_model_t *model, Component *component);

// Remove component from model
void remove_component(
    struct mixture_model_t *model, uint16_t *assignments, int idx);

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
