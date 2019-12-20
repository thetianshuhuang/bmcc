/**
 * Core Mixture Structs and Routines
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>

#include "../include/type_check.h"
#include "../include/mixture.h"


#include <stdio.h>


// ----------------------------------------------------------------------------
//
//                           Create/Destroy Mixture
//
// ----------------------------------------------------------------------------

/**
 * Create components struct
 * @param comp_methods : Component model specifications
 * @param model_methods : Mixture model specifications
 * @param params : Python dict params
 * @param size : Number of data points
 * @param dim : Number of dimensions
 * @param type : Numpy enumerated type
 * @return allocated components_t; initialized empty
 */
struct mixture_model_t *create_mixture(
    ComponentMethods *comp_methods,
    ModelMethods *model_methods,
    PyObject *params,
    uint32_t size, uint32_t dim, int type)
{
    // Allocate vector
    struct mixture_model_t *mixture = (
        (struct mixture_model_t *) malloc(sizeof(struct mixture_model_t)));
    mixture->mem_size = BASE_VEC_SIZE;
    mixture->num_clusters = 0;
    mixture->clusters = (
        (Component **) malloc(sizeof(Component *) * mixture->mem_size));

    // Bind methods, params, dim
    mixture->comp_methods = comp_methods;
    mixture->model_methods = model_methods;

    mixture->comp_params = comp_methods->params_create(params);
    if(mixture->comp_params == NULL) {
        goto error;
    }

    mixture->model_params = model_methods->create(params);
    if(mixture->model_params == NULL) {
        comp_methods->params_destroy(mixture->comp_params);
        goto error;
    }

    // Size
    mixture->size = size;
    mixture->dim = dim;
    mixture->npy_type = type;
    mixture->stride = type_get_size(type);

    return mixture;

    // Error condition: free all and return NULL
    error:
        free(mixture->clusters);
        free(mixture);
        return NULL;
}


/**
 * Destroy mixture model struct
 * @param model : mixture_model_t struct to destroy
 */
void destroy_mixture(PyObject *model_py)
{
    struct mixture_model_t *model_tc = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            model_py, MIXTURE_MODEL_API));
    for(int i = 0; i < model_tc->num_clusters; i++) {
        model_tc->comp_methods->destroy(model_tc->clusters[i]);
        free(model_tc->clusters[i]);
    }
    free(model_tc);
}


// ----------------------------------------------------------------------------
//
//                          Create/Destroy Component
//
// ----------------------------------------------------------------------------


/**
 * Create new component (not bound)
 * @param model mixture_model_t struct to create component from
 * @return Created Component * container
 */
Component *create_component(struct mixture_model_t *model)
{
    Component *container = (Component *) malloc(sizeof(Component));
    container->size = 0;
    container->data = model->comp_methods->create(model->comp_params);
    return container;
}


/**
 * Add Component: allocates new component, and appends to components capsule
 * @param model mixture_model_t struct containing components
 * @param component component to add; if NULL, allocates a new component
 * @return true if addition successful
 */
bool add_component(struct mixture_model_t *model, Component *component)
{
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
    struct mixture_model_t *model, uint16_t *assignments, int idx)
{
    if(idx >= model->num_clusters) {
        printf("[C BACKEND ERROR] Invalid cluster: %d\n", idx);
        return;
    }

    // Optionally update assignments
    if(assignments != NULL) {
        for(int i = 0; i < model->size; i++) {
            if(assignments[i] > idx) { assignments[i] -= 1; }
        }
    }

    // Update components
    model->comp_methods->destroy(model->clusters[idx]);
    free(model->clusters[idx]);
    for(int i = idx; i < (model->num_clusters - 1); i++) {
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
    // Search for empty
    for(int i = 0; i < model->num_clusters; i++) {
        if((model->clusters)[i]->size == 0) {
            // Deallocate component; remove component from vector; update
            // assignments
            remove_component(model, assignments, i);
            return true;
        }
    }
    return false;
}


// ----------------------------------------------------------------------------
//
//                             Capsule Management
//
// ----------------------------------------------------------------------------

/**
 * Initialize model capsules. See docstring (sourced from mixture.h) for more
 * details.
 */
PyObject *init_model_capsules_py(PyObject *self, PyObject *args)
{
    // Get args
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *comp_methods_py;
    PyObject *model_methods_py;
    PyObject *params;
    bool success = PyArg_ParseTuple(
        args, "O!O!OOO",
        &PyArray_Type, &data_py,
        &PyArray_Type, &assignments_py,
        &comp_methods_py, &model_methods_py,
        &params);
    if(!success) { return NULL; }

    // Unpack arrays
    uint16_t *asn = PyArray_DATA(assignments_py);
    void *data = PyArray_DATA(data_py);
    int size = PyArray_DIM(data_py, 0);
    int dim = PyArray_DIM(data_py, 1);
    int data_type = PyArray_TYPE(data_py);

    // Check type
    if(!type_check(data_py, assignments_py)) { return NULL; }

    // Unpack capsules
    ComponentMethods *comp_methods = (
        (ComponentMethods *) PyCapsule_GetPointer(
            comp_methods_py, COMPONENT_METHODS_API));
    ModelMethods *model_methods = (
        (ModelMethods *) PyCapsule_GetPointer(
            model_methods_py, MODEL_METHODS_API));

    // Create mixture_t
    struct mixture_model_t *mixture = create_mixture(
        comp_methods, model_methods, params, size, dim, data_type);
    if(mixture == NULL) { return NULL; }

    // Get number of components
    int max = 0;
    for(int i = 0; i < size; i++) {
        if(asn[i] > max) { max = asn[i]; }
    }

    // Allocate enough components
    for(int i = 0; i <= max; i++) {
        bool success_add = add_component(mixture, NULL);

        // Check for error
        if(!success_add) {
            for(int j = 1; j < i; j++) { remove_component(mixture, NULL, 0); }
            free(mixture->clusters);
            free(mixture);
            return NULL;
        }
    }

    // Add points
    for(int i = 0; i < size; i++) {
        add_point(
            mixture, mixture->clusters[asn[i]], 
            (void *) (data + i * dim * mixture->stride));
    }

    return PyCapsule_New(mixture, MIXTURE_MODEL_API, &destroy_mixture);
}


/**
 * Update mixture model hyperparameters
 */
PyObject *update_mixture_py(PyObject *self, PyObject *args)
{
    PyObject *mixture;
    PyObject *update;
    if(!PyArg_ParseTuple(args, "OO", &mixture, &update)) { return NULL; }

    struct mixture_model_t *mixture_tc = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            mixture, MIXTURE_MODEL_API));

    if(mixture_tc->model_methods->update != NULL) {
        mixture_tc->model_methods->update(mixture_tc->model_params, update);
    }

    Py_RETURN_NONE;
}


/**
 * Update component hyperparameters
 */
PyObject *update_components_py(PyObject *self, PyObject *args)
{
    PyObject *mixture;
    PyObject *update;
    if(!PyArg_ParseTuple(args, "OO", &mixture, &update)) { return NULL; }

    struct mixture_model_t *mixture_tc = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            mixture, MIXTURE_MODEL_API));

    if(mixture_tc->comp_methods->update != NULL) {
        mixture_tc->comp_methods->update(mixture_tc->comp_params, update);
    }

    Py_RETURN_NONE;
}
