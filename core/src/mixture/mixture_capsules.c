/**
 * Capsule Manipulation Functions
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdbool.h>

#include "../include/type_check.h"
#include "../include/mixture/mixture.h"
#include "../include/mixture/mixture_create.h"


/**
 * Initialize model capsules. See docstring (sourced from mixture.h) for more
 * details.
 */
PyObject *init_model_capsules_py(PyObject *Py_UNUSED(self), PyObject *args)
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
            for(int j = 0; j < i; j++) { remove_component(mixture, NULL, 0); }
            free(mixture->clusters);
            free(mixture);
            return NULL;
        }
    }

    // Add points
    for(int i = 0; i < size; i++) {
        add_point(
            mixture, mixture->clusters[asn[i]], 
            (void *) ((char *) data + i * dim * mixture->stride));
    }

    return PyCapsule_New(mixture, MIXTURE_MODEL_API, &destroy_mixture);
}


/**
 * Update mixture model hyperparameters
 */
PyObject *update_mixture_py(PyObject *Py_UNUSED(self), PyObject *args)
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
PyObject *update_components_py(PyObject *Py_UNUSED(self), PyObject *args)
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


/**
 * Inspect Model
 */
PyObject *inspect_mixture_py(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *mixture;
    if(!PyArg_ParseTuple(args, "O", &mixture)) { return NULL; }

    struct mixture_model_t *mixture_tc = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            mixture, MIXTURE_MODEL_API));

    // Check validity
    if(mixture_tc == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "inspect_mixture requires a MixtureModel capsule.");
        return NULL;
    }
    if(mixture_tc->comp_methods->inspect == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "this mixture model does not implement inspect().");
        return NULL;
    }

    // Inspect
    return mixture_tc->comp_methods->inspect(mixture_tc->comp_params);
}


/**
 * Get number of clusters
 */
PyObject *count_clusters_py(PyObject *Py_UNUSED(self), PyObject *args)
{
    PyObject *mixture;
    if(!PyArg_ParseTuple(args, "O", &mixture)) { return NULL; }

    struct mixture_model_t *mixture_tc = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            mixture, MIXTURE_MODEL_API));

    // Check validity
    if(mixture_tc == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "count_clusters requires a MixtureModel capsule.");
        return NULL;
    }

    return Py_BuildValue("i", mixture_tc->num_clusters);
}

