/**
 * Gibbs Sampler
 */

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL BAYESIAN_CLUSTERING_C_ARRAY_API
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdbool.h>

#include "../include/mixture.h"
#include "../include/misc_math.h"
#include "../include/type_check.h"
#include "../include/normal_wishart.h"

/**
 * Execute gibbs iteration.
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 * @return true if returned without error
 */
bool gibbs_iter(
    double *data, uint16_t *assignments,
    struct mixture_model_t *model)
{

    // Assignment weight vector: exponentially-overallocated
    double *weights = (double *) malloc(sizeof(double) * BASE_VEC_SIZE);
    int vec_size = BASE_VEC_SIZE;

    // For each sample:
    for(int idx = 0; idx < model->size; idx++) {

        double *point = &data[idx * model->dim];

        // Check assignments vector for error
        if(assignments[idx] >= model->num_clusters) {
            PyErr_SetString(
                PyExc_IndexError,
                "Assignment index greater than the number of clusters.");
            return false;
        }

        // Remove from currently assigned cluster
        model->comp_methods->remove(
            model->clusters[assignments[idx]], model->comp_params, point);

        // Remove empty components
        remove_empty(model, assignments);

        // Handle vector resizing
        if(model->num_clusters + 1 >= vec_size) {
            free(weights);
            vec_size *= 2;
            double *weights_new = (double *) malloc(sizeof(double) * vec_size);

            // Check for error
            if(weights_new == NULL) {
                free(weights);
                PyErr_SetString(
                    PyExc_MemoryError,
                    "Could not allocate weight vector in memory.");
                return false;
            }
            else { weights = weights_new; }
        }

        // Get assignment weights
        for(int i = 0; i < model->num_clusters; i++) {
            void *c_i = model->clusters[i];
            weights[i] = (
                model->comp_methods->loglik_ratio(
                    c_i, model->comp_params, point) +
                model->model_methods->log_coef(
                    model->model_params,
                    model->comp_methods->get_size(c_i),
                    model->num_clusters));
        }

        // Get new cluster weight
        weights[model->num_clusters] = (
            model->model_methods->log_coef_new(
                model->model_params,
                model->num_clusters) +
            model->comp_methods->loglik_new(model->comp_params, point));

        // Sample new
        int new = sample_log_weighted(weights, model->num_clusters + 1);

        // New cluster?
        if(new == model->num_clusters) {
            bool success_new = add_component(model);
            // Check for allocation failure
            if(!success_new) {
                free(weights);
                return NULL;
            }
        }

        // Update component
        model->comp_methods->add(
            model->clusters[new], model->comp_params, point);
        assignments[idx] = new;
    }

    free(weights);

    return true;
}


/**
 * Run Gibbs Iteration. See docstring (sourced from gibbs.h) for details on
 * Python calling.
 */
PyObject *gibbs_iter_py(PyObject *self, PyObject *args)
{
    // Get args
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *model_py;
    bool success = PyArg_ParseTuple(
        args, "O!O!O",
        &PyArray_Type, &data_py,
        &PyArray_Type, &assignments_py,
        &model_py);
    if(!success) { return NULL; }
    if(!type_check(data_py, assignments_py)) { return NULL; }

    // Unpack capsules
    struct mixture_model_t *model = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            model_py, MIXTURE_MODEL_API));

    // GIL free zone ----------------------------------------------------------    
    Py_INCREF(data_py);
    Py_INCREF(assignments_py);
    Py_INCREF(model_py);
    // Py_BEGIN_ALLOW_THREADS;

    bool gibbs_success = gibbs_iter(
        (double *) PyArray_DATA(data_py),
        (uint16_t *) PyArray_DATA(assignments_py),
        model);

    // Py_END_ALLOW_THREADS;
    Py_DECREF(data_py);
    Py_DECREF(assignments_py);
    Py_DECREF(model_py);
    // ------------------------------------------------------------------------

    // Check for exception
    if(!gibbs_success) { return NULL; }
    else { Py_RETURN_NONE; }
}
