/**
 * Gibbs Sampler
 */

#include <Python.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../include/mixture/mixture.h"
#include "../include/misc_math.h"
#include "../include/type_check.h"
#include "../include/samplers/base_iter.h"

/**
 * Execute gibbs iteration.
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 * @param annealing annealing factor
 * @return true if returned without error
 */
bool gibbs_iter(
    void *data,
    uint16_t *assignments,
    struct mixture_model_t *model,
    double annealing)
{
    // Assignment weight vector: exponentially-overallocated
    // Make sure to check for number of clusters >> BASE_VEC_SIZE!
    uint32_t vec_size = BASE_VEC_SIZE;
    if(model->num_clusters > vec_size) {
        vec_size = model->num_clusters + 1;
    }
    double *weights = (double *) malloc(sizeof(double) * vec_size);

    // For each sample:
    for(uint32_t idx = 0; idx < model->size; idx++) {

        #ifdef SHOW_TRACE
        printf("gibbs_iter:%d\n", idx);
        #endif

        void *point = (char *) data + idx * model->dim * model->stride;

        // Check assignments vector for error
        if(assignments[idx] >= model->num_clusters) {
            PyErr_SetString(
                PyExc_IndexError,
                "Assignment index greater than the number of clusters.");
            return false;
        }

        // Remove current point from currently assigned cluster
        remove_point(model, model->clusters[assignments[idx]], point);

        // Remove empty components
        remove_empty(model, assignments);

        // Handle vector resizing
        while(model->num_clusters + 1 >= vec_size) {

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

        // Get assignment and new cluster weights
        for(uint32_t i = 0; i < model->num_clusters; i++) {
            weights[i] = annealing * marginal_loglik(
                model, model->clusters[i], point);
        }
        weights[model->num_clusters] = (
            annealing * new_cluster_loglik(model, point));

        #ifdef SHOW_LIKELIHOODS
        printf("  weights: ");
        for(uint32_t i = 0; i <= model->num_clusters; i++) {
            printf("%f ", weights[i]);
        }
        printf("\n");
        #endif

        // Sample new
        uint32_t new = sample_log_weighted(weights, model->num_clusters + 1);

        // New cluster?
        #ifdef SHOW_TRACE
        printf("  new: %d [k=%d]\n", new, model->num_clusters);
        #endif

        if(new == model->num_clusters) {
            bool success_new = add_component(model, NULL);
            // Check for allocation failure
            if(!success_new) {
                #ifdef SHOW_FAILURE
                printf("FAILURE: couldn't add new component\n");
                #endif
                free(weights);
                return false;
            }
        }

        // Update component
        add_point(model, model->clusters[new], point);
        assignments[idx] = new;
    }

    free(weights);
    return true;
}


/**
 * Run Gibbs Iteration. See docstring (sourced from gibbs.h) for details on
 * Python calling.
 */
PyObject *gibbs_iter_py(PyObject *self, PyObject *args, PyObject *kwargs)
{
    return base_iter(self, args, kwargs, &supports_gibbs, &gibbs_iter);
}
