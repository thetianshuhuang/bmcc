/**
 *
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdbool.h>

#include <misc_math.h>
#include <type_check.h>

#define BASE_VEC_SIZE = 32


/**
 * Execute gibbs iteration.
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 */
void gibbs_iter(
    float *data, uint16_t *assignments,
    struct mixture_model_t *model)
{
    // Assignment weight vector: exponentially-overallocated
    double *weights = (double *) malloc(sizeof(double) * BASE_VEC_SIZE);
    int vec_size = BASE_VEC_SIZE;

    // For each sample:
    for(int idx = 0; idx < model->size) {

        float *point = &data[idx * model->dim];

        // Remove from currently assigned cluster
        model->comp_methods->remove((model->clusters)[assignments[idx]]);
        // Remove empty components
        remove_empty(model, assignments, size);

        // Handle vector resizing
        if(model->num_clusters + 1 > weight_vec_size) {
            free(weights);
            vec_size *= 2;
            double *weights = (double *) malloc(sizeof(double) * vec_size);
        }

        // Get assignment weights
        for(int i = 0; i < model->num_clusters; i++) {
            void *c_i = (model->clusters)[i];
            weights[i] = (
                model->comp_methods->loglik_ratio(
                    c_i, model->comp_params, point) +
                model->model_methods->log_coef(
                    model->model_params, get_size(c_i), model->num_clusters));
        }

        // Get new cluster weight
        weights[model->num_clusters] = (
            model->model_methods->log_coef_new(
                model->model_params, model->num_clusters) *
            model->comp_methods->loglik_new(
                comp->comp_params, point));

        // Sample new
        int new = sample_log_weighted(weights, model->num_clusters + 1);
        // New cluster?
        if(new == model->num_clusters) {
            add_component(model);
        }
        // Update component
        model->comp_methods->add((model->clusters)[new], point);
    }
}


#define DOCSTRING_GIBBS_ITER \
    "Run Gibbs sampler for one iteration.\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data : np.array\n" \
    "    Data matrix. Should have type np.float32; row-major order.\n" \
    "assignments : np.array\n" \
    "    Assignment array. Should have type np.uint16.\n" \
    "clusters : capsule\n" \
    "    Capsule containing cluster data.\n" \
    "params : dict\n" \
    "    Dictionary containing model and component capsules.\n"

static PyObject *gibbs_iter_py(PyObject *args)
{
    // Get args
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *model_py;
    bool success = PyArg_ParseTuple(
        args, "O!O!OOO",
        &PyArray_Type, &data,
        &PyArray_Type, &assignments,
        &model_py);
    if(!success) { return NULL; }
    if(!type_check(data_py, assignments_py)) { return NULL; }

    // Unpack capsules
    struct mixture_model_t *model = (
        (struct mixture_model_t *model_py) PyCapsule_GetPointer(
            clusters_py, COMPONENT_DATA_API));

    // GIL free zone ----------------------------------------------------------
    Py_BEGIN_ALLOWED_THEADS

    gibbs_iter(
        (float *) PyArray_DATA(data_py),
        (uint16_t *) PyArray_DATA(assignments_py),
        model,
        PyArray_DIM(data, 0));

    Py_END_ALLOW_THREADS
    // ------------------------------------------------------------------------

    Py_RETURN_NONE
}
