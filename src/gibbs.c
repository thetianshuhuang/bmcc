/**
 *
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdbool.h>

#include <misc_math.h>

#define BASE_VEC_SIZE = 32


/**
 * Remove empty components.
 * @param components components_t vector to remove from
 * @param comp_methods CompMethods struct containing dealloc routines
 * @param assignments assignment vector; indices in assignments greater than
 *      the empty vector index are decremented.
 * @param size number of data points
 * @return true if a empty component was removed.
 */
bool remove_empty(
    struct component_t *components,
    ComponentMethods comp_methods,
    uint16_t *assignments, uint32_t size)
{
    // Search for empty
    for(int i = 0; i < components->size; i++) {
        if(get_size((components->values)[i]) == 0) {
            // Deallocate component; remove component from vector
            remove_component(components, comp_methods, i);
            // Decrement values
            for(int j = 0; j < size; j++) {
                if(assignments[j] > i) { assignments[j] -= 1; }
            }
            return true;
        }
    }
    return false;
}


/**
 * Execute gibbs iteration.
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 *      Component methods are responsible for casting to the correct type.
 * @param comp_methods component methods for likelihoods, updates
 * @param model_methods model methods for coefficients
 * @param comp_params component parameters
 * @param model_params model parameters
 */
void gibbs_iter(
    float *data, uint16_t *assignments,
    struct components_t *components,
    ComponentMethods *comp_methods,
    ModelMethods *model_methods, 
    void *comp_params,
    void *model_params,
    int size)
{
    // Assignment weight vector: exponentially-overallocated
    double *weights = (double *) malloc(sizeof(double) * BASE_VEC_SIZE);
    int vec_size = BASE_VEC_SIZE;

    // For each sample:
    for(int idx = 0; idx < size) {

        float *point = &data[idx * comp_params->dim];

        // Remove from currently assigned cluster
        comp_methods->remove((components->values)[assignments[idx]]);
        // Remove empty components
        remove_empty(components, comp_methods, assignments, size);

        // Handle vector resizing
        if(components->size + 1 > weight_vec_size) {
            free(weights);
            vec_size *= 2;
            double *weights = (double *) malloc(sizeof(double) * vec_size);
        }

        // Get assignment weights
        for(int i = 0; i < components->size; i++) {
            void *c_i = (components->values)[i];
            weights[i] = (
                comp_methods->loglik_ratio(c_i, params, point) +
                model_methods->log_coef(
                    params, get_size(c_i), components->size));
        }

        // Get new cluster weight
        weights[components->size] = (
            model_methods->log_coef_new(params, components->size) *
            comp_methods->loglik_new(params, point));

        // Sample new
        int new = sample_log_weighted(weights, components->size + 1);
        // New cluster?
        if(new == components->size) {
            add_component(components, comp_methods, params);
        }
        // Update component
        comp_methods->add((components->values)[new], params, point);
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
    "    Dictionary containing model hyperparameters and component "
        "parameters.\n"

static PyObject *gibbs_iter_py(PyObject *args)
{
    // Get args
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *clusters_py;
    PyObject *params_py;
    bool success = PyArg_ParseTuple(
        args, "O!O!OOO",
        &PyArray_Type, &data, &PyArray_Type, &assignments,
        &clusters_py, &params_py);
    if(!success) { return NULL; }

    // Check types
    if(PyArray_TYPE(data_py) != NPY_FLOAT32) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data must have type float32.");
        return NULL;
    }
    if(PyArray_TYPE(assignments_py) != NPY_UINT16) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignments must have type uint16.");
        return NULL;
    }

    // Check dimensions
    if(PyArray_NDIM(data_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data must have 2 dimensions.");
        return NULL;
    }
    if(PyArray_DIM(data_py, 0) != PyArray_DIM(assignments_py, 0)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignments vector length does not match data.");
        return NULL;
    }
    long ndim_params = PyLong_AsLong(PyDict_GetItemString(params_py, "dim"));
    if(PyArray_DIM(data, 1) != ndim_params) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data dimensions do not match dimensions in parameters.");
        return NULL;
    }

    // Unpack capsules
    ComponentMethods *comp_methods = (
        (ComponentMethods *) PyCapsule_GetPointer(
            PyDict_GetItemString(params_py, "comp_methods"),
            "bayesian_clustering_c.ComponentMethods"));
    ModelMethods *model_methods = (
        (ModelMethods *) PyCapsule_GetPointer(
            PyDict_GetItemString(params_py, "model_methods"),
            "bayesian_clustering_c.ModelMethods"));
    struct components_t *clusters = (
        (struct components_t *clusters) PyCapsule_GetPointer(
            clusters_py,
            "bayseian_clustering_c.Clusters"));
    void *comp_params = PyCapsule_GetPointer(
        PyDict_GetItemString(params_py, "comp_params"));
    void *model_params = PyCapsule_GetPointer(
        PyDict_GetItemString(params_py, "model_params"));

    // GIL free zone ----------------------------------------------------------
    Py_BEGIN_ALLOWED_THEADS

    gibbs_iter(
        (float *) PyArray_DATA(data_py),
        (uint16_t *) PyArray_DATA(assignments_py),
        clusters,
        comp_methods, model_methods,
        comp_params, model_params,
        PyArray_DIM(data, 0));

    Py_END_ALLOW_THREADS
    // ------------------------------------------------------------------------

    Py_RETURN_NONE
}
