#include <cstdlib>
#include <algorithm>

#include "util.h"


#define DOCSTRING_CRP_GIBBS \
    "Run gibbs sampler for bayesian clustering using pCRP prior\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data : np.array\n" \
    "    Data matrix. Points are stored along dimension 0 (i.e. data[i]) is " \
        "a single\n" \
    "    point\n" \
    "assignments : np.array\n"\
    "    Assignment matrix. Should have assignments.shape[1] = data.shape[0] " \
        "and\n" \
    "    assignments.shape[0] = floor((iterations - burn_in) / thinning).\n" \
    "l_uncond : f(data, idx) -> float\n"\
    "    Unconditional likelihood function p(x_i | beta) where beta consists " \
        "of all\n" \
    "    model hyperparameters (distribution of means, etc)\n" \
    "l_cond : f(data, cluster, mean, variance) -> float\n" \
    "    Conditional likelihood function p(z_i = k | z_{!=i}, alpha); the " \
        "likelihood\n" \
    "    function used by the gibbs update step\n" \
    "\n" \
    "Keyword Args\n" \
    "------------\n" \
    "r : float\n" \
    "    pCRP cluster weighting power parameter\n" \
    "    [default: 1 (no power)]\n" \
    "alpha : float\n" \
    "    CRP new cluster probability coefficient\n" \
    "    [default: 1]\n" \
    "thinning : int\n" \
    "    Thinning ratio for gibbs sampler; keep 1/thinning samples\n" \
    "    [default: 1 (1/1 samples kept)]\n" \
    "iterations : int\n" \
    "    Number of iterations to run the gibbs sampler\n" \
    "    [default: 50]\n" \
    "burn_in : int\n" \
    "    Number of burn-in iterations to discard\n" \
    "    [default: 20]\n" \


static char *crp_gibbs_kwlist[] = {
    "r",            // pCRP power
    "alpha",        // CRP new cluster coefficient
    "thinning",     // Keep 1/thinning gibbs samples
    "iterations",   // Number of iterations
    "burn_in",      // Gibbs sampler burn in period
    NULL};

PyObject *crp_gibbs(PyObject *self, PyObject *args, PyObject *kwargs)
{
    // Input arrays
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;

    // Hyperparameters
    float r = 1.0;
    float alpha = 1.0;
    int thinning = 1;
    int iterations = 100;
    int burn_in = 20;

    // Get args
    if(!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!|ffiii", crp_gibbs_kwlist,
            &PyArray_Type, &data_py,
            &PyArray_Type, &assignments_py,
            &r, &alpha, &thinning, &iterations, &burn_in
        )) { return NULL; }

    // Check sizes
    int min_hist_size = (iterations - burn_in) / thinning;
    if(!gibs_crp_dim_check(data, assignments, min_hist_size)) {
        return NULL;
    }

    // Use internal data structures for subroutines
    float *data = PyArray_DATA(data);
    int size = PyArray_DIM(data, 0);
    int dim = PyArray(data, 1);

    // Get initial assignments
    int *assignments = PyArray_DATA(assignments);
    clusters = init_clusters(assignments, size);

    // Assignment vector to iterate on
    int *assignments_current = new int[size];
    for(int i = 0; i < size; i++) { assignments_current[i] = assignments[i]; }

    // Run gibbs sampler
    int history_index = 1;
    for(int i = 1; i <= iterations; i++) {

        gibbs_iteration(
            data, assignments_current, clusters,
            size, dim, r, alpha,
            l_cond, l_uncond);

        // Copy over to history if past burn-in period
        if((i >= burn_in) && (i % thinning == 0)) {
            for(int j = 0; j < size; j++) {
                assignments[history_index * size + j] = assignments_current[j];
            }
            history_index += 1;
        }
    }

    // Clean up
    delete assignments_current;
    for(int i = 0; i < clusters.size(); i++) { delete clusters[i]; }
    delete clusters;

    Py_RETURN_NONE;
}


static PyMethodDef ModuleMethods[] = {
    {"crp_gibbs", (PyCFunction) crp_gibbs, METH_VARARGS, DOCSTRING_CRP_GIBBS},
    {NULL, NULL, 0, NULL}
}


static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "crp",
    "docstring...",
    -1,
    ModuleMethods
}


PyMODINIT_FUNC PyInit_test() {
    import_array()
    return PyModule_Create(&ModuleDef)
}
