
#include <stdint.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


/**
 * Look for empty cluster to be reused
 */
PyObject *try_reuse(PyObject *clusters, int n_clusters, int *index)
{
	PyObject *current_cluster;
	for(int i = 0; i < n_clusters; i++) {
		current_cluster = PyList_GetItem(clusters, i);
		if(PySet_Size(current_cluster) == 0) {
			index = i;
			return current_cluster;
		}
	}
	return NULL;
}


/**
 *	Compute mean and variance of subset of points indexed by a set
 */
void np_var(
	PyArrayObject *data_py, PyObject *cluster_set,
	PyArrayObject *mean_py, PyArrayObject *var_py)
{
	PyObject *cluster_iter = PyObject_GetIter(cluster_set);
	PyObject *item;

	if(cluster_iter == NULL) { // todo: error
	}

	// Fetch dimensions
	int dim = PyArray_DIM(data_py, 1);
	float *data = PyArray_DATA(data_py);
	npy_intp dims_mean[1] = {dim};
	npy_intp dims_var[2] = {dim, dim};

	// Allocate arrays
	mean_py = PyArray_SimpleNew(1, dims_mean, NPY_FLOAT);
	var_py = PyArray_SimpleNew(2, dims_var, NPY_FLOAT);
	float *mean = PyArray_DATA(mean_py);
	float *var = PyArray_DATA(var_py);

	// Compute mean and covariance matrix using iterator
	while(item = PyIter_Next(cluster_iter)) {

		long idx = PyLong_AsLong(item);

		// Compute E[X] and E[XX^T]
		for(int i = 0; i < dim; i++) {
			// Update mean vector E[X]
			mean[i] += data[dim * idx + i];
			// Update covariance vector E[XX^T]
			for(int j = 0; j < dim; j++) {
				var[dim * i + j] = (
					data[dim * idx + i] * data[dim * idx + j]
				);
			}
		}

		// Subtract E[X]E[X]^T from E[XX^T]
		for(int i = 0; i < dim; i++) {
			for(int j = 0; j < dim; j++) {
				var[dim * i + j] -= mean[i] * mean[j];
			}
		}

		// Clean up references for item
		Py_DECREF(item);
	}
}


/**
 *
 */
void crp_gibbs_iter(
	PyArrayObject *data, uint8_t *assignments, PyObject *clusters,
	int size, int dim,
	float r, float alpha, bool callback_var,
	PyObject *l_cond, PyObject *l_uncond)
{
	PyObject *args;
	PyObject *set;
	PyObject *l_cond_res;
	PyObject *l_uncond_res;

	// Run once for every point
	for(int pt = 0; pt < size; pt++) {
		// Weight vector
		int n_clusters = PyList_Size(clusters);
		float *weights = (float *) malloc(sizeof(float) * (n_clusters + 1));

		// Compute weight vector
		for(int i = 0; i < PyList_Size(clusters); i++) {

			// Get cluster set
			set = PyList_GetItem(clusters, i);

			// Run callback
			if(callback_var) {
				// Get mean and variance
				PyArrayObject *mean;
				PyArrayObject *var;
				np_var(PyArray_DATA(data), pt, mean, var);

				args = Py_BuildValue("OiOO", data, pt, set, mean, var);
			}
			else {
				args = Py_BuildValue("OiO", data, pt, set);
			}

			// Add result to weights
			l_cond_res = PyObject_CallObject(l_cond, args);
			Py_DECREF(args);
			weights[i] = (
				pow(PySet_Size(set), r) *
				((float) PyFloat_AsDouble(l_cond_res))
			);
		}

		// New cluster probability
		// Call unconditional probability
		args = PyBuildValue("Oi", data, pt);
		l_uncond_res = PyObject_CallObject(l_uncond, args);
		Py_DECREF(args);
		weights[n_clusters] = alpha * ((float) PyFloat_AsDouble(l_uncond_res));

		// Sample new cluster
		int assign = sample_proportional(weights, n_clusters);

		// Assign to new cluster
		if(assign == n_clusters) {
			// Try to look for empty cluster
			int empty_idx;
			PyObject *empty_set = try_reuse(clusters, n_clusters, &empty_idx);
			if(empty_set) {
				PySet_Add(empty_set, PyLong_FromLong((long) pt));
				assignments[pt] = empty_idx;
			}
			// No empty cluster -> add new
			else {
				PyObject *new_set = PySet_New(NULL);
				PySet_Add(new_set, PyLong_FromLong((long) pt));
				assignments[pt] = n_clusters;
			}
		}
		// Assign to existing cluster
		else {
			// Update set
			PySet_Add(
				PyList_GetItem(clusters, assign),
				PyLong_FromLong((long) pt));
			// Update assignment
			assignments[pt] = assign;
		}

		// Clean up
		free(weights);
	}
}


#define DOCSTRING_CRP_GIBBS_ITER \
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
    "l_cond : f(data, idx, cluster, mean, variance) -> float\n" \
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
    "    [default: 1]\n"

static char *crp_gibbs_kwlist[] = {
    "r",            // pCRP power
    "alpha",        // CRP new cluster coefficient
    "callback_var",
    NULL,
};

static PyObject *crp_gibbs_iter_py(
	PyObject *self, PyObject *args, PyObject *kwargs)
{
    // Inputs and intermediates
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *clusters;

    // Hyperparameters
    float r = 1.0;
    float alpha = 1.0;
    bool callback_var = true;

    // Likelihood functions
    PyObject *l_uncond;
    PyObject *l_cond;

    // Get args
    if(!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!OOO|ffp", crp_gibbs_kwlist,
            &PyArray_Type, &data_py,
            &PyArray_Type, &assignments_py,
            &clusters,
            &l_uncond,
            &l_cond,
            &r, &alpha, &callback_var,
        )) { return NULL; }

    // Check array bounds
    if(PyArray_NDIM(data_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data must have 2 dimensions.");
        return false;
    }
    if(PyArray_NDIM(assignments_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignment history must have 2 dimensions.");
        return false;
    }

    // Make sure likelihood functions are callable
    if (!(PyCallable_Check(l_uncond) && PyCallable_Check(l_cond))) {
        PyErr_SetString(
        	PyExc_TypeError, "Likelihood functions must be callable.");
        return NULL;
    }

    // Run gibbs sampler
    crp_gibbs_iter(
    	data, PyArray_DATA(assignments), clusters,
    	PyArray_DIM(data, 0), PyArray_DIM(data, 1),
    	r, alpha, callback_mean, callback_var,
    	l_cond, l_uncond);

    // Return None
    Py_RETURN_NONE;
}