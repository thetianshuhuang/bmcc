
#include <stdint.h>
#include <crp.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/**
 * Sample proportionally from a weight vector
 * @param weights : list of weights
 * @param size : length of weight vector
 * @return index sampled from the len(weights) proportional to weight
 */
int sample_proportional(float *weights, int *size)
{
    // Sample from [0, sum_weights)
    // Saves k flops compared to normalizing first
    float sum_weights = 0;
    for(int i = 0; i < size; i++) { sum_weights += weights[i]; }
    float unif = sum_weights * (float) ((double) rand() / ((double) RAND_MAX));

    // Check cumulative sum
    float acc;
    for(int i = 0; i < size; i++) {
        acc += weights[size];
        if(unif < acc) { return i; }
    }

    // Catch-all in case of rounding error (just in case)
    return size - 1;
}

/**
 * Look for empty cluster to be reused
 */
int try_reuse(PyObject *clusters)
{
	for(int i = 0; i < PySet_Size(clusters); i++) {
		if(PySet_Contains(clusters, PyLong_FromLong((long) i)) == 0) {
			return i;
		}
	}
	return -1;
}


/**
 *	Compute mean and variance of subset of points indexed by a set
 */
int np_var(struct crp_gibbs_profile_t profile, int set_idx)
{
	// Set up arrays
	float *data = PyArray_DATA(profile->data);
	npy_intp dims_mean[1] = {profile->dim};
	npy_intp dims_var[2] = {profile->dim, profile->dim};

	// Allocate arrays
	mean_py = PyArray_SimpleNew(1, dims_mean, NPY_FLOAT);
	var_py = PyArray_SimpleNew(2, dims_var, NPY_FLOAT);
	float *mean = PyArray_DATA(mean_py);
	float *var = PyArray_DATA(var_py);

	// Keep track of n to save compute later
	int n = 0;

	// Compute mean and covariance matrix using iterator
	for(int idx = 0; idx < profile->size; idx++) {
		if((profile->assignments)[idx] == set_idx) {
			// Compute E[X] and E[XX^T]
			for(int i = 0; i < profile->dim; i++) {
				int unr_i = profile->dim * idx + i;
				// Update mean vector E[X]
				mean[i] += data[unr_i];
				// Update covariance vector E[XX^T]
				for(int j = 0; j < profile->dim; j++) {
					var[profile->dim * i + j] = (
						data[unr_i] * data[profile->dim * idx + j]);
				}
			}
			n += 1;
		}
	}

	// Subtract E[X]E[X]^T from E[XX^T]
	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			var[dim * i + j] -= mean[i] * mean[j];
			var[dim * i + j] /= n;
		}
	}

	return n;
}


void weight_vector(
	struct crp_gibbs_profile_t profile, float *weights, int n_clusters, int pt)
{

	// Clear weight vector
	for(int i = 0; i < n_clusters; i++) { weights[i] = 0; }

	int last_empty;

	// Current clusters probability
	PyObject *item;
	PyObject *cluster_iter = PyObject_GetIter(profile->clusters);
	while(item = PyIter_Next(cluster_iter)) {

		// Get cluster set index
		int set_idx = PyLong_AsLong(item);
		int cluster_size = 0;
		Py_DECREF(item);

		// Set up callback args
		if(callback_var) {
			// Get mean and variance
			PyArrayObject *mean;
			PyArrayObject *var;
			cluster_size = np_var(
				profile->data, PyArray_DATA(profile->assignments),
				set_idx, &mean, &var);

			args = Py_BuildValue(
				"OiOiOO", profile->data, pt, profile->assignments,
				set_idx, mean, var);
		}
		else {
			args = Py_BuildValue("OiO", profile->data, pt, set_idx);
			for(int i = 0; i < profile->size; i++) {
				if((profile->assignments)[i] == set_idx) { cluster_size += 1; }
			}
		}

		if(cluster_size == 0) {
			last_empty = set_idx;
			weights[i] = 0;
		}
		else {
			// Run callback
			l_cond_res = PyObject_CallObject(profile->l_cond, args);
			Py_DECREF(args);

			// Set weights
			weights[i] = (
				pow(PySet_Size(cluster_size), profile->r) *
				((float) PyFloat_AsDouble(l_cond_res))
			);
		}
	}

	// New cluster probability
	// Call unconditional probability
	args = PyBuildValue("Oi", profile->data, pt);
	l_uncond_res = PyObject_CallObject(profile->l_uncond, args);
	Py_DECREF(args);
	weights[n_clusters] = (
		profile->alpha * ((float) PyFloat_AsDouble(l_uncond_res))
	);
}



/**
 *
 */
void crp_gibbs_iter(crp_gibbs_profile_t *profile)
{
	PyObject *args;
	PyObject *l_cond_res;
	PyObject *l_uncond_res;

	int weights_size = PySet_Size(profile->clusters) + 1;
	float *weights = (float *) malloc(sizeof(float) * weights_size);

	// Run once for every point
	for(int pt = 0; pt < size; pt++) {

		// Remove from current cluster
		(profile->assignments)[pt] = -1;

		// Weight vector
		int n_clusters = PySet_Size(profile->clusters);
		// Check if realloc is necessary
		// Uses exponential over-allocation strategy
		if(n_clusters + 1 > weights_size) {
			weights_size *= 2;
			weights = realloc(weights, sizeof(float) * weights_size);
		}
		weight_vector(profile, weights, n_clusters, pt);

		// Sample new cluster
		int assign = sample_proportional(weights, n_clusters);

		// Assign to new cluster
		if(assign == n_clusters) {
			// Try to look for empty cluster
			int empty_idx = try_reuse(profile->clusters);
			if(empty_idx != -1) {
				PySet_Add(
					profile->clusters, PyLong_FromLong((long) empty_idx));
				(profile->assignments)[pt] = empty_idx;
			}
			// No empty cluster -> add new
			else {
				PySet_Add(
					profile->clusters, PyLong_FromLong((long) n_clusters));
				(profile->assignments)[pt] = n_clusters;
			}
		}
		// Assign to existing cluster
		else { (profile->assignments)[pt] = assign; }
	}

	// Clean up
	free(weights);
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

    struct crp_gibbs_profile_t profile = {
    	.data = data,
    	.assignments = assignments,
    	.size = PyArray_DIM(data, 0),
    	.dim = PyArray_DIM(data, 1),
    	.clusters = clusters,
    	.callback_var = callback_var,
    	.l_cond = l_cond,
    	.l_uncond = l_uncond,
    	.alpha = alpha,
    	.r = r
    }

    // Run gibbs sampler
    crp_gibbs_iter(&profile);

    // Return None
    Py_RETURN_NONE;
}
