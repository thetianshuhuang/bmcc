

#include <Python.h>
#include <stdint.h>

#define BASE_VEC_SIZE = 32


void *get_component(PyObject *clusters, int idx)
{
	return PyCapsule_GetPointer(PyList_GetItem(clusters, idx));
}


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

		int nc = PyList_Size(clusters);
		float *point = &data[idx * comp_params->dim];

		// Remove from currently assigned cluster
		comp_methods->remove(get_component(clusters, assignments[idx]));

		// Handle vector resizing
		if(nc + 1 > weight_vec_size) {
			free(weights);
			vec_size *= 2;
			double *weights = (double *) malloc(sizeof(double) * vec_size);
		}

		// Get assignment weights
		for(int i = 0; i < nc; i++) {
			void *c_i = get_component(clusters, i);
			weights[i] = (
				comp_methods->loglik_ratio(c_i, params, point) +
				model_methods->log_coef(params, get_size(c_i)));
		}

		// Get new cluster weight
		weights[nc] = (
			model_methods->log_coef_new(params) *
			comp_methods->loglik_new(params, point));

		// Sample new
		int new = sample_weighted(vec_size, nc + 1);
		// New cluster?
		if(new == nc) { add_component(clusters, comp_methods, params); }
		// Update component
		comp_methods->add(get_component(clusters, new), params, point);

		// Todo: remove empty clusters
	}

}


PyObject *gibbs_iter_py(PyObject *args) {

	// Get args
	PyArrayObject *data_py;
	PyArrayObject *assignments_py;
	PyObject *clusters_py;
	PyObject *params_py;
	bool success = PyArg_ParseTuple(
		args, "O!O!OO",
		&PyArray_Type, &data, &PyArray_Type, &assignments,
		&clusters_py, &params_py)
	if(!success) { return NULL; }

	// Unpack cluster prior capsule
	ComponentMethods *comp_methods = (
		(ComponentMethods *) PyCapsule_GetPointer(
			PyDict_GetItemString(params_py, "component")));
	// Unpack model capsule
	ModelMethods *model_methods = (
		(ModelMethods *) PyCapsule_GetPointer(
			PyDict_GetItemString(params_py, "model")));

	// GIL free zone ----------------------------------------------------------
	Py_BEGIN_ALLOWED_THEADS

	gibbs_iter(
		(float *) PyArray_DATA(data_py),
		(uint16_t *) PyArray_DATA(assignments_py),
		(struct components_t *) PyCapsule_GetPointer(clusters_py),
		comp_methods,
		model_methods,
		comp_methods->parameters(params_py),
		model_methods->parameters(params_py),
		PyArray_DIM(data, 0));

	Py_END_ALLOW_THREADS
	// ------------------------------------------------------------------------

	Py_RETURN_NONE
}
