

struct crp_gibbs_profile_t {
	// Data
	PyArrayObject *data;
	PyArrayObject *assignments;
	int size;
	int dim;

	// Assignments
	PyObject *clusters;

	// Callbacks
	bool callback_var;
	PyObject *l_cond;
	PyObject *l_uncond;

	// Hyperparameters
	float alpha;
	float r;
}
