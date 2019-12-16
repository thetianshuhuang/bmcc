/** 
 * Stochastic Block Model
 */

/*
struct sbm_t {
	// Number of points
	uint32_t n;
	// Pointer to config
	struct sbm_params_t *params;
}

struct sbm_params_t {
	// Q array
	double *Q;
	// Number of points
	uint32_t n;
	// Number of clusters
	uint32_t k;
}



void *sbm_create(void *params)
{
	struct sbm_t *component = (struct sbm_t *) malloc(sizeof(struct sbm_t));
	component->n = 0;
	component->params = (struct sbm_params_t *) params;
	return component;
}


void *sbm_destroy(void *component)
{
	struct sbm_params_t *params = ((struct sbm_t *) component)->params;

	params->k -= 1;
	// TODO: update params->Q
}


void *sbm_params_create(PyObject *dict)
{
	struct sbm_params_t *params = (
		(struct sbm_params_t *) malloc(sizeof(struct sbm_params_t)));

	PyArrayObject *Q_py = (PyArrayObject *) PyDict_GetItemString(dict, "Q");
	PyObject *n_py = (PyObject *) PyDict_GetItemString(dict, "n");
	if((Q_py == NULL) || (n_py == NULL) || (!PyLong_Check(n_py))) {
		PyErr_SetString(
			PyExc_KeyError,
			"SBM requires Q array (SPM likelihood array) and n (number of "
			"points).");
		return NULL;
	}
	params->Q = PyArray_DATA(Q_py);
	params->n = (int) PyLong_AsLong(n_py);
}


void sbm_params_destroy(void *params)
{
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params_tc;
	free(params_tc->Q);
}


void sbm_params_update(void *params, PyObject *dict)
{
	struct sbm_params_t *params = (
		(struct sbm_para	PyArrayObject *Q_py = (PyArrayObject *) PyDict_GetItemString(dict, "Q");
ms_t *) malloc(sizeof(struct sbm_params_t)));

	PyArrayObject *Q_py = (PyArrayObject *) PyDict_GetItemString(dict, "Q");
	PyObject *k_py = (PyObject *) PyDict_GetItemString(dict, "k");
	// Update Q
}


void *sbm_get_size(void *component)
{
	return ((struct sbm_t *) component)->n;
}


void *sbm_add(void *component, void *params, double *point)
{
	((struct sbm_t *) component)->n += 1;
}


void *sbm_remove(void *component, void *params, double *point)
{
	((struct sbm_t *) component)->n -= 1;
}


void *sbm_likelihood(void *params, uint16_t *assignments)
{
	struct sbm_params_t *params_tc = (struct sbm_params_t *) params;

	double res = 0;
	for(int i = 0; i < params_tc->n; i++) {
		for(int j = 0; j < i; j++) {
			res += (
				params_tc->A[i * n + j] *
				log(Q[assignments[i] * params_tc->k + assignments[j]])
			);
		}
	}
	return res;
}


void *sbm_loglik_new(void *params, double *point)
{

}


void *sbm_loglik_ratio(void *component, void *params, double *point, int idx)
{
	double acc = 0;
	for(int i = 0; i < params->n; i++) {
		if(i != idx) {
			int qval = (
				params->Q[component->idx * params->k + params->assignments]);
			acc += log(qval) * point[i] + log(1 - qval) * (1 - point[i]);
		}
	}
	return acc;
}


void *sbm_split_merge(void *params, void *merged, void *c1, void *c2)
{

}


ComponentMethods SBM = {
	&sbm_params_create,
	&sbm_params_destroy,
	&sbm_params_update,
	&sbm_create,
	&sbm_destroy,
	&sbm_get_size,
	&sbm_add,
	&sbm_remove,
	&sbm_loglik_ratio,
	&sbm_loglik_new,
	&sbm_split_merge
}
*/
