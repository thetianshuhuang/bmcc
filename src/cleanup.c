/** 
 * Cleanup Gibbs Sampler
 */

#include <Python.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../include/mixture.h"
#include "../include/mixture_shortcuts.h"
#include "../include/misc_math.h"
#include "../include/type_check.h"
#include "../include/base_iter.h"

/**
 * Execute cleanup iteration
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 * @param annealing annealing factor (ignored)
 * @return true if returned without error
 */
bool cleanup_iter(
    double *data,
    uint16_t *assignments,
    struct mixture_model_t *model,
    double annealing)
{
	for(int idx = 0; idx < model->size; idx++) {

		// Remove point
		double *point = &data[idx * model->dim];
		model->comp_methods->remove(
			model->clusters[assignments[idx]], model->comp_params, point);

		// Remove empty components
		remove_empty(model, assignments);

		// Find maximum likelihood assignment
		int idx_best = 0;
		double loglik_best = -INFINITY;
		for(int i = 0; i < model->num_clusters; i++) {
			double loglik_new = marginal_loglik(
				model, model->clusters[i], point);
			if(loglik_new > loglik_best) {
				loglik_best = loglik_new;
				idx_best = i;
			}
		}

		// Assign
		add_point(model, model->clusters[idx_best], point);
		assignments[idx] = idx_best;
	}

	// No allocation -> should be no system errors
	return true;
}


/* 
 * Run cleanup iteration. See docstring (sourced from cleanup.h) for details on
 * Python calling.
 */
PyObject *cleanup_iter_py(PyObject *self, PyObject *args, PyObject *kwargs)
{
	return base_iter(self, args, kwargs, &supports_gibbs, &cleanup_iter);
}
