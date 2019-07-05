
#ifndef MODEL_H
#define MODEL_H

#include <Python.h>

#define MODEL_METHODS_API "bayesian_clustering_c.ModelMethods"
#define MODEL_PARAMS_API "bayesian_clustering_c.ModelParams"

typedef struct model_methods_t {
	void* (*create)(PyObject *dict);
	void (*destroy)(void *parameters);
	double (*log_coef)(void *params, int size);
	double (*log_coef_new)(void *params);
} ModelMethods;


#endif
