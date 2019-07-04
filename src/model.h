
#ifndef MODEL_H
#define MODEL_H

#include <Python.h>


typedef struct model_methods_t {
	void* (*create)(PyObject *dict);
	void (*destroy)(void *parameters);
	double (*log_coef)(void *params, int size);
	double (*log_coef_new)(void *params);


} ModelMethods;


#endif
