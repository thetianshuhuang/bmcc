#ifndef MIXTURE_CREATE_H
#define MIXTURE_CREATE_H

#include "../include/mixture/mixture.h"


// Create mixture model struct
struct mixture_model_t *create_mixture(
    ComponentMethods *comp_methods,
    ModelMethods *model_methods,
    PyObject *params,
    uint32_t size, uint32_t dim, int type);

// Destroy mixture model struct
void destroy_mixture(PyObject *model_py);

#endif
