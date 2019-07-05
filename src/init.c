

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "component.h"
#include "model.h"
#include "type_check.h"


#define DOCSTRING_INIT_MODEL_CAPSULES \
    "Initialize model capsules\n"
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "data_py : np.array\n" \
    "    Data array\n" \
    "assignments_py : np.array\n" \
    "    Assignment array\n" \
    "comp_methods : capsule\n" \
    "    Capsule containing ComponentMethods struct\n" \
    "model_methods : capsule\n" \
    "    Capsule containing ModelMethods struct\n" \
    "params : dict\n" \
    "    Dictionary containing hyperparameters.\n" \
    "\n" \
    "Returns\n" \
    "-------\n" \
    "capsule\n" \
    "    Capsule containing the created struct mixture_model_t"

PyObject *init_model_capsules_py(PyObject *args)
{
    // Get args
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *comp_methods_py;
    PyObject *model_methods_py;
    PyObject *params;
    bool success = PyArg_ParseTuple(
        args, "OOO!",
        &PyArray_Type, &data_py,
        &PyArray_Type, &assignments_py,
        &comp_methods_py, &comp_params_py,
        &params);
    if(!success) { return NULL; }
    if(!type_check(data_py, assignments_py)) { return NULL; }

    // Unpack capsules
    ComponentMethods *comp_methods = (
        (ComponentMethods *) PyCapsule_GetPointer(
            comp_methods_py, COMPONENT_METHODS_API));
    ModelMethods *model_methdos = (
        (ModelMethods *) PyCapsule_GetPointer(
            model_methods_py, MODEL_METHODS_API));

    uint16_t *assignments = PyArray_DATA(assignments_py);
    int size = PyArray_DIM(data_py, 0);
    int dim = PyArray_DIM(data_py, 1);

    // Allocate components_t
    struct mixture_model_t *mixture = create_components();

    // Bind methods, params, dim
    mixture->comp_methods = comp_methods;
    mixture->model_methods = model_methods;
    mixture->comp_params = comp_methods->create(params);
    mixture->model_params = model_methods->create(params);
    mixture->size = size;
    mixture->dim = dim;

    // Get number of components
    int max = 0;
    for(int i = 0; i < size; i++) {
        if(assignments[i] > max) { max = assignments[i]; }
    }

    // Allocate enough components
    for(int i = 0; i <= max; i++) { add_component(mixture, comp_methods); }

    // Add points
    for(int i = 0; i < size; i++) {
        comp_methods->add(
            mixture->clusters[assignments[i]], &(data[idx * dim]));
    }

    return PyCapsule_New(
        components, MIXTURE_MODEL_API, &destroy_components);
}

