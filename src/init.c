

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "component.h"
#include "model.h"
#include "type_check.h"


#define DOCSTRING_MODEL_INIT_CAPSULES \
    "Initialize parameter capsules\n"
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "params : dict\n" \
    "    Dictionary containing hyperparameters.\n" \
    "comp_methods : capsule\n" \
    "    Capsule containing ComponentMethods struct\n" \
    "model_methods : capsule\n" \
    "    Capsule containing ModelMethods struct"

PyObject *model_init_capsules(PyObject *args)
{
    // Unpack args
    PyObject *dict;
    PyObject *comp_methods_py;
    PyObject *model_methods_py;
    bool success = PyArg_ParseTuple(
        args, "OOO", &dict, &comp_methods_py, &model_methods_py);
    if(!success) { return NULL; }

    // Unpack capsules
    ComponentMethods *comp_methods = (
        (ComponentMethods *) PyCapsule_GetPointer(
            component_methods_py, COMPONENT_METHODS_API));
    ModelMethods *model_methods = (
        (ModelMethods *) PyCapsule_GetPointer(
            model_methods_py, MODEL_METHODS_API));

    // Create parameters
    void *model_params = model_methods->create(dict);
    void *comp_params = comp_methods->params_create(dict);

    // Create return dict
    PyObject *ret = PyDict_New();
    PyDict_SetItem(ret, "comp_methods", comp_methods_py);
    PyDict_SetItem(ret, "model_methods", model_methods_py);
    PyDict_SetItem(ret, "comp_params", PyCapsule_New(
        comp_params, MODEL_PARAMS_API, model_methods->destroy));
    PyDict_SetItem(ret, "model_params", PyCapsule_New(
        model_params, COMPONENT_PARAMS_API, comp_methods->params_destroy));

    return ret;
}


PyObject *init_components(PyObject *args)
{
    // Get args
    PyArrayObject *data_py;
    PyArrayObject *assignments_py;
    PyObject *comp_methods_py;
    PyObject *comp_params_py;
    bool success = PyArg_ParseTuple(
        args, "OOO!",
        &PyArray_Type, &data_py,
        &PyArray_Type, &assignments_py,
        &comp_methods_py, &comp_params_py);
    if(!success) { return NULL; }
    if(!type_check(data_py, assignments_py)) { return NULL; }

    // Unpack capsules
    ComponentMethods *comp_methods = (
        (ComponentMethods *) PyCapsule_GetPointer(
            comp_methods_py, COMPONENT_METHODS_API));
    void *comp_params = PyCapsule_GetPointer(
        comp_params_py, COMPONENT_PARAMS_API);

    uint16_t *assignments = PyArray_DATA(assignments_py);
    int size = PyArray_DIM(data_py, 0);
    int dim = PyArray_DIM(data_py, 1);

    // Allocate components_t
    struct mixture_model_t *components = create_components();

    // Get number of components
    int max = 0;
    for(int i = 0; i < size; i++) {
        if(assignments[i] > max) { max = assignments[i]; }
    }

    // Allocate enough components
    for(int i = 0; i <= max; i++) {
        add_component(components, comp_methods, comp_params);
    }

    // Add points
    for(int i = 0; i < size; i++) {
        comp_methods->add(
            components->values[assignments[i]],
            &(data[idx * dim]));
    }

    // Bind methods, params, dim
    components->methods = comp_methods;
    components->params = comp_params;

    PyCapsule_New(components, COMPONENT_DATA_API, )

    return 

}

