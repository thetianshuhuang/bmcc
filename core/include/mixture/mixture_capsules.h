/**
 * Capsule Manipulation Functions
 */

#ifndef MIXTURE_CAPSULES_H
#define MIXTURE_CAPSULES_H


#define DOCSTRING_INIT_MODEL_CAPSULES \
    "Initialize model capsules\n" \
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

PyObject *init_model_capsules_py(PyObject *self, PyObject *args);


#define DOCSTRING_UPDATE_MIXTURE \
    "Update mixture model hyperparameters\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "mixture : capsule\n" \
    "    Capsule containing mixture struct to update hyperparameters for\n" \
    "update : dict\n" \
    "    Dictionary to update values with"

PyObject *update_mixture_py(PyObject *self, PyObject *args);


#define DOCSTRING_UPDATE_COMPONENTS \
    "Update component hyperparameters\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "mixture : capsule\n" \
    "    Capsule containing mixture struct to update hyperparameters for\n" \
    "update : dict\n" \
    "    Dictionary to update values with"

PyObject *update_components_py(PyObject *self, PyObject *args);


#define DOCSTRING_INSPECT_MIXTURE \
    "Run the inspect method of a Mixture Model\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "mixture : capsule\n" \
    "    Capsule containing mixture struct to inspect\n" \
    "\n" \
    "Returns\n" \
    "-------\n" \
    "Some Object\n" \
    "    Value returned by the inspect method, if available\n" \
    "\n" \
    "Raises\n" \
    "------\n" \
    "TypeError\n" \
    "    Input is not a capsule of the correct API, or the mixture model " \
    "specified\n" \
    "    does not implement inspect()"

PyObject *inspect_mixture_py(PyObject *self, PyObject *args);


#define DOCSTRING_COUNT_CLUSTERS \
    "Get the number of clusters currently in the chain\n" \
    "\n" \
    "Fetches the cluster number directly from the mixture\n" \
    "\n" \
    "Parameters\n" \
    "----------\n" \
    "mixture : capsule\n" \
    "    Capsule containing mixture struct to inspect\n" \
    "\n" \
    "Returns\n" \
    "-------\n" \
    "int\n" \
    "    Number of clusters"

PyObject *count_clusters_py(PyObject *self, PyObject *args);


#endif
