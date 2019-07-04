/**
 *
 */

#include <gibbs.h>
#include <dpm.h>
#include <mfm.h>


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


static PyMethodDef ModuleMethods[] = {
	{
		"gibbs_iter",
		(PyCFunction) gibbs_iter_py,
		METH_VARARGS,
		DOCSTRING_GIBBS_ITER
	},
	{NULL, NULL, 0, NULL}
}

static struct PyModuleDef ModuleDef = {
	PyModuleDef_HEAD_INIT,
	"bayesian_clustering_c",
	"C accelerator functions for bayesian clustering algorithms",
	-1,
	ModuleMethods
};

PyMODINIT_FUNC PyInit_bayesian_clustering_c()
{
	import_array();

	PyObject *mod = PyModule_Create(&ModuleDef);
	static void *API[]
	if(mod == NULL) { return NULL; }

	// Capsules
	PyModule_AddObject(
		mod, "MODEL_DPM", PyCapsule_New(
			&dpm_methods, "bayesian_clustering_c.ModelMethods", NULL));
	PyModule_AddObject(
		mod, "MODEL_MFM", PyCapsule_New(
			&mfm_methods, "bayesian_clustering_c.ModelMethods", NULL));

	return mod;
}

