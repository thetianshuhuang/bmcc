#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Python Base
#include <Python.h>

// Numpy arrays
#include <numpy/arrayobject.h>


static PyObject *test_print(PyObject *self, PyObject *args)
{
	const char *command;
	if(PyArg_ParseTuple(args, "s", &command)) {
		printf("%s\n", command);
	}
	Py_RETURN_NONE;
}


static PyObject *numpy_print(PyObject *self, PyObject *args)
{
	PyArrayObject *array;

	if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) { return NULL; }

	float *data = (float *) PyArray_DATA(array);
	int dim0 = PyArray_DIM(array, 0);

	for(int i = 0; i < dim0; i++) {
		printf("%f\n", data[i]);
	}
	Py_RETURN_NONE;
}


static PyMethodDef ModuleMethods[] = {
	{"test_print", (PyCFunction) test_print, METH_VARARGS, "print string"},
	{"test_np_print", (PyCFunction) numpy_print, METH_VARARGS, "print np array"},
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef ModuleDef = {
	PyModuleDef_HEAD_INIT,
	"test",
	"Docstring...",
	-1,
	ModuleMethods
};


PyMODINIT_FUNC PyInit_test() {
	import_array()
	return PyModule_Create(&ModuleDef);
}
