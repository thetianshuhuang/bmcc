#ifndef TYPE_CHECK_H
#define TYPE_CHECK_H

#include <stdbool.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


bool type_check(PyArrayObject *data_py, PyArrayObject *assignments_py);
bool type_check_square(PyArrayObject *data_py, int dim);

#endif
