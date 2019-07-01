
#ifndef UTIL_H
#define UTIL_H

#include <python.h>
#include <numpy/arrayobject.h>

#include <set.h>

bool gibbs_crp_dim_check(
        PyArrayObject *data,
        PyArrayObject *assignments,
        int min_size);

std::vector<std::set<int>> init_clusters(int *assignments, int size);

int sample_proportional(float *weights, int *size);

#endif
