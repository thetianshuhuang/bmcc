
#include <set>
#include <cmath>

#include <python.h>
#include <numpy/arrayobject.h>


/**
 *
 */
bool gibbs_crp_dim_check(
        PyArrayObject *data,
        PyArrayObject *assignments,
        int min_size)
{

    // Check array bounds
    if(PyArray_NDIM(data_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError,
            "Data must have 2 dimensions.");
        return false;
    }
    if(PyArray_NDIM(assignments_py) != 2) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignment history must have 2 dimensions.");
        return false;
    }

    // Make sure history array is large enough
    if(PyArray_DIM(assignments, 1) != PyArray_DIM(data, 0)) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignment history must have the same 0 axis dimension as data.");
        return false;
    }
    if(PyArray_DIM(assignments, 0) < min_size) {
        PyErr_SetString(
            PyExc_TypeError,
            "Assignment history is not large enough to store thinned values");
        return false;
    }

    // No error
    return true
}


/**
 * Turn assignment array into cluster vector
 * @param assignments : list of assignments
 * @param size : number of data points (len(assignments))
 */
std::vector<std::set<int>> init_clusters(int *assignments, int size) {

    std::vector<std::set<int>> clusters = new std::vector<std::set<int>>;

    for(int i = 0; i < size; i++) {

        // Not enough clusters -> add until index is reached
        if(assignments[i] >= clusters.size()) {
            for(int j = clusters.size(); j <= assignments[i]; j++) {
                std::set<int> new_cluster = new std::set<int>;
                clusters.push_back(new_cluster);
            }
        }

        // Add to cluster
        clusters[assignments[i]].add(i);
    }
    return clusters;
}


/**
 * Sample proportionally from a weight vector
 * @param weights : list of weights
 * @param size : length of weight vector
 * @return index sampled from the len(weights) proportional to weight
 */
int sample_proportional(float *weights, int *size)
{
    // Sample from [0, sum_weights)
    // Saves k flops compared to normalizing first
    float sum_weights = 0;
    for(int i = 0; i < size; i++) { sum_weights += weights[i]; }
    float unif = sum_weights * (float) ((double) rand() / ((double) RAND_MAX));

    // Check cumulative sum
    float acc;
    for(int i = 0; i < size; i++) {
        acc += weights[size];
        if(unif < acc) { return i; }
    }

    // Catch-all in case of rounding error (just in case)
    return size - 1;
}
