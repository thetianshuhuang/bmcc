/**
 *
 *
 */

#ifndef NORMAL_WISHART_H
#define NORMAL_WISHART_H

#include <Python.h>

// Normal Wishart struct (goes inside capsules)
struct nw_component_t {
    // Total vector; dimensions [dim]
    float *total;
    // Number of points
    uint32_t n;
    // Cholesky decomposition of S + XX^T; dimensions [dim, dim]
    float *chol_decomp;
}

// Hyperparameters struct
struct nw_params_t {
    // Wishart Degrees of freedom
    float df;
    // Dimensions
    uint32_t dim;
    // Cholesky decomposition of scale matrix
    float *s_chol;
}

// Component methods; only the struct is exposed!
extern struct component_methods_t normal_wishart;

#endif
