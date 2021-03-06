/**
 * Wishart-distributed Multivariate Gaussian Components
 */

#ifndef NORMAL_WISHART_H
#define NORMAL_WISHART_H

#include <Python.h>
#include "../include/mixture/mixture.h"


// Normal Wishart struct
struct nw_component_t {
    // Total vector; dimensions [dim]
    double *total;
    // Cholesky decomposition of S + XX^T; dimensions [dim, dim]
    double *chol_decomp;
};

// Hyperparameters struct
struct nw_params_t {
    // Wishart Degrees of freedom
    double df;
    // Dimensions
    uint32_t dim;
    // Cholesky decomposition of scale matrix
    double *s_chol;
};

// Component methods; only the struct is exposed!
ComponentMethods NORMAL_WISHART;

#endif
