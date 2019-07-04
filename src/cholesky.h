/**
 *
 */

#ifndef CHOLESKY_H
#define CHOLESKY_H

// Log determinant of a matrix using it's cholesky decomposition
double cholesky_logdet(float *mat, int dim);

// Cholesky update
cholesky_update(float *mat, float *point, int dim);

// Cholesky downdate
cholesky_downdate(float *mat, float *point, int dim);

#endif
