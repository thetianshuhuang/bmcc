/**
 * Cholesky Decomposition Methods
 */

#ifndef CHOLESKY_H
#define CHOLESKY_H

// Log determinant of a matrix using it's cholesky decomposition
double cholesky_logdet(double *mat, int dim);

// Cholesky update
void cholesky_update(double *mat, double *point, double scale, int dim);

// Cholesky downdate
void cholesky_downdate(double *mat, double *point, double scale, int dim);

#endif
