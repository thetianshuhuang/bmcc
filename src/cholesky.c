/**
 * Cholesky Decomposition Methods
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * Helper function for log(det|LL^T|) using the cholesy decomposition
 * @param mat : input L matrix
 * @param dim : size
 * @return log(det|LL^T|) = log(prod(L_ii)^2) = 2sum(log(L_ii))
 */
double cholesky_logdet(double *mat, int dim)
{
    // Get determinant: log(det|LL^T|) = log(prod(L_ii)^2) = 2sum(log(L_ii))
    double logdet = 0;
    for(int i = 0; i < dim; i++) {
        logdet += log(mat[dim * i + i]);
    }
    return 2 * logdet;
}


/**
 * Update Cholesky Decomposition
 * @param mat : matrix L to update
 * @param pt : point x to add; LL^T -> LL^T + xx^T
 * @param dim : number of dimensions
 */
void cholesky_update(double *mat, double *pt, double scale, int dim)
{
	double *pt_cpy = (double *) malloc(sizeof(double) * dim);

	for(int i = 0; i < dim; i++) { pt_cpy[i] = scale * pt[i]; }

	for(int i = 0; i < dim; i++) {
		double r = sqrt(
			mat[i * dim + i] * mat[i * dim + i] + pt_cpy[i] * pt_cpy[i]);
		double c = r / mat[i * dim + i];
		double s = pt_cpy[i] / mat[i * dim + i];
		mat[i * dim + i] = r;
		for(int j = i + 1; j < dim; j++) {
			mat[i * dim + j] = (mat[i * dim + j] + s * pt_cpy[j]) / c;
			pt_cpy[j] = c * pt_cpy[j] - s * mat[i * dim + j];
		}
	}

	free(pt_cpy);
}


/**
 * Downdate Cholesky Decomposition
 * @param mat : matrix L to downdate
 * @param pt : point x to add; LL^T -> LL^T + xx^T
 * @param dim : number of dimensions
 */
void cholesky_downdate(double *mat, double *pt, double scale, int dim)
{
	double *pt_cpy = (double *) malloc(sizeof(double) * dim);
	for(int i = 0; i < dim; i++) { pt_cpy[i] = scale * pt[i]; }

	for(int i = 0; i < dim; i++) {
		double r = sqrt(
			mat[i * dim + i] * mat[i * dim + i] - pt_cpy[i] * pt_cpy[i]);
		double c = r / mat[i * dim + i];
		double s = pt_cpy[i] / mat[i * dim + i];
		mat[i * dim + i] = r;
		for(int j = i + 1; j < dim; j++) {
			mat[i * dim + j] = (mat[i * dim + j] - s * pt_cpy[j]) / c;
			pt_cpy[j] = c * pt_cpy[j] - s * mat[i * dim + j];
		}
	}

	free(pt_cpy);
}