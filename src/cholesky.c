
/**
 * Helper function for log(det|LL^T|) using the cholesy decomposition
 * @param mat : input L matrix
 * @param dim : size
 * @return log(det|LL^T|) = log(prod(L_ii)^2) = 2sum(log(L_ii))
 */
double cholesky_logdet(float *mat, int dim)
{
    // Get determinant: log(det|LL^T|) = log(prod(L_ii)^2) = 2sum(log(L_ii))
    double logdet = 0;
    for(int i = 0; i < dim; i++) {
        logdet += log(mat[dim * i + i]);
    }
    return 2 * logdet;
}
