/**
 * Miscallaneous Math Functions
 */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "../include/cholesky.h"
#include "../include/misc_math.h"

/**
 * Helper function for log multivariate gamma
 * @param x : x value
 * @param p : multivariate gamma dimension
 * @return log(Gamma_p(x))
 */
double log_mv_gamma(int p, double x)
{
    double res = log(M_PI) * p * (p - 1) / 4;
    for(int j = 1; j <= p; j++) {
        res += lgamma(x + (1 - j) / 2);
    }
    return res;
}


/**
 * Sample 45-bit integer, since RAND_MASK only has 15 bits of precision
 * @return Sampled 45-bit integer
 */
uint64_t rand_45_bit()
{
	uint64_t res = 0;
	for(int i = 0; i < 3; i++) {
		res = res << 15;
		res |= rand() & RAND_MASK;
	}
	return res;
}


/**
 * Sample a discrete distribution from a vector of log-weights
 * (not neccessarily likelihoods!). Extreme caution is taken to deal with
 * potential very large or very small values. Since C rand() only has 15 bits
 * of resolution, assignments with very small relative likelihoods may not
 * be accurately represented.
 */
int sample_log_weighted(double *weights, int length)
{
	// Normalize weights and convert to actual likelihood
	// Largest value is set to 0 (1 once converted back)
	double max = weights[0];
	for(int i = 0; i < length; i++) {
		if(weights[i] > max) { max = weights[i]; }
	}
	for(int i = 0; i < length; i++) {
		if(isnan(weights[i])) { weights[i] = 0; }
		else { weights[i] = exp(weights[i] - max); }
	}

	// Normalization constant
	double total = 0;
	for(int i = 0; i < length; i++) { total += weights[i]; }
	for(int i = 0; i < length; i++) { weights[i] /= total; }

	// Generate random number in [0, total]
	double x = ((double) rand_45_bit()) / ((double) RAND_MAX_45);

	// Sample
	double acc = 0;
	for(int i = 0; i < length; i++) {
		acc += weights[i];
		if(x <= acc) { return i; }
	}
	// Catch all for rounding errors
	return length - 1;
}


/*
 * Get log determinant of centered array.
 *
 * "chol" stores
 *		Cholesky(S + Sum_i[x_i x_i^T])
 * but we need
 * 		Cholesky(S + Sum_i[(x_i - mu)(x_i - mu)^T])
 * We center using 
 *		Sum_i[(x_i - mu)(x_i - mu)^T] = Sum_i[x_i x_i^T] - N mu mu^T
 *
 * @param chol Cholesky decomposition of sum matrix S + XX^T
 * @param total Total vector Sum_i[x_i]
 * @param d number of dimensions
 * @param n number of samples
 */
double centered_logdet(double *chol, double *total, int d, int n)
{
	// Make copy
	double *copy = (double *) malloc(sizeof(double) * d * d);
	for(int i = 0; i < d * d; i++) { copy[i] = chol[i]; }

	// Cholesky downdate; (1/sqrt(N) mu)(1/sqrt(N) mu)^T = N mu mu^T
	cholesky_downdate(copy, total, 1 / sqrt(n), d);
	// Get log determinant
	double res = cholesky_logdet(copy, d);

	// Clean up
	free(copy);
	return res;
}
