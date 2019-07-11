/**
 * Miscallaneous Math Functions
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * Helper function for log multivariate gamma
 * @param x : x value
 * @param p : multivariate gamma dimension
 * @return log(Gamma_p(x))
 */
double log_mv_gamma(int p, double x)
{
    double res = log(M_PI) * p * (p - 1) / 4;
    for(int j = 0; j < p; j++) {
        res += lgamma(x + (1 - j) / 2);
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
	double x = ((double) rand()) / ((double) RAND_MAX);

	// Sample
	double acc = 0;
	for(int i = 0; i < length; i++) {
		acc += weights[i];
		if(x <= acc) { return i; }
	}
	// Catch all for rounding errors
	return length - 1;
}

