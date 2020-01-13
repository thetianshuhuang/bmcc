/**
 * Miscallaneous Math Functions
 */

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "../include/cholesky.h"
#include "../include/misc_math.h"

// No longer using GSL; simply replicate PI definition
#define M_PI 3.14159265358979323846264338328

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
 * Helper function for logbeta
 */
double log_beta(int a, int b)
{
    return lgamma(a) + lgamma(b) - lgamma(a + b);
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
 * Random double between 0 and 1 with 45 bits of entropy
 */
double rand_double() {
    return ((double) rand_45_bit()) / ((double) RAND_MAX_45);
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
    double x = rand_double();

    // Sample
    double acc = 0;
    for(int i = 0; i < length; i++) {
        acc += weights[i];
        if(x <= acc) { return i; }
    }
    // Catch all for rounding errors
    return length - 1;
}


/**
 * Get log determinant of centered array.
 *
 * "chol" stores
 *      Cholesky(S + Sum_i[x_i x_i^T])
 * but we need
 *      Cholesky(S + Sum_i[(x_i - mu)(x_i - mu)^T])
 * We center using 
 *      Sum_i[(x_i - mu)(x_i - mu)^T] = Sum_i[x_i x_i^T] - N mu mu^T
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


double rand_norm() {
    double u_1 = rand_double();
    double u_2 = rand_double();

    return sqrt(-2 * log(u_1)) * cos(2 * M_PI * u_2);
}


/**
 * Marsaglia & Tsang
 */
double rand_gamma(double alpha)
{
    // Must have alpha > 1
    if(alpha < 1) { return 0; }

    double d = alpha - (1.0 / 3.0);
    double c = 1.0 / 3.0 / sqrt(d);
    
    while(true) {
        double z = 0;
        double v = 0;
        do {
            z = rand_norm();
            v = 1 + c * z;
        } while(v <= 0);

        v = pow(v, 3);
        double u = rand_double();

        bool accept = (
            (log(u) < 0.5 * z * z + d - d * v + d * log(v)) ||
            (u < 1.0 - 0.0331 * pow(z, 4)));
        if(accept) { return d * v;};
    }
}


double rand_beta(double alpha, double beta) {
    double x = rand_gamma(alpha);
    double y = rand_gamma(beta);
    return x / (x + y);
}
