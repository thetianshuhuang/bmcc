/**
 * Miscallaneous Math Functions
 */

#ifndef MISC_MATH_H
#define MISC_MATH_H

#define RAND_MASK 0x7FFF
#define RAND_MAX_45 0x200000000000

// log(Gamma_p(x)) -- log of multivariate gamma
double log_mv_gamma(int p, double x);

// log(Beta(a, b))
double log_beta(int a, int b);

// Get 45-bit random iteger
uint64_t rand_45_bit();

// Sample from weighted vector
int sample_log_weighted(double *weights, int length);

// Centered log determinant
double centered_logdet(double *chol, double *total, int d, int n);

// Unif[0, 1] random variable as double
double rand_double();

// Normal (0, 1) random variable
double rand_norm();

// Gamma (alpha, 1) random variable
double rand_gamma(double alpha);

// Beta random variable
double rand_beta(double alpha, double beta);

#endif
