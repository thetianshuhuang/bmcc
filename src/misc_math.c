
#include <math.h>

/**
 * Helper function for log multivariate gamma
 * @param x : x value
 * @param p : multivariate gamma dimension
 * @return log(Gamma_p(x))
 */
const double 
double log_mv_gamma(int p, double x)
{
    double res = log(M_PI) * p * (p - 1) / 4;
    for(int j = 0; j < p; j++) {
        res += lgamma(x + (1 - j) / 2);
    }
    return res;
}
