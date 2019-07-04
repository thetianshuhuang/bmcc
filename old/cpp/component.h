/**
 *
 *
 */

#ifndef COMPONENT_H
#define COMPONENT_H


#include <armadillo>


/* T: Data type (half, float, or double) */
template <T>
class NormalWishartComponent {
    private:
        /* Pointer to source data array; each data point is a row */
        arma::mat<T> *data;
        /* Accumulator -- sum of all data members; row vector */
        arma::row<T> total;
        /* Cholesky decomposition of (S + XX^T) for wishart scale matrix S */
        arma::mat<T> chol_decomp;
        /* Wishart Degrees of freedom */
        int df;

    public:
        /* Constructor */
        NormalWishartComponent(arma::mat<T> *data, arma::mat<T> *scale, int df);
        /* Add point */
        void add(int pt);
        /* Remove point */
        void remove(int pt);
        /* Likelihood */
        T marginal_likelihood();
        /* Likelihood ratio */
        T marginal_likelihood_ratio(int pt);
        /* Get mean */
        arma::row<T> mean();

        /* Set of member points */
        std::unordered_set<int> points;
}
