/**
 *
 */

#include <component.h>
#include <cholesky.h>


/**
 * Constructor
 *
 * @param data : Source data matrix; each row is a data point; the row indices
 *      are used by this->points.
 * @param scale_chol : Cholesky decomposition of scale matrix of data. Should
 *      be calculated once at the beginning and passed around.
 * @param df : Degrees of freedom for Wishart distribution
 */
NormalWishartComponent::NormalWishartComponent(
    arma::mat<T> *data,
    arma::mat<T> *scale,
    int df)
{
    // Set data pointer
    this->data = data;

    // Initializations
    points = std::unordered_set<int>();
    this->df = df;
    // total = [0 ... 0] (row vector)
    total = arma::Row<T>(data.n_cols, 0);
    // chol_decomp (initial) = chol(S + XX^T) = chol(S)
    chol_decomp = arma::Mat<T>(S);
}


/**
 * Add point
 *
 * @param point : index of point to add
 */
NormalWishartComponent::add(int pt) {
    // Add to set
    points.insert(pt);

    // Update sum accumulator
    total += data.row(pt);
    // Update Cholesky decomposition
    cholesky_update(chol_decomp, data.row(pt)); // todo
}


/**
 * Remove point
 *
 * @param point : index of point to remove
 */
NormalWishartComponent::remove(int pt) {
    // Remove from set
    points.erase(pt);

    // Update sum accumulator
    total -= data.row(pt);
    // Update Cholesky decomposition
    cholesky_downdate(chol_decomp, data.row(pt)); // todo
}


/**
 * Get Marginal Likelihood m(x_j)
 * Using the conjugate distribution of the Inverse Wishart,
 *
 * m(x_c) =
 *                |S|^{df/2} Gamma_D((df + N) / 2)
 *      ---------------------------------------------------
 *      pi^{ND/2} |S + XX^T|^{(df + N) / 2} Gamma_D(df / 2)
 *
 * @return Marginal likelihood m(x_c)
 */
NormalWishartComponent::marginal_likelihood() {
    // todo
}


/**
 * Get marginal likelihood ratio m(x_{c + j}) / m(x_c)
 *
 * Using the conjugate distribution of the Inverse Wishart,
 *
 * m(x_{c + j}) / m(x_c) =
 *      |S + XX^T|^{(df + N) / 2} * pi^{-D/2}            Gamma_D((df+N+1)/2)
 *      ------------------------------------- *  Prod    -------------------
 *          |S + YY^T|^{(df + N + 1) / 2}       1<=d<=D   Gamma_D((df+N)/2)  
 *
 * Where D = # of dimensions, N = # of samples, df = Wishart degrees of
 * freedom, X = data matrix, Y = data matrix with data point j.
 *
 * @param pt : data point to compute marginal likelihood for
 * @return Marginal likelihood ratio m(x_{c + j}) / m(x_c)
 */
NormalWishartComponent::marginal_likelihood_ratio(int pt) {
    // todo
}


/**
 * Get mean
 *
 * @return Mean row vector
 */
NormalWishartComponent::mean() {
    return total / points.size();
}