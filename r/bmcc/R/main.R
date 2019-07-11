#    _____ _____ _____ _____
#   | __  |     |     |     |
#   | __ -| | | |   --|   --|
#   |_____|_|_|_|_____|_____|
#   Bayesian Markov Chain Clustering
#
# Author
# ------
# Tianshu Huang
# <thetianshuhuang@gmail.com>
#
# Summary
# -------
# Implementation of Markov Chain Bayesian Clustering techniques, including DPM
# (Dirichlet Process Mixture Models [1]) and MFM (Mixture of Finite Mixtures [2])
# mixture models, with an abstract Mixture Model and Component Model API.
#
# Hyperparameter updates for DPM are implemented using an Empirical Bayes update
# procedure [3].
#
# Final configuration selection is implemented using Least Squares clustering
# [4].
#
# References
# ----------
# [1] Radford M. Neal (2000), "Markov Chain Sampling Methods for Dirichlet
#     Process Mixture Models". Journal of Computational and Graphical Statistics,
#     Vol. 9, No. 2.
#
# [2] Jeffrey W. Miller, Matthew T. Harrison (2018),
#     "Mixture Models with a Prior on the Number of Components".
#     Journal of the American Statistical Association, Vol. 113, Issue 521.
#
# [3] Jon D. McAuliffe, David M. Blei, Michael I. Jordan (2006),
#     "Nonparametric empirical Bayes for the Dirichlet process mixture model".
#     Statistics and Computing, Vol. 16, Issue 1.
#
# [4] David B. Dahl (2006), "Model-Based Clustering for Expression Data via a
#     Dirichlet Process Mixture Model". Bayesian Inference for Gene Expression
#     and Proteomics.


#
# -- Load ---------------------------------------------------------------------
#

library(reticulate)

IMPORT_ERROR_MSG = "
'bmcc' python module could not be imported.

bmcc requires Python and the python module 'bmcc' to be installed. Install with
$ pip install bmcc

on the linux / windows terminal.
"

# Check for installation
bmcc <- tryCatch({
    import("bmcc", convert = FALSE)
}, error=function(e) {
    cat(IMPORT_ERROR_MSG)
})

source("cast.R")


#
# -- Functions ----------------------------------------------------------------
#

#' Get membership matrix corresponding to an assignment vector.
#'
#' @param assignments Assignment vector
#'
#' @return Pairwise Membership Matrix
#'
#' @export
membership_matrix <- function(assignments) {
    return(uint16_sanitize(bmcc$membership_matrix))
}


#' Get pairwise probability matrix
#'
#' @param history Assignment history matrix
#' @param burn_in Burn in period
#'
#' @return (Pairwise Membership Matrix, Squared residuals for the history)
#'
#' @export
pairwise_probability <- function(history, burn_in) {
    return(
        bmcc$pairwise_probability(
            uint16_sanitize(history),
            int_sanitize(burn_in)))
}


#
# -- Objects ------------------------------------------------------------------
#

#' Mixture of Finite Mixtures Object
#'
#' @param gamma MFM dirichlet mixing parameter
#' @param prior Prior function: int k -> float likelihood. If NULL, uses
#'              geometric(0.1).
#' @param error Error margin for V_n computation.
#'
#' @return MFM object; use this as the 'mixture_model' for GibbsMixtureModel.
#'
#' @export
MFM <- function(gamma=1, prior=NULL, error=0.001) {
  return(bmcc$MFM(gamma=gamma, prior=prior, error=error))
}


#' Dirichlet Process Mixture Model Object
#'
#' @param alpha DPM mixing parameter
#' @param use_eb Do empirical bayes updates on alpha?
#' @param eb_threshold Threshold to start empirical bayes computation from
#' @param convergence Error threshold for empirical bayes equation
#'
#' @return DPM object; use this as the 'mixture_model' for GibbsMixtureModel.
#'
#' @export
DPM <- function(alpha=1, use_eb=TRUE, eb_threshold=100L, convergence=0.01) {
    return(
        bmcc$DPM(
            alpha=alpha,
            use_eb=use_eb,
            eb_threshold=int_sanitize(eb_threshold),
            convergence=convergence))
}


#' Normal Wishart Component Model Object
#'
#' @param df Degrees of freedom for Wishart distribution
#'
#' @return NormalWishart object; use this as the 'component_model' for
#'         GibbsMixtureModel.
#'
#' @export
NormalWishart <- function(df=2) {
    return(bmcc$NormalWishart(df=df))
}


#' Gibbs Sampler for Abstract Mixture Models
#'
#' @param data Data matrix containing points to cluster on
#' @param component_model Distribution of mixture components
#' @param mixture_model Model for component mixing
#' @param assignments Initial assignments. If NULL, initializes all points to
#'                    the same cluster
#' @param thinning Thinning factor; only keeps 1/thinning samples.
#'
#' @return GibbsMixtureModel object.
#'
#' @export
GibbsMixtureModel <- function(
    data,
    component_model=NULL, mixture_model=NULL,
    assignments=NULL, thinning=1L) {

    if(!is.null(assignments)) {
        assignments <- uint16_sanitize(assignments)
    }
    return(
        bmcc$GibbsMixtureModel(
            float64_sanitize(data),
            component_model=component_model,
            mixture_model=mixture_model,
            assignments=assignments,
            thinning=thinning))
}


#' Perform Least Squares analysis on the output of a MCMC clustering algorithm.
#'
#' @param data Data matrix containing points the clustering was performed on
#' @param hist History of clusterings output by the algorithm
#' @param burn_in Number of samples to discard as burn-in
#'
#' @result LstsqResult object.
#'
#' @export
LstsqResult <- function(data, hist, burn_in=0L) {
    return(
        bmcc$LstsqResult(
            float64_sanitize(data),
            uint16_sanitize(hist),
            int_sanitize(burn_in)))
}


#' Simulate Gaussian Mixture Model with Normal-Wishart components
#'
#' @param n Number of data points
#' @param k Number of clusters
#' @param d Number of dimensions
#' @param r Balance parameter; the kth cluster has assignment probability ~ r^k
#' @param alpha Concentration parameter; higher alpha will result in cluster
#'              means being farther apart
#' @param df Degrees of freedom for Normal-Wishart. If NULL, uses df = d.
#' @param symmetric If TRUE, uses N(0, I) instead of sampling from wishart.
#' @param shuffle If TRUE, created points are in random order.
#'
#' @return GaussianMixture object
#'
#' @export
GaussianMixture <- function(
        n=1000L, k=3L, d=2L, r=1,
        alpha=10, df=NULL, symmetric=FALSE, shuffle=TRUE) {
    return(bmcc$GaussianMixture(
        n=int_sanitize(n),
        k=int_sanitize(k),
        d=int_sanitize(d),
        r=r, alpha=alpha, df=df, symmetric=symmetric, shuffle=shuffle))
}
