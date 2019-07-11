

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


#' @export
CONFIG <- bmcc$CONFIG

#' @export
MODEL_DPM <- bmcc$MODEL_DPM
#' @export
MODEL_MFM <- bmcc$MODEL_MFM
#' @export
COMPONENT_NORMAL_WISHART <- bmcc$COMPONENT_NORMAL_WISHART

#' @export
pairwise_probability <- function(history, burn_in) {
    return(
        bmcc$pairwise_probability(
            uint16_sanitize(history),
            int_sanitize(burn_in)))
}

#' @export
MFM <- function(gamma=1, prior=NULL, error=0.001) {
  return(bmcc$MFM(gamma=gamma, prior=prior, error=error))
}

#' @export
DPM <- function(alpha=1, use_eb=TRUE, eb_threshold=100L, convergence=0.01) {
    return(
        bmcc$DPM(
            alpha=alpha,
            use_eb=use_eb,
            eb_threshold=int_sanitize(eb_threshold),
            convergence=convergence))
}

#' @export
NormalWishart <- function(df=2) {
    return(bmcc$NormalWishart(df=df))
}

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

#' @export
LstsqResult <- function(data, hist, burn_in=0L) {
    return(
        bmcc$LstsqResult(
            float64_sanitize(data),
            uint16_sanitize(hist),
            int_sanitize(burn_in)))
}

#' @export
membership_matrix <- function(assignments) {
    return(uint16_sanitize(bmcc$membership_matrix))
}

#' @export
GaussianMixture <- function(
        n=1000L, k=3L, d=2L, r=1,
        alpha=40, df=NULL, symmetric=FALSE, shuffle=TRUE) {
    return(bmcc$GaussianMixture(
        n=int_sanitize(n),
        k=int_sanitize(k),
        d=int_sanitize(d),
        r=r, alpha=alpha, df=df, symmetric=symmetric, shuffle=shuffle))
}
