
import numpy as np
from scipy import stats

import cluster_util


class ConstantGaussianMixin:
    """Mixins l_cond and l_uncond for likelihood functions of constant
    gaussian priors where each cluster has the same covariance

    Use by mixing this into GibbsClustering and another class that provides
    coefficient functions.
    """

    REQUIRED_LIKELIHOOD_PARAMS = ["cov_means", "cov_within", "mean"]
    LIKELIHOOD_INIT = False

    def l_cond(self, data, cluster, idx, params):
        """Conditional likelihood.

        Returns p(data[idx] assigned to cluster | cluster). Approximates
        covariance by the observed values in the cluster.
        """

        # Get mu, sigma
        # We assume sigma > 1 in order to avoid situations where n=1 -> sigma=0
        mean, cov = cluster_util.cov_by_index(data, cluster)

        return stats.multivariate_normal.pdf(
            data[idx], mean=mean, cov=params["cov_within"])

    def l_uncond(self, data, idx, params):
        """Unconditional likelihood.

        Uses the covariance matrices specified in the parameters.
        """

        return stats.multivariate_normal.pdf(
            data[idx],
            mean=params["mean"],
            cov=params["cov_means"]**2 + params["cov_within"]**2)


class GaussianWishartMixin:

    LIKELIHOOD_INIT = True

    def _likelihood_init(self):
        """Initialization function for data-dependent Gaussian-Wishart prior
        (simulation 7.1 in "Mixture Models with a Prior on the Number of
        Components", Miller, Harrison)
        """

        # Get C^, mu^, V
        self.__cov = np.cov(self.data.T)
        self.__mean = np.mean(self.data, axis=0)
        self.__wishart_scale = np.linalg.inv(self.__cov) / self.data.shape[1]
