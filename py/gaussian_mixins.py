
import numpy as np
from scipy import stats


class GaussianMixins:
    """Mixins l_cond and l_uncond for likelihood functions of gaussian priors

    Use by mixing this into GibbsClustering and another class that provides
    coefficient functions.
    """

    def l_cond(self, data, cluster, idx, params):
        """Conditional likelihood.

        Returns p(data[idx] assigned to cluster | cluster). Approximates
        covariance by the observed values in the cluster.
        """

        cluster_data = np.array([data[i] for i in cluster if i != idx])

        # Get mu, sigma
        # We assume sigma > 1 in order to avoid situations where n=1 -> sigma=0
        # sigma_k = max(np.sqrt(np.var(cluster_data, axis=0)), 1)
        sigma_k = 1
        mu_k = np.mean(cluster_data, axis=0)

        return stats.multivariate_normal.pdf(
            data[idx], mean=mu_k, cov=sigma_k * np.identity(data.shape[1]))

    def l_uncond(self, data, idx, params):
        """Unconditional likelihood.

        Uses the covariance matrices specified in the parameters.
        """

        cov = (
            np.sqrt(
                params["sigma_mu"]**2 + params["sigma_x"]**2
            ) * np.identity(data.shape[1]))

        return stats.multivariate_normal.pdf(data[idx], mean=6, cov=cov)
