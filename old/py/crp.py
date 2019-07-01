
import numpy as np

from gaussian import GaussianMixins
from gibbs_clustering import GibbsClustering


class CRPMixins:
    """Mixins coef_cond and coef_uncond for pCRP clustering prior"""

    REQUIRED_COEF_PARAMS = ["r", "alpha"]

    def coef_cond(self, data, cluster, idx, params):
        """Conditional coefficient function

        Coefficient of p(point | cluster): (# in cluster)^r
        """

        return pow(len(cluster), params["r"])

    def coef_uncond(self, data, idx, params):
        """Unconditional coefficient function

        Coefficient of p(point): concentration parameter alpha
        """

        return params["alpha"]


class CRPGaussianGibbs(CRPMixins, GaussianMixins, GibbsClustering):
    pass


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy.stats import norm

    data = np.concatenate((
        norm.rvs(0, 1, size=100),
        norm.rvs(10, 1, size=100),
        norm.rvs(12, 1, size=100))).astype(np.float64)

    data = data.reshape((-1, 1))

    crp = CRPGaussianGibbs(
        data, params={
            "r": 1, "alpha": 1,
            "mean": 6,
            "cov_means": np.array([[5]]),
            "cov_within": np.array([[1]])})

    for i in range(20):
        print(i, len([c for c in crp.clusters if len(c) > 0]))
        crp.iter()

    means, assignments, pairwise, pairwise_prob = crp.select_lstsq()
    print(means)
    plt.subplot(1, 2, 1)
    plt.imshow(pairwise)
    plt.subplot(1, 2, 2)
    plt.imshow(pairwise_prob)
    plt.show()

