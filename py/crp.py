
import numpy as np

from gaussian_mixins import GaussianMixins
from gibbs_clustering import GibbsClustering


class CRPGibbs(GaussianMixins, GibbsClustering):

    def coef_cond(self, data, cluster, idx, params):

        return pow(len(cluster), params["r"])

    def coef_uncond(self, data, idx, params):

        return params["alpha"]


if __name__ == '__main__':

    from scipy.stats import norm
    from matplotlib import pyplot as plt

    data = np.concatenate((
        norm.rvs(0, 1, size=200),
        norm.rvs(10, 1, size=200),
        norm.rvs(12, 1, size=200)))

    data = data.reshape((-1, 1))

    crp = CRPGibbs(
        data, params={"r": 1, "sigma_mu": 1, "sigma_x": 1, "alpha": 1})

    for i in range(20):
        crp.iter()

    lstsq, means, idx = crp.select_lstsq()
    plt.subplot(1, 2, 1)
    plt.imshow(means)
    plt.subplot(1, 2, 2)
    plt.imshow(lstsq)
    plt.show()
