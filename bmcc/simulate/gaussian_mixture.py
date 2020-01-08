"""Simulate Gaussian Mixture

Based on scheme described by Miller, Harrison (Miller & Harrison, 2018) with
normal-wishart
components.
Some modifications are made: mixing weights are fixed as a
geometric progression instead of sampled from a dirichlet or gamma distribution
in order to simpilify comparison, and means are sampled from a symmetric
multivariate normal instead of being fixed.

References
----------
Jeffrey W. Miller, Matthew T. Harrison (2018),
    "Mixture Models with a Prior on the Number of Components".
    Journal of the American Statistical Association, Vol. 113, Issue 521.
"""


import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from ..analysis import plot_clusterings
from .base_model import BaseModel, raw
from bmcc.core import oracle_matrix


class GaussianMixture(BaseModel):
    """Simulate A Gaussian Mixture Dataset.

    Keyword Args
    ------------
    load : bool
        If True, instead takes a single string argument, which should be a file
        containing a saved GaussianMixture object. Defaults to False.
    n : int
        Number of data points
    k : int
        Number of clusters
    d : int
        Number of dimensions
    r : float
        Balance ratio; the nth cluster has a weight of r^n.
    alpha : float
        Density parameter; larger alpha results in more separation between
        cluster centers.
    df : float
        Degrees of freedom for wishart distribution
    means : array-like
        If not None, these are used as the means instead of sampling means
        from a symmetric normal.
    symmetric : bool
        If False, sample cluster covariances from a normal-wishart with df=n.
        Else, set each cluster covariance as the identity.
    shuffle : bool
        If False, sorts the assignments. Use this to keep the pairwise matrices
        clean.
    """

    _KEYS = {
        int: ['n', 'k', 'd'],
        float: ['r', 'alpha', 'df'],
        bool: ['symmetric', 'shuffle'],
        raw: ['cov', 'means', 'weights', 'assignments', 'data']
    }

    API_NAME = "bmcc_GaussianMixture"
    MODEL_NAME = "Gaussian Mixture"

    def _init_new(
            self, n=1000, k=3, d=2, r=1, alpha=40, df=None, means=None,
            symmetric=False, shuffle=True):
        """Initialize New Gaussian Mixture Simulation"""

        if df is None:
            df = d

        # Save params
        self.n = n
        self.k = k
        self.d = d
        self.r = r
        self.alpha = alpha
        self.df = df
        self.symmetric = symmetric
        self.shuffle = shuffle

        # Make assignments
        self.weights, self.assignments = self._make_assignments(
            n, k, r, shuffle)

        # Means: normal, with radius proportional to clusters
        if means is None:
            self.means = [
                stats.multivariate_normal.rvs(
                    mean=np.zeros(d),
                    cov=np.identity(d) * (alpha * k) ** (1 / d)
                ) for _ in range(k)
            ]
        else:
            self.means = means

        # Covariances: normal wishart (if not symmetric), else normal
        if symmetric:
            self.cov = [np.identity(d) for _ in range(k)]
        else:
            self.cov = [
                stats.wishart.rvs(df, np.identity(d)) for _ in range(k)
            ]

        # Points
        self.data = np.zeros((n, d), dtype=np.float64)
        for idx, _ in enumerate(self.data):
            self.data[idx, :] = stats.multivariate_normal.rvs(
                mean=self.means[self.assignments[idx]],
                cov=self.cov[self.assignments[idx]])

    @property
    def likelihoods(self):
        """Likelihood table"""

        if not hasattr(self, "__likelihoods"):
            self.__likelihoods = np.zeros((self.n, self.k))
            for idx, x in enumerate(self.data):
                __iter = enumerate(zip(self.weights, self.means, self.cov))
                # Calculate likelihoods
                for k, (weight, mu, cov) in __iter:
                    self.__likelihoods[idx, k] = (
                        weight *
                        stats.multivariate_normal.pdf(
                            x, mean=mu, cov=cov, allow_singular=True))
                # Normalize
                self.__likelihoods[idx] /= sum(self.__likelihoods[idx])

        return self.__likelihoods

    @property
    def oracle(self):
        """Oracle assignments"""

        lk = self.likelihoods

        if not hasattr(self, "__oracle"):
            # Compute oracle clustering (maximum likelihood given all
            # parameters)
            self.__oracle = np.zeros(self.n, dtype=np.uint16)
            for idx in range(self.n):
                self.__oracle[idx] = np.argmax(lk[idx])

        return self.__oracle

    @property
    def oracle_matrix(self):
        """Oracle Pairwise Probability Matrix"""

        if not hasattr(self, "__oracle_matrix"):
            self.__oracle_matrix = oracle_matrix(self.likelihoods)

        return self.__oracle_matrix

    def plot_actual(self, plot=False, **kwargs):
        """Plot actual clusterings (binding to bmcc.plot_clusterings)"""
        fig = plot_clusterings(self.data, self.assignments, **kwargs)

        if plot:
            plt.show()
            return None
        else:
            return fig

    def plot_oracle(self, plot=False, **kwargs):
        """Plot oracle clusterings (binding to bmcc.plot_oracle)"""
        fig = plot_clusterings(self.data, self.oracle, **kwargs)

        if plot:
            plt.show()
            return None
        else:
            return fig

    def __str__(self):
        return (
            "Simulated Gaussian Mixture [n={}, k={}, d={}, r={}, alpha={}, "
            "df={}, symmetric={}]".format(
                self.n, self.k, self.d, self.r, self.alpha,
                self.df, self.symmetric))

    def __repr__(self):
        return self.__str__()
