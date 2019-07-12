"""Simulate Gaussian Mixture

Based on scheme described by Miller, Harrison [1] with normal-wishart
components.
Some modifications are made: mixing weights are fixed as a
geometric progression instead of sampled from a dirichlet or gamma distribution
in order to simpilify comparison, and means are sampled from a symmetric
multivariate normal instead of being fixed.

References
----------
[1] Jeffrey W. Miller, Matthew T. Harrison (2018),
    "Mixture Models with a Prior on the Number of Components".
    Journal of the American Statistical Association, Vol. 113, Issue 521.
"""


import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from bmcc.plot import plot_clusterings
from bmcc.core import oracle_matrix


class GaussianMixture:
    """Simulate A Gaussian Mixture Dataset.

    Parameters
    ----------
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
    symmetric : bool
        If False, sample cluster covariances from a normal-wishart with df=n.
        Else, set each cluster covariance as the identity.
    shuffle : bool
        If False, sorts the assignments. Use this to keep the pairwise matrices
        clean.
    """

    def __init__(
            self, n=1000, k=3, d=2, r=1, alpha=5, df=None,
            symmetric=False, shuffle=True):

        if df is None:
            df = d

        # Save params
        self.n = n
        self.k = k
        self.d = d
        self.r = r
        self.alpha = alpha
        self.symmetric = symmetric
        self.df = df

        # Compute weights
        self.weights = np.array([r**i for i in range(k)])
        self.weights = self.weights / sum(self.weights)

        # Make assignments
        self.assignments = np.random.choice(
            k, size=n, p=self.weights).astype(np.uint16)
        if not shuffle:
            self.assignments.sort()

        # Means: normal, with radius proportional to clusters
        self.means = [
            stats.multivariate_normal.rvs(
                mean=np.zeros(d),
                cov=np.identity(d) * alpha * k
            ) for _ in range(k)
        ]

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

        # Likelihood table
        self.likelihoods = np.zeros((n, k))
        for idx, x in enumerate(self.data):
            __iter = enumerate(zip(self.weights, self.means, self.cov))
            # Calculate likelihoods
            for k, (weight, mu, cov) in __iter:
                self.likelihoods[idx, k] = (
                    weight *
                    stats.multivariate_normal.pdf(x, mean=mu, cov=cov))
            # Normalize
            self.likelihoods[idx] /= sum(self.likelihoods[idx])

        # Compute oracle clustering (maximum likelihood given all parameters)
        self.oracle = np.zeros(n, dtype=np.uint16)
        for idx in range(n):
            self.oracle[idx] = np.argmax(self.likelihoods[idx])

        # Compute pairwise probability matrix from likelihood table
        self.oracle_matrix = oracle_matrix(self.likelihoods)

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
