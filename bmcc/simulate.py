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
    symmetric : bool
        If False, sample cluster covariances from a normal-wishart with df=n.
        Else, set each cluster covariance as the identity.
    shuffle : bool
        If False, sorts the assignments. Use this to keep the pairwise matrices
        clean.

    Attributes
    ----------
    API_NAME : str
        String identifier for npz objects saved by this class. This class will
        only save to and load from npz files with the attribute
        f[API_NAME] = True.
    """

    __INT_KEYS = ['n', 'k', 'd']
    __FLOAT_KEYS = ['r', 'alpha', 'df']
    __BOOL_KEYS = ['symmetric', 'shuffle']
    __ARRAY_KEYS = [
        'cov', 'means', 'weights',
        'assignments', 'data']

    API_NAME = "bmcc_GaussianMixture"

    def __init__(self, *args, load=False, **kwargs):
        if load:
            self.__init_load(*args, **kwargs)
        else:
            self.__init_new(*args, **kwargs)

    def __init_load(self, src):
        """Load from file"""

        fz = np.load(src)

        if self.API_NAME not in fz:
            raise Exception(
                "Target file is not a valid GaussianMixture save file.")

        for attr in self.__INT_KEYS:
            setattr(self, attr, int(fz[attr]))
        for attr in self.__FLOAT_KEYS:
            setattr(self, attr, float(fz[attr]))
        for attr in self.__BOOL_KEYS:
            setattr(self, attr, bool(fz[attr]))
        for attr in self.__ARRAY_KEYS:
            setattr(self, attr, fz[attr])

    def __init_new(
            self, n=1000, k=3, d=2, r=1, alpha=40, df=None,
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
                cov=np.identity(d) * (alpha * k) ** (1 / d)
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
                        stats.multivariate_normal.pdf(x, mean=mu, cov=cov))
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

    def save(self, dst):
        """Save simulated dataset to a file."""

        save_params = {
            attr: getattr(self, attr)
            for attr in (
                self.__INT_KEYS + self.__BOOL_KEYS +
                self.__FLOAT_KEYS + self.__ARRAY_KEYS
            )
        }
        save_params[self.API_NAME] = True

        np.savez(dst, **save_params)

    def __str__(self):
        return (
            "Simulated Gaussian Mixture [n={}, k={}, d={}, r={}, alpha={}, "
            "df={}, symmetric={}]".format(
                self.n, self.k, self.d, self.r, self.alpha,
                self.df, self.symmetric))

    def __repr__(self):
        return self.__str__()
