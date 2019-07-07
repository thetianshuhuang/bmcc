import numpy as np
from bclust.core import pairwise_probability
from bclust.plot import plot_clusterings

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from matplotlib import pyplot as plt


def membership_matrix(asn):
    """Get membership matrix of an assignment vector.

    Parameters
    ----------
    asn : array-like
        Cluster assignment vector

    Returns
    -------
    np.array
        Membership matrix of asn.
    """

    res = np.zeros((len(asn), len(asn)))
    for i in range(len(asn)):
        for j in range(len(asn)):
            if asn[i] == asn[j]:
                res[i, j] = 1
    return res


class LstsqResult:
    """Least squares analysis on MCMC results

    Parameters
    ----------
    hist : np.array
        Array containing MCMC results; each assignment is a row.
    burn_in : int
        Burn in duration
    actual : np.array
        Array containing ground truth clusterings. If None, then it is assumed
        that no ground truth is available.
    nmi_method : str
        NMI method for mutual information calculations
        (see sklearn.metrics.normalized_mutual_info_score.html)

    Attributes
    ----------
    hist : np.array
        History array (same as the input)
    burn_in : int
        Burn in duration used in analysis
    matrix : np.array
        Pairwise Probability Matrix
    residuals : np.array
        Squared residuals between membership matrix of each iteration and
        pairwise probability matrix
    best_idx : int
        Index of 'best' clustering configuration according to squared residuals
    best : np.array
        Best clustering configuration
    nmi : np.array
        Normalized Mutual Information trace.
    nmi_best : np.array
        NMI of 'best' configuration (as selected above)
    rand : np.array
        Rand index trace.
    rand_best : np.array
        Rand Index of 'best' configuration (as selected above)
    """

    def __score_trace(self, actual, nmi_method):
        """Compute NMI and Rand Index trace.

        Parameters
        ----------
        actual : np.array
            Ground truth assignments
        """
        if type(actual) != np.ndarray:
            raise TypeError(
                "Ground truth clustering ('actual') must be a numpy "
                "array.")
        if len(actual.shape) != 1 or actual.shape[0] != self.hist.shape[1]:
            raise TypeError(
                "Ground truth clustering must have the same "
                "dimensionality as the MCMC history.")
        self.actual = actual

        self.nmi = np.array([
            normalized_mutual_info_score(actual, x, nmi_method)
            for x in self.hist])
        self.nmi_best = self.nmi[self.best_idx]

        self.rand = np.array([
            adjusted_rand_score(actual, x)
            for x in self.hist])
        self.rand_best = self.rand[self.best_idx]

    def __init__(self, data, hist, burn_in=0):

        self.data = data
        self.hist = hist
        self.burn_in = burn_in
        self.truth_known = False

        # Check burn in
        if type(burn_in) != int:
            raise TypeError("Burn in period must be an integer.")
        if burn_in >= hist.shape[0]:
            raise ValueError(
                "Burn in period larger than the number of saved samples.")

        # Get pairwise probability matrix and run least squares procedure
        (self.matrix,
         self.residuals) = pairwise_probability(self.hist, self.burn_in)

        self.best_idx = np.argmin(self.residuals)
        self.best = self.hist[self.best_idx, :]

        self.num_clusters = np.array([np.max(x) for x in self.hist])

    def evaluate(self, actual, nmi_method='arithmetic'):
        """Evaluate clustering against ground truth.

        Parameters
        ----------
        actual : np.array
            Actual assignments
        nmi_method : str
            Normalization method for NMI calculations
        """

        self.actual = actual
        self.clusters_true = np.max(actual) + 1
        self.truth_known = True
        self.__score_trace(actual, nmi_method)

    def trace(self):
        """Plot NMI, Rand Index, and # of clusters as a trace over MCMC
        iterations.
        """

        if not self.truth_known:
            raise ValueError("Ground truth comparison not run.")

        fig, (top, middle, bottom) = plt.subplots(3, 1)

        top.plot(self.nmi)
        top.axhline(1)
        top.set_title("Normalized Mutual Information")

        middle.plot(self.rand)
        middle.axhline(1)
        middle.set_title("Adjusted Rand Index")

        bottom.plot(self.num_clusters)
        bottom.axhline(self.clusters_true)
        bottom.set_title("Number of Clusters")

        plt.show()

    def pairwise(self):
        """Show pairwise probability matrix and membership matrix of least
        squares configuration.
        """

        fig, (left, right) = plt.subplots(1, 2)

        left.matshow(self.matrix)
        left.set_title("Pairwise Probability Matrix")

        right.matshow(membership_matrix(self.best))
        right.set_title("Membership Matrix of Least Squares Configuration")

        plt.show()

    def clustering(self, bins=20):
        """Show clustering as a multi-dimensional array of scatterplots

        Parameters
        ----------
        bins : int
            Number of bins
        """

        plot_clusterings(self.data, self.best, bins=bins)
