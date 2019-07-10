"""Analysis Routines

Implements Least Squares configuration selection [1].

References
----------
[1] David B. Dahl (2006), "Model-Based Clustering for Expression Data via a
    Dirichlet Process Mixture Model". Bayesian Inference for Gene Expression
    and Proteomics.
"""


import numpy as np
from bclust.core import (
    pairwise_probability,
    aggregation_score,
    segregation_score)
from bclust.plot import plot_clusterings

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from matplotlib import pyplot as plt


def membership_matrix(asn):
    """Get membership matrix of an assignment vector.

    Parameters
    ----------
    asn : array-like
        Cluster assignment vector a

    Returns
    -------
    np.array
        Membership matrix M(a):
        M_{i,j}(a) = 1_{a[i] = a[j]}
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
        Pairwise Probability Matrix [1]
    residuals : np.array
        Squared residuals between membership matrix of each iteration and
        pairwise probability matrix [1]
    best_idx : int
        Index of 'best' clustering configuration according to residuals [1]
    best : np.array
        Best clustering configuration [1]
    num_clusters : np.array
        Trace of number of clusters
    nmi : np.array
        Normalized Mutual Information trace.
    nmi_best : float
        NMI of 'best' configuration (as selected above)
    rand : np.array
        Rand index trace.
    rand_best : float
        Rand Index of 'best' configuration (as selected above)
    aggregation : np.array
        Aggregation score trace: for selected assignment vector a, true
        assignment vector A,
        P[a[i] = a[j] | A[i] = A[j]]
    aggregation_best : float
        Aggregation score of 'best' configuration (as selected above)
    segregation : np.array
        Segregation score trace:
        P[a[i] != a[j] | A[i] != A[j]]
    segregation_best : float
        Segregation score of 'best' configuration (as selected above)
    oracle : np.array
        Oracle clustering assignments
    oracle_nmi, oracle_rand, oracle_segregation, oracle_aggregation : float
        Scores for oracle clustering

    """

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

        self.num_clusters = np.array([np.max(x) + 1 for x in self.hist])

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

        self.aggregation = np.array([
            aggregation_score(actual, x) for x in self.hist])
        self.aggregation_best = self.aggregation[self.best_idx]
        self.segregation = np.array([
            segregation_score(actual, x) for x in self.hist])
        self.segregation_best = self.segregation[self.best_idx]

    def evaluate(self, actual, oracle=None, nmi_method='arithmetic'):
        """Evaluate clustering against ground truth.

        Parameters
        ----------
        actual : np.array
            Actual assignments
        oracle : np.array
            Oracle assignments (used as point of comparison instead of
            'perfect' clustering). Used only if present.
        nmi_method : str
            Normalization method for NMI calculations
        """

        if actual.dtype != np.uint16:
            print("Assignments cast to np.uint16.")
            actual = actual.astype(np.uint16)

        self.actual = actual
        self.clusters_true = np.max(actual) + 1

        if oracle is not None:
            self.oracle = oracle
            self.oracle_nmi = normalized_mutual_info_score(
                actual, oracle, nmi_method)
            self.oracle_rand = adjusted_rand_score(actual, oracle)
            self.oracle_aggregation = aggregation_score(actual, oracle)
            self.oracle_segregation = segregation_score(actual, oracle)

        else:
            self.oracle = actual
            self.oracle_nmi = 1
            self.oracle_rand = 1

        self.truth_known = True
        self.__score_trace(actual, nmi_method)

    DEFAULT_TRACE_PLOTS = {
        "Metrics (NMI, Rand)": {
            "Normalized Mutual Information": "nmi",
            "Adjusted Rand Index": "rand",
            "Oracle NMI": "oracle_nmi",
            "Oracle Rand": "oracle_rand"
        },
        "Number of Clusters": {
            "Number of Clusters": "num_clusters",
            "Actual": "clusters_true"
        },
        "Aggregation / Segregation Score": {
            "Aggregation Score": "aggregation",
            "Segregation Score": "segregation",
            "Oracle Aggregation": "oracle_aggregation",
            "Oracle Segregation": "oracle_segregation"
        }
    }

    def trace(self, plots=DEFAULT_TRACE_PLOTS):
        """Plot NMI, Rand Index, and # of clusters as a trace over MCMC
        iterations.

        Parameters
        ---------
        plots : dict
            Dictionary of (key, value) pairs. Each key corresponds to a plot,
            which is plotted with the attribute corresponding to value. If
            value is a dict, multiple keys are plotted on the same plot.

        Returns
        -------
        plt.figure.Figure
            Created figure; plot with fig.show().
        """

        if not self.truth_known:
            raise ValueError(
                "Cannot show scores: ground truth not known.")

        if len(plots) > 1:
            fig, axs = plt.subplots(len(plots), 1)
        else:
            fig, axs = plt.subplots(1, 1)
            axs = [axs]

        for ax, (k, v) in zip(axs, plots.items()):
            ax.set_title(k)
            ax.axvline(self.burn_in, color='black', label='Burn In')
            if type(v) == str:
                ax.plot(getattr(self, v))
            elif type(v) == dict:
                for k_inner, v_inner in v.items():
                    if np.isscalar(getattr(self, v_inner)):
                        ax.plot(
                            [0, self.hist.shape[0]],
                            [getattr(self, v_inner), getattr(self, v_inner)],
                            label=k_inner)
                    else:
                        ax.plot(getattr(self, v_inner), label=k_inner)
                ax.legend(loc='lower right')
            else:
                raise TypeError(
                    "Attribute name is not a string or dict.")

        return fig

    def matrices(self):
        """Show pairwise probability matrix and membership matrix of least
        squares configuration.

        Returns
        -------
        plt.figure.Figure
            Created figure; plot with fig.show().
        """

        if not self.truth_known:
            raise ValueError(
                "Cannot show scores: ground truth not known.")

        fig, p = plt.subplots(2, 2)

        p[0][0].matshow(self.matrix)
        p[0][0].set_title("Pairwise Probability Matrix")

        p[0][1].matshow(membership_matrix(self.best))
        p[0][1].set_title("Membership Matrix of Least Squares Configuration")

        p[1][0].matshow(membership_matrix(self.actual))
        p[1][0].set_title("Membership Matrix of Actual Clusters")

        p[1][1].matshow(membership_matrix(self.oracle))
        p[1][1].set_title("Membership Matrix of Oracle Clustering")

        return fig

    def clustering(self, bins=20):
        """Show clustering as a multi-dimensional array of scatterplots

        Parameters
        ----------
        bins : int
            Number of bins

        Returns
        -------
        plt.figure.Figure
            Created figure; plot with fig.show().
        """

        return plot_clusterings(self.data, self.best, bins=bins)
