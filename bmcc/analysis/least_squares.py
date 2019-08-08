"""Analysis Routines

Implements Least Squares configuration selection [1].

References
----------
[1] David B. Dahl (2006), "Model-Based Clustering for Expression Data via a
    Dirichlet Process Mixture Model". Bayesian Inference for Gene Expression
    and Proteomics.
"""


import numpy as np
from bmcc.core import pairwise_probability
from bmcc.base_result import BaseResult

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


class LstsqResult(BaseResult):
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
    DEFAULT_TRACE_PLOTS:
        Show NMI/Rand, Num Clusers, and Aggregation/Segregation by default
    matrix : np.array
        Pairwise Probability Matrix [1]
    residuals : np.array
        Squared residuals between membership matrix of each iteration and
        pairwise probability matrix [1]
    best_idx : int
        Index of 'best' clustering configuration according to residuals [1]
    best : np.array
        Best clustering configuration [1]
    """

    def cluster(self):
        """Obtain clustering configuration; calling specified by BaseResult"""

        # Get pairwise probability matrix and run least squares procedure
        (self.matrix,
         self.residuals) = pairwise_probability(self.hist, self.burn_in)

        self.best_idx = np.argmin(self.residuals)
        self.best = self.hist[self.best_idx, :]

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

    def matrices(self, plot=False):
        """Show pairwise probability matrix and membership matrix of least
        squares configuration.

        Parameters
        ----------
        plot : bool
            If True, calls plt.show(). Otherwise, returns the generated figure.

        Returns
        -------
        plt.figure.Figure or None
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

        p[1][0].matshow(self.oracle_matrix)
        p[1][0].set_title("Oracle Pairwise Probability Matrix")

        p[1][1].matshow(membership_matrix(self.oracle))
        p[1][1].set_title("Membership Matrix of Oracle Clustering")

        if plot:
            plt.show()
            return None
        else:
            return fig
