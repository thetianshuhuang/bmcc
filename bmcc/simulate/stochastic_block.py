"""Simulate Stochastic Block Model

References
----------
Junxian Geng, Anirban Bhattacharya, Depdeep Pati (2018),
    "Probabilistic community detection with unknown number of communities".
    Journal of the American Statistical Association, Vol. 114, Issue 526.
"""


import numpy as np
from scipy import stats

from .base_model import BaseModel, raw
from bmcc.core import sbm_simulate


class StochasticBlockModel(BaseModel):
    """Simulate A Stochastic Block Model Dataset.

    Keyword Args
    ------------
    load : bool
        If True, instead takes a single string argument, which should be a file
        containing a saved GaussianMixture object. Defaults to False.
    n : int
        Number of data points
    k : int
        Number of clusters
    r : float
        Balance ratio; the nth cluster has a weight of r^n.
    Q : np.array
        Q array to simulate from. If None, generates a Q array with independent
        beta(a, b) random variables for each entry.
    a, b : float
        Parameters for Beta distribution. Not used if Q is provided.
    shuffle : bool
        If False, sorts the assignments. Use this to keep the pairwise matrices
        clean.
    """

    _KEYS = {
        int: ['n', 'k'],
        float: ["a", "b"],
        bool: ['shuffle'],
        raw: ['Q', 'data', 'assignments'],
    }

    API_NAME = "bmcc_StochasticBlockModel"
    MODEL_NAME = "Stochastic Block Model"

    def _init_new(
            self, n=1000, k=3, r=1, Q=None, shuffle=True, a=1, b=1):

        self.n = n
        self.k = k
        self.r = r
        self.a = a
        self.b = b
        self.shuffle = shuffle

        self.weights, self.assignments = self._make_assignments(
            n, k, r, shuffle)

        if Q is None:
            self.Q = np.zeros((self.k, self.k), dtype=np.float64)
            for i in range(self.k):
                for j in range(i + 1):
                    self.Q[i, j] = stats.beta.rvs(a, b)
                    self.Q[j, i] = self.Q[i, j]
        else:
            assert(Q.shape == (self.k, self.k))
            self.Q = Q

        self.data = np.zeros((self.n, self.n), dtype=np.uint8)
        sbm_simulate(self.Q, self.data, self.assignments)
