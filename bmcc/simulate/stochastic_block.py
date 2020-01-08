"""Simulate Stochastic Block Model
"""


import numpy as np
from scipy import stats

from .base_model import BaseModel, raw
from bmcc.core import sbm_simulate


class StochasticBlockModel(BaseModel):

    _KEYS = {
        int: ['n', 'k', 'd'],
        float: ["alpha", "beta"],
        bool: ['shuffle'],
        raw: ['Q', 'data'],
    }

    API_NAME = "bmcc_StochasticBlockModel"
    MODEL_NAME = "Stochastic Block Model"

    def _init_new(
            self, n=1000, k=3, r=1, Q=None, shuffle=True, alpha=1, beta=1):

        self.n = n
        self.k = k
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.shuffle = shuffle

        self.weights, self.assignments = self._make_assignments(
            n, k, r, shuffle)

        if Q is None:
            self.Q = np.zeros((self.k, self.k), dtype=np.float64)
            for i in range(self.k):
                for j in range(i + 1):
                    self.Q[i, j] = stats.beta.rvs(alpha, beta)
                    self.Q[j, i] = self.Q[i, j]
        else:
            assert(Q.shape == (self.k, self.k))
            self.Q = Q

        self.data = np.zeros((self.n, self.n), dtype=np.uint8)
        sbm_simulate(self.Q, self.data, self.assignments)
