"""Mixture of Finite Mixtures Model (Miller & Harrison, 2018)

References
----------
Jeffrey W. Miller, Matthew T. Harrison (2018),
    "Mixture Models with a Prior on the Number of Components".
    Journal of the American Statistical Association, Vol. 113, Issue 521.
"""

import numpy as np
import math
from scipy.stats import poisson

from bmcc.core import MODEL_MFM


class MFM:
    """Mixture of Finite Mixtures Model (Miller & Harrison, 2018)

    Keyword Args
    ------------
    gamma : float
        Dirichlet parameter (model 2, Miller & Harrison, 2018)
    prior : function(int) -> float
        Prior likelihood on the number of clusters; p_K in model 2.
        Defaults to geometric(0.5).
    error : float
        Margin for computation of V_n coefficients
        (equation 3.2, Miller & Harrison, 2018)

    Attributes
    ----------
    CAPSULE : capsule
        Capsule containing component methods (export from C module)
    """

    CAPSULE = MODEL_MFM

    def __init__(self, gamma=1, prior=None, error=0.001):

        self.gamma = gamma
        self.error = error
        if prior is None:
            def prior(k):
                return poisson.logpmf(k, 4)
        self.prior = prior

    def log_v_n(self, N):
        """Get log(V_n(t)) for 1<=t<=N (equation 3.2, Miller & Harrison, 2018).

        Parameters
        ----------
        N : int
            Number of data points (and therefore the maximum possible number of
            clusters)

        Returns
        -------
        np.array
            Computed log(V_n(t)) coefficients
        """

        res = np.zeros(N, dtype=np.float64)

        # Compute
        for t in range(1, N + 1):
            prev = 0
            current = np.NINF
            k = t  # skip first t terms since they're equal to 0
            while True:
                prev = current
                term = (
                    self.prior(k) +
                    math.lgamma(k + 1) +
                    math.lgamma(self.gamma * k) -
                    math.lgamma(k - t + 1) -
                    math.lgamma(self.gamma * k + N)
                )
                current = np.logaddexp(current, term)
                if current - prev < self.error:
                    break
                k += 1
            res[t - 1] = current

        return res

    def get_args(self, data, assignments):
        """Get Model Hyperparameters

        Parameters
        ----------
        data : np.array
            Data array to run on

        Returns
        -------
        dict
            Argument dictionary to be passed to core.init_mode, with entries:
            "V_n": log(V_n(t)) coefficients
                (equation 3.2, Miller & Harrison, 2018)
            "gamma": Dirichlet distribution parameters
                (model 2, Miller & Harrison, 2018)
        """

        return {
            "V_n": self.log_v_n(data.shape[0]),
            "gamma": float(self.gamma)
        }

    def update(self, mixture):
        """Run Hyperparameter update

        Parameters
        ----------
        mixture : MixtureModel object
            Object to update for

        Returns
        -------
        None
            No updates for MFM model.
        """
        return None
