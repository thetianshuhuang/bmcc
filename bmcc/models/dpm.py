"""Dirichlet Process Mixture Model with Empirical Bayes updates on mixing
parameter alpha

References
----------
Jon D. McAuliffe, David M. Blei, Michael I. Jordan (2006),
    "Nonparametric empirical Bayes for the Dirichlet process mixture model".
    Statistics and Computing, Vol. 16, Issue 1.
"""

import numpy as np
from scipy import optimize

from bmcc.core import MODEL_DPM


class DPM:
    """Dirichlet Process Mixture Model (also known as Chinese Restaurant
    Process Model)

    Keyword Args
    ------------
    alpha : float
        DPM mixing parameter
    use_eb : bool
        Do empirical bayes updates on alpha?
    eb_threshold : int
        When do we start doing empirical bayes updates? (since EB is extremely
        unstable for very small N)
    convergence : float
        Convergence criteria for numerical solver of EB update
        (equation 8 in McAuliffe et. al, 2006). Since exact accuracy of alpha
        is not critical, a relatively large margin (default=0.01) can be used.

    Attributes
    ----------
    CAPSULE : capsule
        Capsule containing component methods (export from C module)
    """

    CAPSULE = MODEL_DPM

    def __init__(
            self, alpha=1,
            use_eb=False, eb_threshold=100, convergence=0.01):

        self.alpha = alpha
        self.use_eb = use_eb
        self.eb_threshold = eb_threshold
        self.convergence = convergence

        self.nc_total = 0
        self.nc_n = 0

    def get_args(self, data, assignments):
        """Get Model Hyperparameters

        Returns
        -------
        dict
            Argument dictionary to be passed to core.init_model, with entries:
            "alpha": DPM mixing parameter
        """

        return {"alpha": float(self.alpha)}

    def __dp_update_lhs(self, alpha, N, K):
        """LHS of equation 8 (McAuliffe et. al, 2006)

        K = sum_{1<=n<=N} alpha / (alpha + n - 1)

        Parameters
        ----------
        alpha : float
            Mixing parameter; value being solved for
        N : int
            Number of iterations
        K : int
            Observed mean number of clusters

        Returns
        -------
        float
            Value of LHS of equation 8 (McAuliffe et. al, 2006)
        """

        return sum(alpha / (alpha + n) for n in range(N)) - K

    def update(self, mixture):
        """Run empirical bayes update (McAuliffe et. al, 2006)

        Parameters
        ----------
        mixture : MixtureModel Object
            Object to update alpha for

        Returns
        -------
        dict or None
            If EB enabled and past threshold, returns updated hyperparameters,
            with entries:
            "alpha": DPM mixing parameter, updated according to equation 8
                (McAuliffe et. al, 2006). Otherwise, returns None.
        """

        self.nc_total += np.max(mixture.assignments) + 1

        # Update estimate of K
        if mixture.iterations > self.eb_threshold and self.use_eb:

            # Compute alpha: sum_{1<=n<=N} alpha / (alpha + n - 1) = K
            self.alpha = optimize.newton(
                self.__dp_update_lhs, self.alpha,
                args=(mixture.iterations, self.nc_total / mixture.iterations),
                tol=self.convergence)

            return {"alpha": float(self.alpha)}

        else:
            return None
