
from bclust.core import MODEL_DPM, MODEL_MFM, COMPONENT_NORMAL_WISHART
import numpy as np
from scipy import optimize
import math


class NormalWishart:

    CAPSULE = COMPONENT_NORMAL_WISHART

    def __init__(self, df=2):

        self.df = df

    def get_args(self, data):

        return {
            "df": float(self.df),
            "s_chol": np.linalg.cholesky(np.cov(data.T))
        }

    def update(self, mixture):
        return None


class DPM:

    CAPSULE = MODEL_DPM

    def __init__(
            self, alpha=1,
            use_eb=True, eb_threshold=100, convergence=0.01):

        self.alpha = alpha
        self.use_eb = use_eb
        self.eb_threshold = eb_threshold
        self.convergence = convergence

        self.nc_total = 0
        self.nc_n = 0

    def get_args(self, data):

        return {"alpha": float(self.alpha)}

    def __dp_update_lhs(self, alpha, N, K):
        """LHS of equation 8 in "Nonparametric empirical Bayes for the
        Dirichlet process mixture model" (McAuliffe et. al., 2006)

        sum_{1<=n<=N} alpha / (alpha + n - 1)
        """

        return sum(alpha / (alpha + n) for n in range(N)) - K

    def update(self, mixture):

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


class MFM:

    CAPSULE = MODEL_MFM

    def __init__(
            self, gamma=1,
            prior=lambda k: k * math.log(0.1), error=0.001):

        self.gamma = gamma
        self.error = error
        self.prior = prior

    def log_v_n(self, N):

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
                    math.lgamma(k + 1) + math.lgamma(self.gamma * k) -
                    math.lgamma(k - t + 1) - math.lgamma(self.gamma * k + N)
                )
                current = np.logaddexp(current, term)
                if current - prev < self.error:
                    break
                k += 1
            res[t - 1] = current

        return res

    def get_args(self, data):

        return {
            "V_n": self.log_v_n(data.shape[0]),
            "gamma": float(self.gamma)
        }

    def update(self, mixture):
        return None
