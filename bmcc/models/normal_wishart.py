"""Multivariate Normal Component Distribution with Wishart-distributed
Covariance Matrix

References
----------
[1] Jeffrey W. Miller, Matthew T. Harrison (2018),
    "Mixture Models with a Prior on the Number of Components".
    Journal of the American Statistical Association, Vol. 113, Issue 521.
"""

import numpy as np

from bmcc.core import COMPONENT_NORMAL_WISHART


class NormalWishart:
    """Normal Wishart Component type

    Parameters
    ----------
    df : int
        Degrees of freedom

    Attributes
    ----------
    CAPSULE : capsule
        Capsule containing component methods (export from C module)
    """

    CAPSULE = COMPONENT_NORMAL_WISHART

    def __init__(self, df=2):

        self.df = df

    def get_args(self, data):
        """Get component hyperparameters

        Parameters
        ----------
        data : np.array
            Dataset; used for scale matrix S = 1/df * Cov(data) [1]

        Returns
        -------
        dict
            Argument dictionary to be passed to core.init_model, with keys:
            "df": Wishart degrees of freedom
            "s_chol": Cholesky decomposition of scale matrix

        """

        return {
            "df": float(self.df),
            "s_chol": np.linalg.cholesky(
                1 / self.df * np.linalg.inv(np.cov(data.T)))
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
            No updates for normal wishart collapsed gibbs sampler
        """
        return None
