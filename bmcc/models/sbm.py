"""Stochastic Block Model

References
----------
Junxian Geng, Anirban Bhattacharya, Depdeep Pati (2018),
    "Probabilistic community detection with unknown number of communities".
    Journal of the American Statistical Association, Vol. 114, Issue 526.
"""

import numpy as np
from bmcc.core import COMPONENT_STOCHASTIC_BLOCK_MODEL, sbm_update


class SBM:
    """Stochastic Block Model Component Type

    Keyword Args
    ------------
    a: float
        Beta prior a parameter
    b : float
        Beta prior b parameter

    Attributes
    ----------
    CAPSULE : capsule
        Capsule containing component methods (export from C module)
    DATA_TYPE : type
        Numpy type for this model's input data
    DATA_TYPE_NAME : str
        Human readable data type name
    """

    DATA_TYPE = np.uint8
    DATA_TYPE_NAME = "uint8"

    CAPSULE = COMPONENT_STOCHASTIC_BLOCK_MODEL

    def __init__(self, a=1, b=1):
        self.a = float(a)
        self.b = float(b)

    def get_args(self, data, assignments):
        """Get component hyperparameters

        Parameters
        ----------
        data : np.array
            Input dataset; linked to struct sbm_params_t
        assignments : np.array
            Input assignments; linked to struct sbm_params_t

        Returns
        -------
        dict
            Argument dictionary to be passed to core.init_model, with keys:
            "a", "b": Beta prior parameter
            "asn": assignments array (reference)
            "data": data array (reference)
        """

        return {
            "a": self.a,
            "b": self.b,
            "asn": assignments,
            "data": data
        }

    def update(self, mixture):
        """Run hyperparameter update

        Parameters
        ----------
        mixture : MixtureModel object
            Object to update for

        Returns
        -------
        dict
            Update dictionary, with keys:
            "Q": new resampled Q matrix.
        """

        return {
            "Q": sbm_update(
                mixture.data, mixture.assignments,
                mixture.num_clusters, self.a, self.a)
        }
