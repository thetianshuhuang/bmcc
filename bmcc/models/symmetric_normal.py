"""Symmetric Normal Component Distribution

This can be used to 'fairly' compare to algorithms implicitly assuming
symmetric equal-variance clusters such as K-Means or SLINK (or other naive
hierarchical schemes)
"""

import numpy as np

from bmcc.core import COMPONENT_SYMMETRIC_NORMAL


class SymmetricNormal:
    """Symmetric Normal Component Type

    Parameters
    ----------
    scale : float
        Normal component scale

    Attributes
    ----------
    CAPSULE : capsule
        Capsule containing component methods (export from C module)
    DATA_TYPE : type
        Numpy type for this model's input data
    DATA_TYPE_NAME : str
        Human readable data type name
    """

    DATA_TYPE = np.float64
    DATA_TYPE_NAME = "float64"

    CAPSULE = COMPONENT_SYMMETRIC_NORMAL

    def __init__(self, scale=1.0):

        self.scale = float(scale)

    def get_args(self, data, assignments):
        """Get component hyperparameters

        Parameters
        ----------
        data : np.array
            Dataset; used for estimate of overall variance

        Returns
        -------
        dict
            Argument dictionary to be passed to core.init_model, with keys:
            "scale": Scale of symmetric normal components
            "scale_all": Overall scale of all points. Uses ||Cov||_2 as an
                dimension-invariant 'compression' of the overal covariance
                matrix to a symmetric normal.
        """
        return {
            "scale": self.scale,
            "scale_all": np.linalg.norm(np.cov(data), ord=2)
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
            No updates for symmetric normal distribution
        """
        return None
