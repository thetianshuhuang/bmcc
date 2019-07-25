"""
.
   _____ _____ _____ _____
  | __  |     |     |     |
  | __ -| | | |   --|   --|
  |_____|_|_|_|_____|_____|
  Bayesian Markov Chain Clustering

Author
------
Tianshu Huang
<thetianshuhuang@gmail.com>

Summary
-------
Implementation of Markov Chain Bayesian Clustering techniques, including DPM
(Dirichlet Process Mixture Models [1]) and MFM (Mixture of Finite Mixtures [2])
mixture models, with an abstract Mixture Model and Component Model API.

Hyperparameter updates for DPM are implemented using an Empirical Bayes update
procedure [3].

Final configuration selection is implemented using Least Squares clustering
[4].

For more information, see the Github repository:
https://github.com/thetianshuhuang/bmcc

References
----------
[1] Radford M. Neal (2000), "Markov Chain Sampling Methods for Dirichlet
    Process Mixture Models". Journal of Computational and Graphical Statistics,
    Vol. 9, No. 2.

[2] Jeffrey W. Miller, Matthew T. Harrison (2018),
    "Mixture Models with a Prior on the Number of Components".
    Journal of the American Statistical Association, Vol. 113, Issue 521.

[3] Jon D. McAuliffe, David M. Blei, Michael I. Jordan (2006),
    "Nonparametric empirical Bayes for the Dirichlet process mixture model".
    Statistics and Computing, Vol. 16, Issue 1.

[4] David B. Dahl (2006), "Model-Based Clustering for Expression Data via a
    Dirichlet Process Mixture Model". Bayesian Inference for Gene Expression
    and Proteomics.
"""


# -----------------------------------------------------------------------------
#
#                                   Imports
#
# -----------------------------------------------------------------------------

# -- C API --------------------------------------------------------------------

from bmcc.core import (
    # Model Capsules
    MODEL_DPM,
    MODEL_MFM,
    MODEL_HYBRID,
    COMPONENT_NORMAL_WISHART,
    COMPONENT_SYMMETRIC_NORMAL,

    # Functions
    pairwise_probability,
    aggregation_score,
    segregation_score,

    # Build Configuration
    BASE_VEC_SIZE,
    COMPONENT_METHODS_API,
    MODEL_METHODS_API,
    MIXTURE_MODEL_API,
    BUILD_DATETIME
)


# -- R Extension Utils --------------------------------------------------------

from bmcc.r_helpers import (
    is_uint16,
    is_float64,
    is_contiguous,
    is_np_array,
)


# -- Python Utilities ---------------------------------------------------------

from bmcc.models import MFM, DPM, Hybrid, NormalWishart, SymmetricNormal
from bmcc.mixture import GibbsMixtureModel
from bmcc.base_result import BaseResult
from bmcc.least_squares import LstsqResult, membership_matrix
from bmcc.plot import plot_clusterings
from bmcc.simulate import GaussianMixture


# -- Build Constants, API Constants, and Build Metadata -----------------------

CONFIG = {
    "BASE_VEC_SIZE": BASE_VEC_SIZE,
    "BUILD_DATETIME": BUILD_DATETIME,
    "COMPONENT_METHODS_API": COMPONENT_METHODS_API,
    "MODEL_METHODS_API": MODEL_METHODS_API,
    "MIXTURE_MODEL_API": MIXTURE_MODEL_API
}


# -- Package Metadata ---------------------------------------------------------

__author__ = "Tianshu Huang"
__license__ = "MIT"
__maintainer__ = "Tianshu Huang"
__email__ = "thetianshuhuang@gmail.com"


# -----------------------------------------------------------------------------
#
#                                   Exports
#
# -----------------------------------------------------------------------------

__all__ = [
    # Helpers
    "is_np_array",
    "is_uint16",
    "is_float64",
    "is_contiguous",

    # Capsules
    "MODEL_DPM",
    "MODEL_MFM",
    "MODEL_HYBRID",
    "COMPONENT_NORMAL_WISHART",
    "COMPONENT_SYMMETRIC_NORMAL",

    # Constants
    "CONFIG",

    # Utility
    "pairwise_probability",
    "aggregation_score",
    "segregation_score",

    # Models
    "MFM",
    "DPM",
    "Hybrid",
    "NormalWishart",
    "SymmetricNormal",

    # Core
    "GibbsMixtureModel",

    # Analysis
    "BaseResult",
    "LstsqResult",
    "membership_matrix",

    # Plots
    "plot_clusterings",

    # Simulation
    "GaussianMixture"
]
