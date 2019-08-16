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
Implementation of Markov Chain Monte Carlo Bayesian Clustering techniques,
including DPM (Dirichlet Process Mixture Models; Neal, 2000) and MFM (Mixture
of Finite Mixtures; Miller & Harrison, 2018)) mixture models, with an abstrac
 Mixture Model and Component Model API.

Both Gibbs samplers Neal, 2000, Split Merge samplers Jain & Neal, 2012 are
implemented.

Hyperparameter updates for DPM are (optionally) implemented using an Empirical
Bayes update procedure (McAuliffe et. al., 2006).

Final configuration selection is implemented using Least Squares clustering
(Dahl, 2006).

For more information, see the Github repository:
https://github.com/thetianshuhuang/bmcc

References
----------
David B. Dahl (2006), "Model-Based Clustering for Expression Data via a
    Dirichlet Process Mixture Model". Bayesian Inference for Gene Expression
    and Proteomics.
Sonia Jain, Radford M. Neal (2004), "A Split-Merge Markov Chain Monte Carlo
    Procedure for the Dirichlet Process Mixture Model". Journal of
    Computational and Graphical Statistics, Vol 13, Issue 1.
Jon D. McAuliffe, David M. Blei, Michael I. Jordan (2006), "Nonparametric
    empirical Bayes for the Dirichlet process mixture model". Statistics and
    Computing, Vol. 16, Issue 1.
Jeffrey W. Miller, Matthew T. Harrison (2018), "Mixture Models with a Prior on
    the Number of Components". Journal of the American Statistical Association,
    Vol. 113, Issue 521.
Radford M. Neal (2000), "Markov Chain Sampling Methods for Dirichlet Process
    Mixture Models". Journal of Computational and Graphical Statistics, Vol. 9,
    No. 2.
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

    # MCMC functions
    gibbs,
    split_merge,

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

from bmcc.analysis import *
from bmcc.analysis import __all__ as __analysis_all
from bmcc.models import *
from bmcc.models import __all__ as __models_all

from bmcc.mixture import BayesianMixture
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

__all__ = __models_all + __analysis_all + [
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

    # Core
    "BayesianMixture",

    # Simulation
    "GaussianMixture"
]
