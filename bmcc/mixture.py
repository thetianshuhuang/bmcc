"""Gibbs Sampler for Abstract Mixture Model

Implements a generalized form of Algorithm 3 [1], with support for traditional
DPM and MFM [2].

Notes
-----
Use by providing a abstract Mixture model and a Component Model.

Both should have:
- a 'get_args' method, which takes in the data matrix and returns a dictionary
  containing hyperparameters, which will be used later
- a 'update' method, which takes in a MixtureModel object
  (like GibbsMixtureModel below) and returns either an updated hyperparameter
  dictionary (like the 'get_args' method), or None (no update).
- a 'CAPSULE' attribute, containing the C methods.

The capsule should contain either a 'ModelMethods' or a 'ComponentMethods'
struct. See the header files for more details. These types can be accessed
by copying mixture.h and #including it.
- Make sure that the capsules are created with the names
  'bmcc.core.ModelMethods' (defined by MODEL_METHODS_API) and
  'bmcc.core.ComponentMethods' (COMPONENT_METHODS_API), respectively.

References
----------
[1] Radford M. Neal (2000), "Markov Chain Sampling Methods for Dirichlet
    Process Mixture Models". Journal of Computational and Graphical Statistics,
    Vol. 9, No. 2.
[2] Jeffrey W. Miller, Matthew T. Harrison (2018),
    "Mixture Models with a Prior on the Number of Components".
    Journal of the American Statistical Association, Vol. 113, Issue 521.
"""

import numpy as np

from bmcc.core import (
    init_model, gibbs,
    update_mixture,
    update_components)
from .analysis import LstsqResult
from .util.type_check import (
    check_data,
    check_assignments,
    check_mixture_model,
    check_component_model
)
from .util.get_params import get_params


class BayesianMixture:
    """Abstract MCMC Sampler for a Mixture Model

    Parameters
    ----------
    data : np.array
        Data points; row-major (each row is a data point)

    Keyword Args
    ------------
    sampler : function
        Sampler function (such as ``bmcc.core.gibbs``)
    component_model : Object
        Component object (NormalWishart, etc)
    mixture_model : Object
        Mixture object (DPM, MFM, etc)
    assignments : np.array
        Assignment vector. If None (not passed), is initialized at all 0s.
    thinning : int
        Thinning factor (only saves one sample every <thinning> iterations).
        Use thinning=1 for no thinning.
    """

    __BASE_HIST_SIZE = 32

    #
    # -- Initialization -------------------------------------------------------
    #

    def __init__(
            self, data,
            sampler=gibbs,
            component_model=None,
            mixture_model=None,
            assignments=None,
            thinning=1):

        # Run type checks
        self.data = check_data(data)
        self.assignments = check_assignments(assignments, data.shape[0])
        self.mixture_model = check_mixture_model(mixture_model)
        self.component_model = check_component_model(
            component_model, data.shape[1])

        # Save sampler
        self.sampler = sampler

        # Create model
        self.__model = init_model(
            self.data,
            self.assignments,
            self.component_model.CAPSULE,
            self.mixture_model.CAPSULE,
            get_params(data, self.component_model, self.mixture_model))

        # Set up thinning
        if type(thinning) != int:
            raise TypeError("Thinning factor must be an integer.")
        self.thinning = thinning
        self.iterations = 0

        # Initialize history
        self.__history = np.zeros(
            (self.__BASE_HIST_SIZE, data.shape[0]), dtype=np.uint16)
        self.__hist_size = self.__BASE_HIST_SIZE
        self.__hist_idx = 0

    #
    # -- Iteration & Results --------------------------------------------------
    #

    def iter(self, iterations=1):
        """Run gibbs sampler for one iteration. If the iteration index is
        divisible by the thinning factor, save the assignments.

        Parameters
        ----------
        iterations : int
            Number of iterations to run
        """

        for _ in range(iterations):

            self.sampler(self.data, self.assignments, self.__model)

            comp_update = self.component_model.update(self)
            if comp_update is not None:
                update_components(self.__model, comp_update)

            mix_update = self.mixture_model.update(self)
            if mix_update is not None:
                update_mixture(self.__model, mix_update)

            self.iterations += 1

            # Save?
            if self.iterations % self.thinning == 0:
                self.__history[self.__hist_idx, :] = self.assignments
                self.__hist_idx += 1

            # Resize -- exponential over-allocation
            if self.__hist_idx >= self.__hist_size:
                self.__hist_size *= 2
                self.__history.resize(
                    (self.__hist_size, self.data.shape[0]), refcheck=False)

    @property
    def hist(self):
        """Get the valid slice of the history array.

        Returns
        -------
        np.array
            View of history array containing valid entries
        """
        return self.__history[:self.__hist_idx, :]

    def select_lstsq(self, burn_in=0):
        """Do least-squares type analysis on gibbs sampling results.

        Parameters
        ----------
        burn_in : int
            Burn in duration; defaults to 0 (no burn in).

        Returns
        -------
        LstsqResult object
            Least squares result based on history, ignoring burn-in duration
        """

        return LstsqResult(self.data, self.hist, burn_in=burn_in)
