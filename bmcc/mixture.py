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
    init_model, gibbs_iter,
    update_mixture,
    update_components)
from bmcc.least_squares import LstsqResult
from bmcc.models import NormalWishart, DPM
from bmcc.errors import (
    WARNING_CONTIGUOUS_CAST,
    WARNING_FLOAT64_CAST,
    WARNING_UINT16_CAST
)


class GibbsMixtureModel:
    """Gibbs Sampler for a Mixture Model

    Parameters
    ----------
    data : np.array
        Data points; row-major (each row is a data point)
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
    # -- Check Types ----------------------------------------------------------
    #

    def __check_data(self, data):
        """Check data array type."""

        # Check types
        if type(data) != np.ndarray:
            raise TypeError(
                "Data must be a numpy array.")
        if len(data.shape) != 2:
            raise TypeError(
                "Data must have 2 dimensions. The points should be stored in "
                "row-major order (each data point is a row).")
        if data.dtype != np.float64:
            print(WARNING_FLOAT64_CAST)
            data = data.astype(np.float64)
        if not data.flags['C_CONTIGUOUS']:
            print(WARNING_CONTIGUOUS_CAST)
            data = np.ascontiguousarray(data, dtype=np.float64)
        return data

    def __check_assignments(self, assignments, size):
        """Check assignment array type."""

        # Check assignments type
        if type(assignments) != np.ndarray:
            raise TypeError("Assignments must be an array.")
        if len(assignments.shape) != 1:
            raise TypeError("Assignments must be one-dimensional.")
        if assignments.shape[0] != size:
            raise TypeError(
                "Assignments must have the same dimensionality as the number "
                "of data points.")
        if assignments.dtype != np.uint16:
            print(WARNING_UINT16_CAST)
            assignments = assignments.astype(np.uint16)

        return assignments

    def __check_capsules(self, cmodel, mmodel):
        """Check for correct capsule types."""

        # Check for capsule
        if not hasattr(cmodel, "CAPSULE"):
            raise TypeError(
                "Component Model must have 'CAPSULE' attribute (containing C "
                "functions describing model methods)")
        if not hasattr(mmodel, "CAPSULE"):
            raise TypeError(
                "Mixture Model must have 'CAPSULE' attribute (containing C "
                "functions describing model methods)")

    #
    # -- Initialization -------------------------------------------------------
    #

    def __init__(
            self, data,
            component_model=None,
            mixture_model=None,
            assignments=None,
            thinning=1):

        # Check data first
        self.data = self.__check_data(data)

        # Provide models
        if component_model is None:
            print(
                "No component model provided; using Normal Wishart with "
                "df=dim.")
            component_model = NormalWishart(df=self.data.shape[1])
        if mixture_model is None:
            print(
                "No mixture model provided; using DPM with initial alpha = 1.")
            mixture_model = DPM(alpha=1)
        if assignments is None:
            print(
                "No initial assignments provided. Assigning all points to the "
                "same cluster at initialization.")
            assignments = np.zeros(data.shape[0], dtype=np.uint16)

        # Save models
        self.component_model = component_model
        self.mixture_model = mixture_model

        # Check assignments now that we have data dimensions
        self.assignments = self.__check_assignments(assignments, data.shape[0])

        # Create args dict
        params = {"dim": data.shape[1]}
        try:
            params.update(component_model.get_args(data))
            params.update(mixture_model.get_args(data))
        except AttributeError:
            raise TypeError(
                "Component Model and Mixture Model must have 'get_args' "
                "attribute (used to fetch dictionary args for capsule "
                "initializer)")

        # Make sure capsules are present
        self.__check_capsules(component_model, mixture_model)

        # Create model
        self.__model = init_model(
            self.data, self.assignments,
            component_model.CAPSULE, mixture_model.CAPSULE, params)

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

            gibbs_iter(self.data, self.assignments, self.__model)

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
