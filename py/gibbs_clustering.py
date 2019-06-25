"""Generic Gibbs Sampler for Generic Clustering Priors

Usage
-----
class CustomClustering(
        YourPriorMixins,
        YourMethodCoefficients,
        GibbsClustering):
    \"\"\"Clustering Object with the appropriate priors and method\"\"\"
    pass
"""

import numpy as np
import random

import cluster_util as util


class GibbsClustering:
    """Collapsed Gibbs Sampler for Generic Clustering Priors

    Parameters
    ----------
    data : np.array
        Data points
    clusters : set[]
        List of sets, with each set corresponding to a cluster
    assignments : np.array
        Assignment vector. Data type must be integer; defaults to np.uint16
    burn_in : int
        Burn in samples for gibbs sampler
    thinning : int
        Thinning ratio for gibbs sampler
    params : dict
        Hyperparameters and model priors

    Attributes
    ----------
    LIKELIHOOD_INIT : bool
        If True, calls _likelihood_init() on start to populate parameters
    COEFFICIENT_INIT : bool
        If True, calls _coefficient_init() on start
    """

    LIKELIHOOD_INIT = False
    COEFFICIENT_INIT = True

    def __init__(
            self, data,
            clusters=None, assignments=None,
            burn_in=0, thinning=1,
            params={}):

        # No cluster or assignments give -> initialize to all the same cluster
        if clusters is None or assignments is None:
            assignments = np.zeros(data.shape[0], dtype=np.uint16)
            clusters = [set(i for i in range(data.shape[0]))]

        # Check for wrong types
        if assignments.dtype != np.uint16:
            assignments = assignments.astype(np.uint16)

        # Check for wrong dimensions
        if assignments.shape[0] != data.shape[0]:
            raise TypeError(
                "Assignment vector must be the same size as data.shape[0].")

        # Save values
        self.data = data
        self.clusters = clusters
        self.assignments = assignments

        # Gibbs parameters
        self.burn_in = burn_in
        self.thinning = int(thinning)

        # Gibbs history
        self.iteration = 0
        self.history = []
        self.history_pairwise = []

        # Hyperparameters
        # Check parameters
        plist = self.REQUIRED_COEF_PARAMS + self.REQUIRED_LIKELIHOOD_PARAMS
        for p in plist:
            if p not in params:
                raise Exception(
                    "Missing required hyperparameter or prior: {}".format(p))
        self.params = params

        # Permutation private array (to avoid having to constantly recreate it)
        self.__permutation = [i for i in range(data.shape[0])]

        # Initializations
        if self.LIKELIHOOD_INIT:
            self._likelihood_init()
        if self.COEFFICIENT_INIT:
            self._coefficient_init()

    def __sample_proportional(self, weights):
        """Sample proportionally with weight vector

        Parameters
        ----------
        weights : array-like
            List of weights

        Returns
        -------
        int
            Sampled index, with indices weighted based on the weight vector
        """

        unif = random.random()

        for idx, value in enumerate(np.cumsum(weights / sum(weights))):
            if unif < value:
                return idx
        return -1

    def iter_n(self, n):
        """Run gibbs iteration n times"""
        for _ in range(n):
            self.iter()

    def iter(self):
        """Run gibbs iteration"""

        # Shuffle
        random.shuffle(self.__permutation)

        # Apply gibbs moves to each point
        for idx in self.__permutation:

            # Get clutser and remove from current
            current_cluster = self.assignments[idx]
            self.clusters[current_cluster].remove(idx)

            # Compute cluster probabilities
            new = self.__sample_proportional(
                [
                    self.l_cond(self.data, cluster, idx, self.params) *
                    self.coef_cond(self.data, cluster, idx, self.params)
                    for cluster in self.clusters if len(cluster) > 0
                ] + [
                    self.l_uncond(self.data, idx, self.params) *
                    self.coef_uncond(self.data, idx, self.params)
                ])

            # Current cluster
            if new < len(self.clusters):
                self.clusters[new].add(idx)
                self.assignments[idx] = new
            # New cluster
            else:
                # For-nobreak loop
                for c_idx, cluster in enumerate(self.clusters):
                    # Found empty -> reuse instead of creating new
                    if len(cluster) == 0:
                        cluster.add(idx)
                        self.assignments[idx] = c_idx
                        break
                # Nobreak -> create new cluster
                else:
                    self.clusters.append(set([idx]))
                    self.assignments[idx] = new

        # Save
        self.iteration += 1
        if(
                self.iteration > self.burn_in and
                (self.iteration % self.thinning == 0)):
            self.history.append(
                np.copy(self.assignments))
            self.history_pairwise.append(
                util.pairwise_matrix(self.assignments))

    def select_lstsq(self):
        """Select final clustering using least-squares strategy

        Returns
        -------
        [np.array[], np.array, np.array]
            [0] List of means. The index corresponds to the assignment number.
            [1] Assignment vector
            [2] Pairwise assignment matrix of 'final' clustering
                [i, j] = 1_{x_i and x_j are in the same cluster}
            [3] Pairwise probability matrix
                [i,j] = Pr{x_i, x_j are in the same cluster}
        """

        # Compute mean
        # Sum casts to int64, and divide casts to float64
        mean = (
            np.sum(self.history_pairwise, axis=0) /
            len(self.history_pairwise)
        )

        # Compute squared norm (element-wise)
        res = [
            np.linalg.norm(m - mean, ord='fro')
            for m in self.history_pairwise
        ]

        # Return least squares
        min_res = np.argmin(res)

        # Calculate clusters
        cluster_means = [
            (
                np.mean([self.data[i] for i in cluster], axis=0)
                if len(cluster) > 0 else np.nan
            ) for cluster in self.clusters
        ]

        return (
            cluster_means,
            self.history[min_res],
            self.history_pairwise[min_res],
            mean
        )

    #
    # -- Placeholder Likelihood and Coefficient Functions ---------------------
    #

    def l_cond(self, data, cluster, idx, params):
        """Conditional Likelihood function

        Parameters
        ----------
        data : np.array
            Data array
        cluster : set[]
            List of sets containing indices of points belonging to each cluster
        idx : int
            Index of point to compute likelihood for
        params : dict
            Model hyperparameters and prior parameters
        """
        raise Exception("Conditional likelihood function not defined.")

    def l_uncond(self, data, idx, params):
        """Unconditional Likelihood function

        Parameters
        ----------
        data : np.array
            Data array
        idx : int
            Index of point to compute likelihood for
        params : dict
            Model hyperparameters and prior parameters
        """
        raise Exception("Unconditional likelihood function not defined.")

    def coef_cond(self, data, cluster, idx, params):
        """Conditional Coefficient function

        Parameters
        ----------
        data : np.array
            Data array
        cluster : set[]
            List of sets containing indices of points belonging to each cluster
        idx : int
            Index of point to compute likelihood for
        params : dict
            Model hyperparameters and prior parameters
        """
        raise Exception("Conditional coefficient function not defined.")

    def coef_uncond(self, data, idx, params):
        """Unconditional Coefficient function

        Parameters
        ----------
        data : np.array
            Data array
        idx : int
            Index of point to compute likelihood for
        params : dict
            Model hyperparameters and prior parameters
        """
        raise Exception("Unconditional coefficient function not defined.")
