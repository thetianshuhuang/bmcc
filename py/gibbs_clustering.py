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
    """

    def __init__(
            self, data,
            clusters=None, assignments=None,
            burn_in=0, thinning=1,
            params={}):

        # No cluster or assignments give -> initialize to all the same cluster
        if clusters is None or assignments is None:
            assignments = np.zeros(data.shape[0], dtype=np.uint16)
            clusters = [set(i for i in range(data.shape[0]))]

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

        # Hyperparameters
        self.params = params

        # Permutation private array (to avoid having to constantly recreate it)
        self.__permutation = [i for i in range(data.shape[0])]

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
                self.clusters.append(set([idx]))
                self.assignments[idx] = new

        # Save
        self.iteration += 1
        if(
                self.iteration > self.burn_in and
                (self.iteration % self.thinning == 0)):
            self.history.append(np.copy(self.assignments))

    def select_lstsq(self):
        """Select final clustering using least-squares strategy

        Returns
        -------
        """

        pmatrix = []
        res = []

        # Compute probability matrices
        for x in self.history:
            pmatrix_new = np.zeros((x.shape[0], x.shape[0]))
            for i, x_i in enumerate(x):
                for j, x_j in enumerate(x):
                    pmatrix_new[i][j] = 1 if x_i == x_j else 0

            pmatrix.append(pmatrix_new)

        # Compute mean
        mean = sum(pmatrix) / len(pmatrix)

        # Compute squared norm (element-wise)
        for m in pmatrix:
            res.append(np.linalg.norm(np.reshape(m - mean, -1)))

        # Return least squares
        min_res = np.argmin(res)
        return pmatrix[min_res], mean, min_res

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
