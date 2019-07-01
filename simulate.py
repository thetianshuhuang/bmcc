
from scipy import stats
import numpy as np


def assign_balanced(samples, k=3):
    """Create balanced cluster assignment

    Parameters
    ----------
    samples : int
        Number of samples
    k : int
        Number of clusters (unique values in output)

    Returns
    -------
    int[]
        List of assignments
    """

    return [np.random.randint(0, k) for _ in range(samples)]


def __sample_weighted(weights):
    """Get weighted discrete RV sample"""

    rval = np.random.rand()
    for idx, value in enumerate(np.cumsum(weights / sum(weights))):
        if rval < value:
            return idx
    return -1


def assign_unbalanced(samples, k=3, r=0.8):
    """Create unbalanced cluster assignment with cluster weights proportional
    to r^i for cluster index i

    Parameters
    ----------
    samples : int
        Number of samples
    k : int
        Number of clusters
    r : float
        Balance parameter (must have r<1); smaller r = more unbalanced

    Returns
    -------
    int[]
        List of assignments
    """

    weights = [r**i for i in range(k)]

    return [__sample_weighted(weights) for _ in range(samples)]


def sample_means(alpha=2.0, d=2, k=3):
    """Sample cluster means

    Parameters
    ----------
    alpha : float
        Cluster density parameter. Variance of means is (alpha * k)^(1/d), so
        that when the number of clusters increases, the between-cluster
        variance increases to keep the cluster density constant
    d : int
        Number of dimensions
    k : int
        Number of clusters

    Returns
    -------
    np.array[]
        List of means
    """

    return [
        stats.multivariate_normal.rvs(
            mean=0, cov=np.identity * pow(alpha * k, 1 / d))
        for _ in range(k)
    ]


def sample_points_symmetric(cluster_centers, assignments, d=2):
    """Sample points with symmetric between-cluster variance

    Parameters
    ----------
    cluster_centers : np.array[]
        List of cluster centers. Index i corresponds to cluster i.
    assignments : int[]
        List of cluster assignments.
    d : int
        Dimensions
    """

    return [
        stats.multivariate_normal.rvs(
            mean=cluster_centers[i], cov=np.identity(d))
        for i in assignments
    ]


def sample_points_wishart(cluster_centers, assignments, d=2, k=3):
    """Sample points with within-cluster variances sampled from a wishart
    distribution.

    Parameters
    ----------
    cluster_centers : np.array[]
        List of cluster centers
    assignemnts : int[]
        List of cluster assignments
    d : int
        Dimensions
    k : int
        Number of clusters
    """

    wishart_params = [stats.wishart.rvs(d, np.identity(d)) for _ in range(k)]

    return [
        stats.multivariate_normal.rvs(
            mean=cluster_centers[i], cov=wishart_params[i])
        for i in assignments
    ]

