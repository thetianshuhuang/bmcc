import numpy as np
import random
from scipy import stats


def gaussian_mixture_likelihood(data, cluster, idx):
    """Conditional likelihood for gaussian mixture model with
    all other points and cluster assignments fixed

    Parameters
    ----------
    data : np.array
        Source data
    cluster : set
        Set of indices in the target cluster
    idx : int
        Index of current pivot point

    Returns
    -------
    float
        Computed likelihood
    """

    # Fetch points associated with the cluster
    cluster_data = np.array([data[i] for i in cluster if i != idx])

    # Get mu, sigma
    # We assume sigma > 1 in order to avoid situations where n=1 -> sigma=0
    sigma_k = max(np.sqrt(np.var(cluster_data, axis=0)), 1)
    mu_k = np.mean(cluster_data, axis=0)
    # p(x_i | X_{k, !=i}, beta) = N(x_i; mu_k, sigma_k)
    return stats.multivariate_normal.pdf(
        data[idx], mean=mu_k, cov=sigma_k * np.identity(data.shape[1]))


def gaussian_likelihood(data, idx, sigma_mu=1, sigma_x=1):
    """Example conditional likelihood for gaussian mixture model with normally
    distributed means, where all gaussians are symmetric

    Parameters
    ----------
    data : np.array
        Source data
    idx : int
        Index of current pivot point
    sigma_mu : float
        Standard deviation of cluster means
    sigma_x : float
        Standard deviation of points within cluster

    Returns
    -------
    float
        Computed likelihood
    """

    return stats.multivariate_normal.pdf(
        data[idx], mean=6,
        cov=np.sqrt(sigma_mu**2 + sigma_x**2) * np.identity(data.shape[1]))


def _get_cluster(idx, clusters):

    for i, cluster in enumerate(clusters):
        if idx in cluster:
            return i
    return -1


def _get_p_cluster(
        data, clusters, idx, alpha=1, r=1,
        likelihood_conditional=gaussian_mixture_likelihood,
        likelihood_unconditional=gaussian_likelihood):
    return [
        likelihood_conditional(data, cluster, idx) * len(cluster)**r
        for cluster in clusters if len(cluster) > 0
    ] + [likelihood_unconditional(data, idx) * alpha]


def crp_init(
        data, r=1, alpha=1,
        likelihood_conditional=gaussian_mixture_likelihood,
        likelihood_unconditional=gaussian_likelihood):
    """Generate initial clusterings based on Chinese Restaurant Process prior

    Parameters
    ----------
    data : np.array
        Source data
    r : float
        pCRP power penalty
    alpha : float
        CRP hyperparameter
    likelihood_conditional : f(data, cluster, idx) -> float
        Conditional likelihood function p(z_i = k | z_{!=i}, alpha)
    likelihood_unconditional : f(data, idx) -> float
        Unconditional likelihood function p(x_i | beta)

    Returns
    -------
    set[]

    """

    clusters = []
    for idx in range(data.shape[0]):
        new = sample_proportional(
            _get_p_cluster(
                data, clusters, idx, alpha=alpha, r=r,
                likelihood_conditional=likelihood_conditional,
                likelihood_unconditional=likelihood_unconditional))
        try:
            clusters[new].add(idx)
        except IndexError:
            clusters.append(set([idx]))

    return clusters


def gibbs_sampling(
        data, clusters,
        r=1, alpha=1,
        likelihood_conditional=gaussian_mixture_likelihood,
        likelihood_unconditional=gaussian_likelihood):
    """Run gibbs sampling iteration

    Parameters
    ----------
    data : np.array
        Source data
    clusters : set[]
        List of sets containing indices in each cluster.
    r : float
        pCRP power penalty
    alpha : float
        CRP hyperparameter
    likelihood_conditional : f(data, cluster, idx) -> float
        Conditional likelihood function p(z_i = k | z_{!=i}, alpha)
    likelihood_unconditional : f(data, idx) -> float
        Unconditional likelihood function p(x_i | beta)

    Returns
    -------
    set[]
        Updated list of cluster assignments
    """

    # Shuffle points
    permutation = [i for i in range(data.shape[0])]
    random.shuffle(permutation)
    for idx in permutation:

        # Remove from current cluster
        current = _get_cluster(idx, clusters)
        clusters[current].remove(idx)

        # Compute cluster probabilities
        # Sample and add
        new = sample_proportional(
            _get_p_cluster(
                data, clusters, idx, alpha=alpha, r=r,
                likelihood_conditional=likelihood_conditional,
                likelihood_unconditional=likelihood_unconditional))
        try:
            clusters[new].add(idx)
        except IndexError:
            clusters.append(set([idx]))

        # Remove empty clusters
        clusters = [cluster for cluster in clusters if len(cluster) > 0]

    return clusters


def sample_proportional(weights):
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


def select_lstsq(assignments):

    pmatrix = []
    res = []

    for x in assignments:
        pmatrix_new = np.zeros((x.shape[0], x.shape[0]))
        for i, x_i in enumerate(x):
            for j, x_j in enumerate(x):
                pmatrix_new[i][j] = 1 if x_i == x_j else 0

        pmatrix.append(pmatrix_new)

    mean = sum(pmatrix) / len(pmatrix)

    for m in pmatrix:
        res.append(np.linalg.norm(np.reshape(m - mean, -1)))

    min_res = np.argmin(res)

    return pmatrix[min_res], mean, min_res
