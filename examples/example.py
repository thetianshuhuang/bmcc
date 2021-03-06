"""Example usage of bmcc.

This example simulates a mixture of gaussian with 4 clusters, 3 dimensions,
unbalanced (r=0.7), and asymmetric (Wishart-distributed normal with df=3).

Gibbs Sampling is then run with a MFM prior.
"""


import numpy as np
from scipy.stats import poisson
import time
from tqdm import tqdm
import bmcc

# Settings
ITERATIONS = 500
THINNING = 5
POINTS = 500
BURN_IN = 50

# Create dataset
dataset = bmcc.GaussianMixture(
    n=POINTS, k=3, d=3, r=0.7, alpha=40, df=3, symmetric=False, shuffle=False)


def hybrid(*args, **kwargs):

    for _ in range(5):
        bmcc.gibbs(*args, **kwargs)
    bmcc.split_merge(*args, **kwargs)


mm = bmcc.MFM(gamma=1, prior=lambda k: poisson.logpmf(k, 3))
# mm = bmcc.DPM(alpha=1, use_eb=False)
cm = bmcc.NormalWishart(df=3)

# Create mixture model
model = bmcc.BayesianMixture(
    data=dataset.data,
    sampler=hybrid,
    component_model=cm,
    mixture_model=mm,
    assignments=np.zeros(POINTS).astype(np.uint16),
    thinning=THINNING)

# Run Iterations
start = time.time()
for i in tqdm(range(ITERATIONS)):
    model.iter()
print("gibbs_iterate: {:.2f}s [{:.2f} ms/iteration]".format(
    time.time() - start, (time.time() - start) * 1000 / ITERATIONS))

# Select Least Squares clustering
start = time.time()
res = model.select_lstsq(burn_in=BURN_IN)
res.evaluate(
    dataset.assignments,
    oracle=dataset.oracle,
    oracle_matrix=dataset.oracle_matrix)
print("evaluate_lstsq: {:.2f}s".format(time.time() - start))
print("num_clusters: {}".format(res.num_clusters[res.best_idx]))

# Plot
# res.trace(plot=True)
# res.matrices(plot=True)
res.clustering(kwargs_scatter={"marker": "."}, plot=True)

# bmcc.cleanup_maximum_likelihood(dataset.data, res.best, cm, mm)
# res.clustering(kwargs_scatter={"marker": "."}, plot=True)

# dataset.plot_oracle(kwargs_scatter={"marker": "."}, plot=True)
