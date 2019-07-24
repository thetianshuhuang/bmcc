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

# Create dataset
dataset = bmcc.GaussianMixture(
    n=1000, k=3, d=3, r=0.7, alpha=40, df=3, symmetric=False, shuffle=False)

# Create mixture model
model = bmcc.GibbsMixtureModel(
    data=dataset.data,
    component_model=bmcc.NormalWishart(df=3),
    mixture_model=bmcc.MFM(
        gamma=1, prior=lambda k: poisson.logpmf(k, 4)),
    assignments=np.zeros(1000).astype(np.uint16),
    thinning=5)

# Run Iterations
start = time.time()
for i in tqdm(range(5000)):
    model.iter()
print("gibbs_iterate: {:.2f}s [{:.2f} ms/iteration]".format(
    time.time() - start, (time.time() - start) * 1000 / 5000))

# Select Least Squares clustering
start = time.time()
res = model.select_lstsq(burn_in=100)
res.evaluate(
    dataset.assignments,
    oracle=dataset.oracle,
    oracle_matrix=dataset.oracle_matrix)
print("evaluate_lstsq: {:.2f}s".format(time.time() - start))
print("num_clusters: {}".format(res.num_clusters[res.best_idx]))

# Plot
res.trace(plot=True)
res.matrices(plot=True)
res.clustering(kwargs_scatter={"marker": "."}, plot=True)
dataset.plot_oracle(kwargs_scatter={"marker": "."}, plot=True)
