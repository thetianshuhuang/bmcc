import numpy as np
import time
from tqdm import tqdm
# import math

import bmcc

ITERATIONS = 5000

start = time.time()
dataset = bmcc.GaussianMixture(
    n=1000, k=4, d=3, r=0.7, alpha=10, df=3, symmetric=False, shuffle=False)
print("simulate: {:.2f}s".format(time.time() - start))

model = bmcc.GibbsMixtureModel(
    data=dataset.data,
    component_model=bmcc.NormalWishart(df=3),
    mixture_model=bmcc.DPM(alpha=1, use_eb=True),
    # mixture_model=bmcc.MFM(gamma=1, prior=lambda k: k * math.log(0.8)),
    assignments=np.zeros(1000).astype(np.uint16),
    thinning=5)

start = time.time()
for i in tqdm(range(ITERATIONS)):
    model.iter()
print("gibbs_iterate: {:.2f}s [{:.2f} ms/iteration]".format(
    time.time() - start,
    (time.time() - start) * 1000 / ITERATIONS))

start = time.time()
res = model.select_lstsq(burn_in=100)
res.evaluate(
    dataset.assignments,
    oracle=dataset.oracle,
    oracle_matrix=dataset.oracle_matrix)
print("evaluate_lstsq: {:.2f}s".format(time.time() - start))

print("num_clusters: {}".format(res.num_clusters[res.best_idx]))

res.trace(plot=True)
res.matrices(plot=True)
res.clustering(kwargs_scatter={"marker": "."}, plot=True)
