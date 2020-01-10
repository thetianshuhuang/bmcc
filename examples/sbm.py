from matplotlib import pyplot as plt
import bmcc
import numpy as np
from tqdm import tqdm
from scipy.stats import poisson
import time


N = 200
ITERATIONS = 100

ds = bmcc.StochasticBlockModel(
    n=N, k=3, r=1, alpha=0.8, beta=1, shuffle=False)
print(ds.Q)

# plt.matshow(ds.data)
# plt.show()

start = time.time()

model = bmcc.BayesianMixture(
    data=ds.data,
    sampler=bmcc.gibbs,
    component_model=bmcc.SBM(
        alpha=1, beta=1, Q=np.ones((1, 1))),
    # mixture_model=bmcc.MFM(gamma=1, prior=lambda k: poisson.logpmf(k, 2)),
    mixture_model=bmcc.DPM(alpha=1),
    assignments=np.zeros(N).astype(np.uint16),
    thinning=1)

for _ in tqdm(range(ITERATIONS)):
    model.iter()

print(time.time() - start)


res = model.select_lstsq(burn_in=50)

fig, (left, right) = plt.subplots(1, 2)
left.matshow(ds.data)
right.matshow(res.matrix)
plt.show()
0