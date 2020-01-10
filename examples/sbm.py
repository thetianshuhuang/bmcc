from matplotlib import pyplot as plt
import bmcc
import numpy as np
from tqdm import tqdm
from scipy.stats import poisson
import time


N = 200
ITERATIONS = 2000
K = 3
Q = np.identity(K) * 0.2 + np.ones((K, K)) * 0.1

ds = bmcc.StochasticBlockModel(
    n=N, k=K, r=1, a=0.8, b=1, shuffle=False, Q=Q)
print(ds.Q)

# plt.matshow(ds.data)
# plt.show()

start = time.time()

model = bmcc.BayesianMixture(
    data=ds.data,
    sampler=bmcc.gibbs,
    component_model=bmcc.SBM(a=1, b=1),
    mixture_model=bmcc.MFM(gamma=1, prior=lambda k: poisson.logpmf(k, K)),
    # mixture_model=bmcc.DPM(alpha=1),
    assignments=np.zeros(N).astype(np.uint16),
    thinning=1)

for _ in tqdm(range(ITERATIONS)):
    model.iter()

print(time.time() - start)


res = model.select_lstsq(burn_in=1500)

fig, axs = plt.subplots(2, 2)
axs[0][0].matshow(ds.data)
axs[0][1].matshow(res.matrix)
axs[1][0].plot(res.num_clusters)
axs[1][1].matshow(bmcc.membership_matrix(res.best))
plt.show()
