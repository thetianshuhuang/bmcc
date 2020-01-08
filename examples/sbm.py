from matplotlib import pyplot as plt
import bmcc
import numpy as np
from tqdm import tqdm


ds = bmcc.StochasticBlockModel(
    n=100, k=3, r=1, alpha=0.5, beta=2, shuffle=False)
print(ds.Q)

# plt.matshow(ds.data)
# plt.show()

model = bmcc.BayesianMixture(
    data=ds.data,
    sampler=bmcc.gibbs,
    component_model=bmcc.SBM(
        alpha=1, beta=1, Q=np.ones((1, 1))),
    mixture_model=bmcc.DPM(alpha=1, use_eb=False),
    assignments=np.zeros(100).astype(np.uint16),
    thinning=1)

for _ in tqdm(range(100)):
    model.iter()


res = model.select_lstsq(burn_in=0)

fig, (left, right) = plt.subplots(1, 2)
left.matshow(ds.data)
right.matshow(res.matrix)
plt.show()
