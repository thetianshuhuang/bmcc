import numpy as np
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import math

import bmcc

import faulthandler
faulthandler.enable()


SCALE = 100
ITERATIONS = 2500


dataset = bmcc.GaussianMixture(
    n=1000, k=4, d=3, r=0.7, alpha=100, df=3, symmetric=False, shuffle=False)

# dataset.plot_actual()
# plt.show()

# dataset.plot_oracle()
# plt.show()

test = bmcc.GibbsMixtureModel(
    data=dataset.data,
    component_model=bmcc.NormalWishart(df=3),
    # mixture_model=bmcc.DPM(alpha=1, use_eb=True),
    mixture_model=bmcc.MFM(gamma=1, prior=lambda k: k * math.log(1 / 4)),
    assignments=np.zeros(10 * SCALE).astype(np.uint16),
    thinning=5)

start = time.time()
for i in tqdm(range(ITERATIONS)):
    test.iter()
print("gibbs_iterate: {:.2f} s [{:.2f} ms/iteration]".format(
    time.time() - start,
    (time.time() - start) * 1000 / ITERATIONS))

start = time.time()
res = test.select_lstsq(burn_in=50)
res.evaluate(dataset.assignments, oracle=dataset.oracle)
print("evaluate_lstsq: {:.2f} s".format(time.time() - start))

res.trace()
plt.show()
res.matrices()
plt.show()
res.clustering()
plt.show()
dataset.plot_oracle()
plt.show()
