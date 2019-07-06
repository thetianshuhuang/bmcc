import bayesian_clustering_c as bc
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from matplotlib import pyplot as plt
import time

from tqdm import tqdm

import faulthandler
faulthandler.enable()

SCALE = 100
ITERATIONS = 500

data = np.concatenate((
    mvnorm.rvs(
        mean=np.array([-3, -3]),
        cov=np.identity(2), size=5 * SCALE),
    mvnorm.rvs(
        mean=np.array([3, 0]),
        cov=np.identity(2), size=2 * SCALE),
    mvnorm.rvs(
        mean=np.array([-3, 3]),
        cov=np.identity(2), size=3 * SCALE))).astype(np.float64)
asn = np.zeros(10 * SCALE).astype(np.uint16)

mm = bc.init_model(
    data, asn,
    bc.COMPONENT_NORMAL_WISHART,
    bc.MODEL_DPM,
    {
        "alpha": 0.1,
        "df": 2.,
        "dim": 2,
        "s_chol": np.identity(2).astype(np.float64),
    })

start = time.time()
hist = np.zeros((ITERATIONS, 10 * SCALE), dtype=np.uint16)
for i in tqdm(range(ITERATIONS)):
    bc.gibbs_iter(data, asn, mm)
    hist[i, :] = asn
print("{:.2f} ms/iteration".format((time.time() - start) * 1000 / ITERATIONS))

pm, res = bc.pairwise_probability(hist, 100)
best = np.argmin(res)


fig, (left, right) = plt.subplots(1, 2)

left.matshow(pm)
right.scatter(data[:, 0], data[:, 1], c=hist[best, :])
plt.show()

# trace = [np.max(x) for x in hist]

# plt.plot(trace)
# plt.show()
