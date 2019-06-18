import numpy as np
from scipy.stats import norm
# from matplotlib import pyplot as plt

from crp import gibbs_sampling, crp_init

data = np.concatenate((
    norm.rvs(0, 1, size=200),
    norm.rvs(10, 1, size=200),
    norm.rvs(12, 1, size=200)))

data = data.reshape((-1, 1))

np.random.shuffle(data)

clusters = crp_init(data)
n = []
means = []
for i in range(50):
    clusters = gibbs_sampling(data, clusters, alpha=1, r=1)

    n.append(len(clusters))
    m = [np.mean([data[i] for i in cluster]) for cluster in clusters]
    print("[{}] : {} clusters : {}".format(i, len(clusters), m))
    means.append(m)

"""
plt.subplot(1, 2, 1)
plt.hist(n[20:])
plt.subplot(1, 2, 2)
plt.hist(data, bins=25)
plt.show()
"""
