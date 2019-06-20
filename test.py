import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from crp import gibbs_sampling, crp_init, select_lstsq

data = np.concatenate((
    norm.rvs(0, 1, size=200),
    norm.rvs(10, 1, size=200),
    norm.rvs(12, 1, size=200)))

data = data.reshape((-1, 1))

# np.random.shuffle(data)

clusters = crp_init(data)
n = []
assign_hist = []
means = []
for i in range(20):
    clusters = gibbs_sampling(data, clusters, alpha=1, r=1)

    assignments = np.zeros(len(data))
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            assignments[point] = idx

    assign_hist.append(assignments)
    n.append(len(clusters))
    m = [np.mean([data[i] for i in cluster]) for cluster in clusters]
    print("[{}] : {} clusters : {}".format(i, len(clusters), m))
    means.append(m)

lstsq, mean, idx = select_lstsq(assign_hist)

print(lstsq)
print(means[idx])
print(idx)
plt.imshow(lstsq)
plt.imshow(mean)
plt.show()

"""
plt.subplot(1, 2, 1)
plt.hist(n[20:])
plt.subplot(1, 2, 2)
plt.hist(data, bins=25)
plt.show()
"""
