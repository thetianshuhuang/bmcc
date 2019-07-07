import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

import bclust

import faulthandler
faulthandler.enable()


SCALE = 100
ITERATIONS = 100

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
actual = np.concatenate((
    np.zeros(5 * SCALE),
    np.ones(2 * SCALE),
    np.ones(3 * SCALE) * 2))

test = bclust.GibbsMixtureModel(
    data=data,
    component_model=bclust.NormalWishart(),
    mixture_model=bclust.DPM(alpha=1),
    assignments=np.zeros(10 * SCALE))

start = time.time()
for i in tqdm(range(ITERATIONS)):
    test.iter()

res = test.select_lstsq(burn_in=10)
res.evaluate(actual)

res.trace()
res.pairwise()
res.clustering()
