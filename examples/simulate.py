"""Create simulated dataset.

This example creates 3840 (192 * 20) simulated datasets, with 192 different
sets of simulation parameters. The ressults are saved to
'./results/<params>/<uuid>.npz'.
"""

import bmcc
import os
import uuid
from tqdm import tqdm
from multiprocessing import Pool


def make(args):

    n, d, k, r, sym = args

    path = "data/r={},sym={}/n={},d={},k={}".format(r, sym, n, d, k)
    if not os.path.exists(path):
        os.makedirs(path)

    for _ in range(20):
        bmcc.GaussianMixture(
            n=n, k=k, d=d, r=r, alpha=40, df=d,
            symmetric=sym, shuffle=False
        ).save(os.path.join(path, str(uuid.uuid1())))


if __name__ == '__main__':

    args = [
        (n, d, k, r, sym)
        for n in [500, 1000, 2000]
        for d in [2, 3, 5, 10]
        for k in [3, 4, 5, 6]
        for r in [0.7, 1.0]
        for sym in [True, False]]

    pool = Pool()
    with tqdm(total=len(args)) as pbar:
        for i, _ in tqdm(enumerate(pool.imap_unordered(make, args))):
            pbar.update()
