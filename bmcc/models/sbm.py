import numpy as np
from bmcc.core import COMPONENT_STOCHASTIC_BLOCK_MODEL, sbm_update


class SBM:

    DATA_TYPE = np.uint8
    DATA_TYPE_NAME = "uint8"

    CAPSULE = COMPONENT_STOCHASTIC_BLOCK_MODEL

    def __init__(self, Q, alpha=1, beta=1):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.Q = Q

    def get_args(self, data, assignments):

        return {
            "Q": self.Q,
            "n": data.shape[0],
            "alpha": self.alpha,
            "beta": self.beta,
            "asn": assignments,
            "data": data
        }

    def update(self, mixture):

        # Q_old = mixture.inspect()

        # print(Q_old.shape)
        # print(Q_old)
        # print(mixture.num_clusters)

        return {
            "Q": sbm_update(
                mixture.data, mixture.assignments,
                mixture.num_clusters, self.alpha, self.beta)
        }
