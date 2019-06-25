
import numpy as np


class MFMMixins:

    REQUIRED_COEF_PARAMS = ["gamma"]

    def coef_cond(self, data, cluster, idx, params):
        """Conditional coefficient function

        Coefficient of p(point | cluster): |cluster| + gamma
        """

        return len(cluster) + params["gamma"]

    def coef_uncond(self, data, idx, params):
        """Unconditional coefficient function

        Coefficient of 
        """

        return params["alpha"]