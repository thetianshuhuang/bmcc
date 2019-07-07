
from bclust.core import MODEL_DPM, MODEL_MFM, COMPONENT_NORMAL_WISHART
import numpy as np


class NormalWishart:

    CAPSULE = COMPONENT_NORMAL_WISHART

    def __init__(self, df=2):

        self.df = df

    def get_args(self, data):

        return {
            "df": float(self.df),
            "s_chol": np.linalg.cholesky(np.cov(data.T))
        }


class DPM:

    CAPSULE = MODEL_DPM

    def __init__(self, alpha=1):

        self.alpha = alpha

    def get_args(self, data):

        return {"alpha": float(self.alpha)}


class MFM:

    CAPSULE = MODEL_MFM

    def __init__(self, gamma=1, df=None):

        self.gamma = gamma

    def get_args(self, data):

        return {"v_n": None, "gamma": float(self.gamma)}  # todo


