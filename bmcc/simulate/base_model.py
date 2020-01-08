
import numpy as np

from bmcc.core import oracle_matrix


def raw(x):
    return x


class BaseModel:
    """Simulate A Gaussian Mixture Dataset.

    Attributes
    ----------
    API_NAME : str
        String identifier for npz objects saved by this class. This class will
        only save to and load from npz files with the attribute
        f[API_NAME] = True.
    MODEL_NAME : str
        String identifier for this object.
    """

    _KEYS = {
        int: [],
        float: [],
        bool: [],
        raw: [],
    }

    API_NAME = "bmcc_BaseModel"
    MODEL_NAME = "BaseModel"

    def __init__(self, *args, load=False, **kwargs):
        if load:
            self._init_load(*args, **kwargs)
        else:
            self._init_new(*args, **kwargs)

    def _make_assignments(self, n, k, r, shuffle):
        # Compute weights
        weights = np.array([r**i for i in range(k)])
        weights = weights / sum(weights)

        # Make assignments
        assignments = np.random.choice(
            k, size=n, p=weights).astype(np.uint16)
        if not shuffle:
            assignments.sort()

        return weights, assignments

    def _init_load(self, src):
        """Load from file"""

        fz = np.load(src)

        if self.API_NAME not in fz:
            raise Exception(
                "Target file is not a valid {} save file.".format(
                    self.MODEL_NAME))

        for key, value in self._KEYS.items():
            for attr in value:
                setattr(self, attr, key(fz[attr]))

    @property
    def likelihoods(self):
        """Likelihood table"""
        raise Exception("This attribute not implemented.")

    @property
    def oracle(self):
        """Oracle assignments"""
        raise Exception("This attribute not implemented.")

    @property
    def oracle_matrix(self):
        """Oracle Pairwise Probability Matrix"""
        raise Exception("This attribute not implemented.")

    def save(self, dst):
        """Save simulated dataset to a file."""

        save_params = {
            attr: getattr(self, attr)
            for key, value in self._KEYS.items()
            for attr in value
        }
        save_params[self.API_NAME] = True

        np.savez(dst, **save_params)

    def __str__(self):
        return "{} object".format(self.MODEL_NAME)

    def __repr__(self):
        return self.__str__()
