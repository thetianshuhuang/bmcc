"""Hybrid Model combining DPM and MFM.

DPM tends to over-cluster, but converges very quickly; on the other hand, MFM
greatly reduces over-clustering, at the cost of tending to take much longer to
converge. Therefore, this method combines DPM and MFM by using DPM at the
start, while gradually transitioning to MFM using some assignment function
```beta``` which may or may not be stochastic.
"""

from bmcc.core import MODEL_HYBRID


class Hybrid:
    """Hybrid Mixture Model combining DPM and MFM

    The model choice per iteration is selected using a function
    beta(iteration #), which may or may not be stochastic.

    Parameters
    ----------
    MFM : bmcc.MFM object
        MFM object to mix
    DPM : bmcc.DPM object
        DPM object to mix

    Keyword Args
    ------------
    beta : function(n) -> bool (use MFM?)
        Function to decide whether to use MFM or DPM depending on the iteration
        number. Use random.random() or some other randomizer to make this
        function stochastic; otherwise, a step function can be used
        such as 'lambda n: (n > 100)'
    """

    CAPSULE = MODEL_HYBRID

    def __init__(self, MFM, DPM, beta=None):

        if beta is None:
            def beta(n):
                return (n > 100)

        self.MFM = MFM
        self.DPM = DPM
        self.beta = beta
        self.iter = 0

    def get_args(self, data):
        """Get Model Hyperparameters; calls the get_args functions on the
        underlying MFM and DPM objects, and combines them.

        Parameters
        ----------
        data : np.array
            Dataset; used for scale matrix S = 1/df * Cov(data)

        Returns
        -------
        dict
            Argument dictionary to be passed to core.init_model, with entries:
            "alpha": DPM mixing parameter
        """

        args = {"is_mfm": False}
        args.update(self.MFM.get_args(data))
        args.update(self.DPM.get_args(data))

        return args

    def update(self, mixture):
        """Update per-iteration choice of MFM or DPM using the supplied
        function

        Parameters
        ----------
        mixture : MixtureModel Object
            Object to update alpha for

        Returns
        -------
        dict
            Updated hyperparameters, with entry:
            "is_mfm": bool; value of beta(iteration #)
        """

        self.iter += 1
        return {
            "is_mfm": self.beta(self.iter)
        }
