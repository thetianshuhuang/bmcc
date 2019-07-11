
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from bmcc.core import aggregation_score, segregation_score
from bmcc.plot import plot_clusterings
from bmcc.errors import WARNING_UINT16_CAST


class BaseResult:
    """Base Results Class

    Parameters
    ----------
    hist : np.array
        Array containing MCMC results; each assignment is a row.
    burn_in : int
        Burn in duration
    actual : np.array
        Array containing ground truth clusterings. If None, then it is assumed
        that no ground truth is available.
    nmi_method : str
        NMI method for mutual information calculations
        (see sklearn.metrics.normalized_mutual_info_score.html)

    Attributes
    ----------
    data : np.array
        Source data array
    hist : np.array
        History array (same as the input)
    burn_in : int
        Burn in duration used in analysis
    best_idx : int
        Index of 'best' clustering configuration. This class is a template
        class that does not select a clustering configuration; it is up to
        extending classes to provide a 'cluster' method that creates this
        attribute.
    best : np.array
        Best clustering configuration
    num_clusters : np.array
        Trace of number of clusters

    Attributes
    ----------
    [After 'evaluate' is called]
    nmi : np.array
        Normalized Mutual Information trace.
    nmi_best : float
        NMI of 'best' configuration (as selected above)
    rand : np.array
        Rand index trace.
    rand_best : float
        Rand Index of 'best' configuration (as selected above)
    aggregation : np.array
        Aggregation score trace: for selected assignment vector a, true
        assignment vector A,
        P[a[i] = a[j] | A[i] = A[j]]
    aggregation_best : float
        Aggregation score of 'best' configuration (as selected above)
    segregation : np.array
        Segregation score trace:
        P[a[i] != a[j] | A[i] != A[j]]
    segregation_best : float
        Segregation score of 'best' configuration (as selected above)
    oracle : np.array
        Oracle clustering assignments
    oracle_nmi, oracle_rand, oracle_segregation, oracle_aggregation : float
        Scores for oracle clustering
    """
    def __init__(self, data, hist, burn_in=0):

        # Check burn in
        if type(burn_in) != int:
            raise TypeError("Burn in period must be an integer.")
        if burn_in >= hist.shape[0]:
            raise ValueError(
                "Burn in period larger than the number of saved samples.")

        # Bind arguments
        self.data = data
        self.hist = hist
        self.burn_in = burn_in
        self.truth_known = False

        # Metadata Trace
        self.num_clusters = np.array([np.max(x) + 1 for x in self.hist])

        # Run selection
        self.cluster()

        # Make sure selection has a 'best' attribute set
        if not hasattr(self, 'best'):
            raise Exception(
                "'cluster' method of class extending BaseResult did not set "
                "a valid 'best' clustering.")

    def cluster(self):
        """Select clustering configuration from MCMC samples.

        This method is a placeholder, and will raise an error of called. This
        method should be overwritten by analysis types.
        """

        raise Exception(
            "Class extending BaseResult did not supply a 'cluster' method.")

    def __score_trace(self, actual, nmi_method):
        """Compute NMI and Rand Index trace.

        Parameters
        ----------
        actual : np.array
            Ground truth assignments
        """
        if type(actual) != np.ndarray:
            raise TypeError(
                "Ground truth clustering ('actual') must be a numpy "
                "array.")
        if len(actual.shape) != 1 or actual.shape[0] != self.hist.shape[1]:
            raise TypeError(
                "Ground truth clustering must have the same "
                "dimensionality as the MCMC history.")
        self.actual = actual

        self.nmi = np.array([
            normalized_mutual_info_score(actual, x, nmi_method)
            for x in self.hist])
        self.nmi_best = self.nmi[self.best_idx]

        self.rand = np.array([
            adjusted_rand_score(actual, x)
            for x in self.hist])
        self.rand_best = self.rand[self.best_idx]

        self.aggregation = np.array([
            aggregation_score(actual, x) for x in self.hist])
        self.aggregation_best = self.aggregation[self.best_idx]
        self.segregation = np.array([
            segregation_score(actual, x) for x in self.hist])
        self.segregation_best = self.segregation[self.best_idx]

    def evaluate(
            self, actual,
            oracle=None, oracle_matrix=None,
            nmi_method='arithmetic'):
        """Evaluate clustering against ground truth.

        Parameters
        ----------
        actual : np.array
            Actual assignments
        oracle : np.array
            Oracle assignments (used as point of comparison instead of
            'perfect' clustering). Used only if present.
        oracle_matrix : np.array
            Oracle pairwise probability matrix.
        nmi_method : str
            Normalization method for NMI calculations
        """

        if actual.dtype != np.uint16:
            print(WARNING_UINT16_CAST)
            actual = actual.astype(np.uint16)

        self.actual = actual
        self.clusters_true = np.max(actual) + 1

        if oracle is not None:

            if oracle.dtype != np.uint16:
                print(WARNING_UINT16_CAST)
                oracle = oracle.astype(np.uint16)

            self.oracle = oracle
            self.oracle_nmi = normalized_mutual_info_score(
                actual, oracle, nmi_method)
            self.oracle_rand = adjusted_rand_score(actual, oracle)
            self.oracle_aggregation = aggregation_score(actual, oracle)
            self.oracle_segregation = segregation_score(actual, oracle)
            self.oracle_matrix = oracle_matrix

        else:
            self.oracle = actual
            self.oracle_nmi = 1
            self.oracle_rand = 1

        self.truth_known = True
        self.__score_trace(actual, nmi_method)

    DEFAULT_TRACE_PLOTS = {}

    def trace(self, plots=None, plot=False):
        """Plot NMI, Rand Index, and # of clusters as a trace over MCMC
        iterations.

        Parameters
        ---------
        plots : dict
            Dictionary of (key, value) pairs. Each key corresponds to a plot,
            which is plotted with the attribute corresponding to value. If
            value is a dict, multiple keys are plotted on the same plot.
        plot : bool
            If True, calls plt.show(). Otherwise, returns the generated figure.

        Returns
        -------
        plt.figure.Figure or None
            Created figure; plot with fig.show().
        """

        if plots is None:
            plots = self.DEFAULT_TRACE_PLOTS

        if not self.truth_known:
            raise ValueError(
                "Cannot show scores: ground truth not known.")

        if len(plots) > 1:
            fig, axs = plt.subplots(len(plots), 1)
        else:
            fig, axs = plt.subplots(1, 1)
            axs = [axs]

        for ax, (k, v) in zip(axs, plots.items()):
            ax.set_title(k)
            ax.axvline(self.burn_in, color='black', label='Burn In')
            if type(v) == str:
                ax.plot(getattr(self, v))
            elif type(v) == dict:
                for k_inner, v_inner in v.items():
                    if np.isscalar(getattr(self, v_inner)):
                        ax.plot(
                            [0, self.hist.shape[0]],
                            [getattr(self, v_inner), getattr(self, v_inner)],
                            label=k_inner)
                    else:
                        ax.plot(getattr(self, v_inner), label=k_inner)
                ax.legend(loc='lower right')
            else:
                raise TypeError(
                    "Attribute name is not a string or dict.")

        if plot:
            plt.show()
            return None
        else:
            return fig

    def clustering(
            self, bins=20, plot=False,
            kwargs_hist={}, kwargs_scatter={}):
        """Show clustering as a multi-dimensional array of scatterplots

        Parameters
        ----------
        bins : int
            Number of bins
        plot : bool
            If True, calls plt.show(). Otherwise, returns the generated figure.
        kwargs_hist : dict
            Keyword args to pass onto plt.hist
        kwargs_scatter : dict
            Keyword args to pass onto plt.scatter

        Returns
        -------
        plt.figure.Figure
            Created figure; plot with fig.show().
        """

        fig = plot_clusterings(
            self.data, self.best, bins=bins,
            kwargs_hist=kwargs_hist,
            kwargs_scatter=kwargs_scatter)

        if plot:
            plt.show()
            return None
        else:
            return fig
