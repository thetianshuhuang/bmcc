"""Plot Clusterings"""


from matplotlib import pyplot as plt
import numpy as np


def plot_clusterings(
        data, assignments, bins=20,
        kwargs_scatter={}, kwargs_hist={}):
    """Plot N-dimensional clustering.

    Creates a NxN grid of plots; the [i, j] plot shows the [x_i, x_j]
    marginal scatterplot of the points. If i=j, the marginal histogram is
    shown.

    Parameters
    ----------
    data : np.array
        Data points. Must have 2 dimensions.
    assignments : np.array
        Point assignments. Pass an array of all 0s
        (i.e. np.zeros(data.shape[0])) if you only want to look at the points,
        with no computed assignments.
    bins : int
        Number of bins for each histogram.
    kwargs_scatter : dict
        Arguments to be passed on to plt.scatter
    kwargs_hist : dict
        Arguments to be passed on to plt.hist

    Returns
    -------
    plt.figure.Figure
        Created figure; plot with fig.show().
    """
    if type(data) != np.ndarray or len(data.shape) != 2:
        raise TypeError("data must be a 2-dimensional numpy array.")
    if type(assignments) != np.ndarray or len(assignments.shape) != 1:
        raise TypeError("assignments must be a 1-dimensional numpy array.")
    if data.shape[0] != assignments.shape[0]:
        raise TypeError(
            "assignments must have the same number of points as data")

    fig, plots = plt.subplots(data.shape[1], data.shape[1])

    for x in range(data.shape[1]):
        for y in range(data.shape[1]):
            # Histogram
            if x == y:
                plots[x][y].hist(
                    data[:, x], bins=bins, **kwargs_hist)
                plots[x][y].set_xlabel('x_{}'.format(x))
            # Scatterplot
            else:
                plots[x][y].scatter(
                    data[:, x], data[:, y], c=assignments, **kwargs_scatter)
                plots[x][y].set_xlabel('x_{}'.format(x))
                plots[x][y].set_ylabel('x_{}'.format(y))

    return fig
