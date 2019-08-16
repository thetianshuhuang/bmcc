"""Clustering post-processing operations"""

from bmcc.core import cleanup_gibbs, init_model
from bmcc.util import (
    check_data, check_assignments,
    check_mixture_model, check_component_model,
    get_params
)


def cleanup_maximum_likelihood(
        data, assignments,
        component_model=None,
        mixture_model=None):
    """Apply maximum likelihood post-processing procedure

    Parameters
    ----------
    data : np.array
        Data matrix
    assignments : np.array
        Assignments; output of select_lstsq or equivalent
    component_model : object
        Component model. Can reuse the same model used by the main sampler.
    mixture_model : object
        Mixture model. Can reuse the same model used by the main sampler.

    Returns
    -------
    np.array
        Cleaned assignments vector.
    """

    data = check_data(data)
    assignments = check_assignments(assignments, data.shape[0])
    mixture_model = check_mixture_model(mixture_model)
    component_model = check_component_model(component_model, data.shape[1])

    model = init_model(
        data, assignments,
        component_model.CAPSULE,
        mixture_model.CAPSULE,
        get_params(data, component_model, mixture_model))
    cleanup_gibbs(data, assignments, model)

    return assignments
