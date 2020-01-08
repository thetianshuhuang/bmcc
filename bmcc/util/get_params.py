"""Get Mixture and Component model parameters"""


def get_params(data, assignments, cm, mm):
    """Get Model Parameters from component and mixture model.

    Parameters
    ----------
    data : np.array
        Data matrix. Used for data-dependent priors.
    assignments : np.array
        Assignment vector. Passed to models that require a reference to the
        assigment vector.
    cm : Object
        Component Model object
    mm : Object
        Mixture Model object

    Returns
    -------
    dict
        Dictionary containing model hyperparameters and prior parameters
    """

    params = {"dim": data.shape[1]}

    try:
        params.update(cm.get_args(data, assignments))
        params.update(mm.get_args(data, assignments))
    except AttributeError:
        raise TypeError(
            "Component Model and Mixture Model must have 'get_args' "
            "attribute (used to fetch dictionary args for capsule "
            "initializer)")

    return params
