

def get_params(data, cm, mm):
    """Get Model Parameters from component and mixture model.

    Parameters
    ----------
    data : np.array
        Data matrix. Used for data-dependent priors.
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
        params.update(cm.get_args(data))
        params.update(mm.get_args(data))
    except AttributeError:
        raise TypeError(
            "Component Model and Mixture Model must have 'get_args' "
            "attribute (used to fetch dictionary args for capsule "
            "initializer)")

    return params
