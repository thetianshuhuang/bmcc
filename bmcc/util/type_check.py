"""Error and type checking routines"""

import numpy as np
from .errors import (
    WARNING_CONTIGUOUS_CAST,
    WARNING_FLOAT64_CAST,
    WARNING_UINT16_CAST,
    ERROR_NO_CAPSULE,
    ERROR_CAPSULE_WRONG_API
)
from bmcc.models import NormalWishart, DPM
from bmcc.core import (
    get_capsule_name,
    COMPONENT_METHODS_API,
    MODEL_METHODS_API
)


def check_data(data):
    """Check data array type.

    Parameters
    ----------
    data : np.array
        Array to check

    Raises
    ------
    TypeError
        data is not a numpy array, or does not have 2 dimensions

    Returns
    -------
    np.array
        Original array if all checks passed. If not a contiguous float64
        C-order array, is cast to the appropriate type and returned as a copy.
    """

    # Check types
    if type(data) != np.ndarray:
        raise TypeError(
            "Data must be a numpy array.")
    if len(data.shape) != 2:
        raise TypeError(
            "Data must have 2 dimensions. The points should be stored in "
            "row-major order (each data point is a row).")
    if data.dtype != np.float64:
        print(WARNING_FLOAT64_CAST)
        data = data.astype(np.float64)
    if not data.flags['C_CONTIGUOUS']:
        print(WARNING_CONTIGUOUS_CAST)
        data = np.ascontiguousarray(data, dtype=np.float64)
    return data


def check_assignments(assignments, size):
    """Check assignment array type."""

    # No assignments -> skip checks and just build from scratch
    if assignments is None:
        print(
            "No initial assignments provided. Assigning all points to the "
            "same cluster at initialization.")
        return np.zeros(size, dtype=np.uint16)

    # Check assignments type
    if type(assignments) != np.ndarray:
        raise TypeError("Assignments must be an array.")
    if len(assignments.shape) != 1:
        raise TypeError("Assignments must be one-dimensional.")
    if assignments.shape[0] != size:
        raise TypeError(
            "Assignments must have the same dimensionality as the number "
            "of data points.")
    if assignments.dtype != np.uint16:
        print(WARNING_UINT16_CAST)
        assignments = assignments.astype(np.uint16)

    return assignments


def check_mixture_model(model):

    # No model -> skip checks and create new
    if model is None:
        print(
            "No mixture model provided; using DPM with initial alpha = 1.")
        return DPM(alpha=1)

    # Check capsule
    if not hasattr(model, "CAPSULE"):
        raise TypeError(ERROR_NO_CAPSULE.format(mtype="Mixture"))

    # Check API name
    name = get_capsule_name(model.CAPSULE)
    if(name != MODEL_METHODS_API):
        raise TypeError(ERROR_CAPSULE_WRONG_API.format(
            mtype="Mixture",
            expected=MODEL_METHODS_API,
            recieved=name))

    return model


def check_component_model(model, dim):

    # No model -> skip checks and create new
    if model is None:
        print(
            "No component model provided; using Normal Wishart with "
            "df=dim.")
        return NormalWishart(df=dim)

    # Check for capsule
    if not hasattr(model, "CAPSULE"):
        raise TypeError(ERROR_NO_CAPSULE.format(mtype="Component"))

    # Check API name
    name = get_capsule_name(model.CAPSULE)
    if(name != COMPONENT_METHODS_API):
        raise TypeError(ERROR_CAPSULE_WRONG_API.format(
            mtype="Component",
            expected=COMPONENT_METHODS_API,
            recieved=name))

    return model
