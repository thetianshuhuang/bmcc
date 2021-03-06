"""Warning and Error Descriptions"""


WARNING_FLOAT64_CAST = """
Data array cast to np.{dtype}. To suppress this message, copy data to a {dtype}
array:
    Python:
        data = data.astype({dtype})
    R/Reticulate:
        data_py = np_array(data, dtype="{dtype}", order="C")
"""

WARNING_CONTIGUOUS_CAST = """
Data array copied onto contiguous C array. To suppress this message, copy to a
contiguous C-style array:
    Python:
        data = data.ascontiguousarray(data, dtype=np.{dtype})
    R/Reticulate:
        data_py = np_array(data, dtype="{dtype}", order="C")
"""

WARNING_UINT16_CAST = """
Assignment vector cast to np.uint16. To suppress this message, copy data to a
uint16 array:
    Python:
        assignments = assignments.astype(np.uint16)
    R/Reticulate:
        assignments_py = np_array(assignments, dtype="uint16", order="C")
"""

ERROR_NO_CAPSULE = """
{mtype} model must have a 'CAPSULE' attribute (containing C functions
describing model methods)
"""

ERROR_CAPSULE_WRONG_API = """
{mtype} model capsule does not match the required API.
Expected API name: {expected}
Recieved API name: {recieved}
"""
