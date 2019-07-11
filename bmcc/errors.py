"""Warning Descriptions"""


WARNING_FLOAT64_CAST = """
Data array cast to np.float64. To suppress this message, copy data to a float64
array:
    Python:
        data = data.astype(float64)
    R/Reticulate:
        data_py = np_array(data, dtype="float64", order="C")
"""

WARNING_CONTIGUOUS_CAST = """
Data array copied onto contiguous C array. To suppress this message, copy to a
contiguous C-style array:
    Python:
        data = data.ascontiguousarray(data, dtype=np.float64)
    R/Reticulate:
        data_py = np_array(data, dtype="float64", order="C")
"""

WARNING_UINT16_CAST = """
Assignment vector cast to np.uint16. To suppress this message, copy data to a
uint16 array:
    Python:
        assignments = assignments.astype(np.uint16)
    R/Reticulate:
        assignments_py = np_array(assignments, dtype="uint16", order="C")
"""
