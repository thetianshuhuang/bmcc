
UINT16_WARNING_MSG = '
Array cast to contiguous C-style numpy array with type uint16. To suppress this
message, convert to a numpy array with the correct type:

arr <- np_array(arr, dtype="uint16", order="C")
'

FLOAT64_WARNING_MSG = '
Array cast to contiguous C-style numpy array with type float64. To suppress
this message, convert to a numpy array with the correct type:

arr <- np_array(arr, dtype="float64", order="C"
'

INT_WARNING_MSG = '
"Numeric" (double) converted to "integer" type. To suppress this message, pass
integer arguments (such as indices) as explicit integers:

n <- 10
n <- as.integer(n)
or
n <- 10L
'

SCALAR_ERROR_MSG = '
Attempted to pass non-scalar value where a scalar value is required:
'

INT_ERROR_MSG = '
Attempted to pass non-integer value where an integer is required:
'

uint16_sanitize <- function(arr) {
    # Check whether arr is a valid uint16_t C-order numpy array, and cast to
    # the appropriate type if not.
    # Raises a warning if a cast was made.
    valid <- (
        py_to_r(bmcc$is_np_array(arr)) &&
        py_to_r(bmcc$is_contiguous(arr)) &&
        py_to_r(bmcc$is_uint16(arr))
    )

    if(!valid) {
        warning(UINT16_WARNING_MSG)
        arr <- tryCatch({
            np_array(arr, dtype="uint16", order="C")
        }, error=function(e) {
            stop(paste("Could not convert to numpy array:\n", print(e)))
        })
    }
    return(arr)
}

float64_sanitize <- function(arr) {
    valid <- (
        py_to_r(bmcc$is_np_array(arr)) &&
        py_to_r(bmcc$is_contiguous(arr)) &&
        py_to_r(bmcc$is_float64(arr))
    )
    if(!valid) {
        warning(FLOAT64_WARNING_MSG)
        arr <- tryCatch({
            np_array(arr, dtype="float64", order="C")
        }, error=function(e) {
            stop(paste("Could not convert to numpy array:\n", print(e)))
        })
    }
    return(arr)
}


int_sanitize <- function(x) {
    # Check for non-scalar
    if(!(is.atomic(x) && (length(x) == 1L))) {
        stop(paste(SCALAR_ERROR_MSG, print(x)))
    }
    # Check for non-integer
    if(as.integer(x) != x) {
        stop(paste(INT_ERROR_MSG, print(x)))
    }
    # Cast
    if(typeof(x) != "integer") {
        warning(INT_WARNING_MSG)
        x <- as.integer(x)
    }
    return(x)
}
