#
#
#

library(reticulate)


IMPORT_ERROR_MSG = "
'bmcc' python module could not be imported.

bmcc requires Python and the python module 'bmcc' to be installed. Install with
$ pip install bmcc

on the linux / windows terminal.
"


# Check for installation
__bmcc_installed <- FALSE;

_bmcc_src = tryCatch({
    import("bmcc", import=FALSE)
    __bmcc_installed <<- TRUE
}, error=funtion(e) {
    __bmcc_installed <<- FALSE
}
if(!__bmcc_installed) {
    print(IMPORT_ERROR_MSG)
}
else {




}
