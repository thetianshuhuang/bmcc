#
#
#

library(reticulate)


IMPORT_ERROR_MSG = "
'bmcc' python module could not be imported.

bmcc requires Python to be installed, and 
"


bmcc_installed <- FALSE;

_bmcc_src = tryCatch({
	import("bmcc", import=FALSE)
	bmcc_installed <- TRUE
}, error=funtion(e) {
	bmcc_installed <- FALSE
}

if(!bmcc_installed) {
	print(IMPORT_ERROR_MSG)

}
