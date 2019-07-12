source("main.R")

# Create Dataset
dataset <- GaussianMixture(
    n=1000L, k=4L, d=3L, r=0.7, alpha=10, df=3, symmetric=FALSE, shuffle=FALSE)

# Select Model
model <- GibbsMixtureModel(
    data=dataset$data,
    component_model=NormalWishart(df = 3),
    mixture_model=MFM(gamma = 1, prior = py_func(function(k) {k * log(0.8) })),
    assignments=np_array(rep(0, 1000), dtype="uint16"),
    thinning=5L
)

# Run Iterations
pb <- txtProgressBar(min = 0, max = 5000, style = 3)
for(i in 1:5000) {
    model$iter()
    setTxtProgressBar(pb, i)
}
close(pb)

# Select Least Squares clustering
res <- model$select_lstsq(burn_in=100L)
res$evaluate(
    dataset$assignments,
    oracle=dataset$oracle,
    oracle_matrix=dataset$oracle_matrix
)
cat(paste("num_clusters: ", print(res$num_clusters[res$best_idx])))

# Plot
res$trace(plot=TRUE)
res$matrices(plot=TRUE)
res$clustering(plot=TRUE)
