source("../r/bmcc/R/main.R")

dataset <- GaussianMixture(
    n=1000L, k=3L, d=2L, r=0.7, alpha=10, df=2, symmetric=FALSE, shuffle=FALSE)

model <- GibbsMixtureModel(
    data=dataset$data,
    component_model=NormalWishart(df=2),
    mixture_model=DPM(alpha=1, use_eb=TRUE),
    assignments=np_array(rep(0, 1000), dtype="uint16"),
    thinning=1L
)

for(i in 1:1000) {
    model$iter()
}

res <- model$select_lstsq(burn_in=100L)
res$evaluate(
    dataset$assignments,
    oracle=dataset$oracle,
    oracle_matrix=dataset$oracle_matrix
)

res$trace(plot=TRUE)
res$matrices(plot=TRUE)
res$clustering(plot=TRUE)
