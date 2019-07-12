"""Example usage of bmcc.

This sample simulates a mixture of gaussian, then runs Gibbs Sampling with both
a MFM and DPM prior.

This example illustrates how 'savefig' can be used to automatically export
plots.
"""


import numpy as np
from tqdm import tqdm
import bmcc


# Create dataset
dataset = bmcc.GaussianMixture(
    n=1000, k=4, d=3, r=0.7, alpha=6, df=3, symmetric=False, shuffle=False)

# Create mixture models
model_mfm = bmcc.GibbsMixtureModel(
    data=dataset.data,
    component_model=bmcc.NormalWishart(df=3),
    mixture_model=bmcc.MFM(
        gamma=1, prior=lambda k: (k - 1) * np.log(0.75) + np.log(0.25)),
    assignments=np.zeros(1000).astype(np.uint16),
    thinning=5)
model_dpm = bmcc.GibbsMixtureModel(
    data=dataset.data,
    component_model=bmcc.NormalWishart(df=3),
    mixture_model=bmcc.DPM(alpha=1, use_eb=True),
    assignments=np.zeros(1000).astype(np.uint16),
    thinning=5)

# Run Iterations
print("MFM:")
for i in tqdm(range(5000)):
    model_mfm.iter()
print("DPM:")
for i in tqdm(range(5000)):
    model_dpm.iter()

# Select Least Squares clustering
res_mfm = model_mfm.select_lstsq(burn_in=100)
res_mfm.evaluate(
    dataset.assignments,
    oracle=dataset.oracle,
    oracle_matrix=dataset.oracle_matrix)
res_dpm = model_dpm.select_lstsq(burn_in=100)
res_dpm.evaluate(
    dataset.assignments,
    oracle=dataset.oracle,
    oracle_matrix=dataset.oracle_matrix)


# Plots
def save_fig(fig, name):
    fig.set_size_inches(10, 8)
    fig.savefig(name + '.png')


save_fig(res_mfm.trace(), 'mfm_trace')
save_fig(res_mfm.matrices(), 'mfm_mat')
save_fig(res_mfm.clustering(kwargs_scatter={"marker": "."}), 'mfm_cluster')

save_fig(res_dpm.trace(), 'dpm_trace')
save_fig(res_dpm.matrices(), 'dpm_mat')
save_fig(res_dpm.clustering(kwargs_scatter={"marker": "."}), 'dpm_cluster')

save_fig(dataset.plot_oracle(), 'oracle_cluster')
