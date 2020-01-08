import bmcc

dataset = bmcc.GaussianMixture(
    n=1000, k=3, d=3, r=0.7, alpha=40, df=3, symmetric=False, shuffle=False)
dataset.plot_actual(plot=True)
dataset.save("tmp.npz")

dataset = bmcc.GaussianMixture("tmp.npz", load=True)
dataset.plot_actual(plot=True)
