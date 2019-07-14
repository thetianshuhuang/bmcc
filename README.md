# Bayesian Markov Chain Monte Carlo Clustering

Implementation of Markov Chain Monte Carlo Bayesian Clustering techniques, including DPM (Dirichlet Process Mixture Models [1]) and MFM (Mixture of Finite Mixtures [2]) mixture models, with an abstract Mixture Model and Component Model API.

Hyperparameter updates for DPM are implemented using an Empirical Bayes update procedure [3].

Final configuration selection is implemented using Least Squares clustering [4].


## Usage

### Installation and Setup
Python:
First, install with ```pip install bmcc``` (or ```pip3```, depending on your version). Then, simply ```import bmcc```.
**NOTE**: Only Python 3 is officially supported.

R:
First, make sure Python 3 is installed, and install bmcc with ```pip install bmcc```. Then, install the R package with
```R
library(devtools)
install_github("https://github.com/thetianshuhuang/bmcc_r")
```

To use, load the package ```bmcc```. You will also need to load ```reticulate``` in order to deal with type conversions.
```R
library(bmcc)
library(reticulate)
```

### Expected Types

The dataset should be an array with two dimensions, where each row is a data point. Arrays should be numpy arrays with data type float64, in contiguous C order.

- Python: ```data = np.array(<source>, dtype=np.float64)```
- R/reticulate: ```data = np_array(<source>, dtype="float64", order="C")```

Assignment vectors should be arrays. The value at each index indicates the cluster assignment. Since clusters are unordered, the value itself has no meaning, and should be ignored other than to determine uniqueness. Assignment vectors have type uint16.

- Python: ```assignments = np.array(<source>, dtype=np.uint16)```
- R/reticulate: ```assignments = np_array(<source>, dtype="uint16")```

Note that when using reticulate, R types default to 'numeric' (double). When calling functions that require integer arguments (i.e. indices, number of dimensions), integers must be explicitly specified:

```R
x <- 25
x <- as.integer(x)
# or
x <- 25L
```

### Creating the Model

The model has two parts: the mixture model, and the component model. Currently, the component model has one option, and the mixture model has two options.

#### Normal Wishart
The normal wishart model assumes each component is drawn from a wishart distribution with degrees of freedom specified in the initializer, and scale matrix proportional to C^-1/df, where C is the covariance matrix of the observed points.

- Python:
```
component_model = bmcc.NormalWishart(df=3)
```

- R/reticulate:
```
component_model = NormalWishart(df = 3)
```

#### MFM (Mixture of Finite Mixtures)
See [2]. Takes two arguments: the dirichlet mixing parameter gamma, and a log prior function on the number of clusters. Gamma defaults to 1, and the prior defaults to poisson(mu=4).
**NOTE**: Make sure that the prior is given in log form!

- Python:
```python
mixture_model = bmcc.MFM(gamma=1, prior=lambda k: poisson.logpmf(k, 4))
```

- R/reticulate:
```R
prior <- function(k) { dpois(k, 4, log = TRUE) }
mixture_model = MFM(gamma=1, prior=py_func(prior))
```

### Running the Gibbs Sampler

Currently, only collapsed gibbs samplers are implemented. Is is possible to extend this to a general gibbs sampler using the API (documentation todo), but for now, the core library only implements collapsed gibbs.

The GibbsMixtureModel gibbs sampler takes 5 arguments: the dataset, the models created previously, an initial assignment vector (usually assigning all points to the same cluster), and a thinning factor. If the thinning factor is 1, all samples are kept; otherwise, only one sample out of every ```thinning``` samples are kept, with the rest being immediately discarded.

- Python:
```python
sampler = bmcc.GibbsMixtureModel(
    data=data,
    component_model=component_model,
    mixture_model=mixture_model,
    assignments=np.zeros(1000).astype(np.uint16),
    thinning=1)
```

- R:
```R
sampler = GibbsMixtureModel(
    data=data,
    component_model=component_model,
    mixture_model=mixture_model,
    assignments=np_array(rep(0, 1000), dtype="uint16"),
    thinning=1L)
```

Finally, simply call the ```iter``` method once for every iteration:

- Python:
```python
for i in range(1000):
    sampler.iter()
```

- R:
```R
for(i in 1:1000):
    sampler$iter()
```

You can also call ```iter``` with an argument (i.e. ```sampler.iter(10)```) to run multiple iterations at once. I suggest running the loop with a progress bar of some sort:

- Python:
```python
import tqdm
# ...
for i in tqdm(range(1000)):
    sampler.iter()
```

- R:
```R
pb <- txtProgressBar(min = 0, max = 5000, style = 3)
for(i in 1:5000) {
    model$iter()
    setTxtProgressBar(pb, i)
}
close(pb)
```

### Selecting a Result

Currently, only least squares configuration selection is implemented. Run by calling the ```select_lstsq``` method of the ```GibbsMixtureModel``` object with the burn in duration.

- Python:
```python
res = sampler.select_lstsq(burn_in=100)
```

- R:
```R
res <- sampler$select_lstsq(burn_in=100L)
```

If ground truths are available, call the ```evaluate``` method of the resulting ```LstsqResult``` object to run evaluation statistics. If oracle information is available, the ```oracle``` (oracle assignments) and ```oracle_matrix``` (oracle pairwise probability) arguments can optionally be passed to allow comparison.

- Python:
```python
res.evaluate(
    dataset.assignments,
    oracle=dataset.oracle,
    oracle_matrix=dataset.oracle_matrix)
# Plot traces
res.trace(plot=True)
# Plot pairwise membership and probability matrices
res.matrices(plot=True)
# Plot clustering
res.clustering(plot=True)
```

- R:
```R
res$evaluate(
    dataset$assignments,
    oracle=dataset$oracle,
    oracle_matrix=dataset$oracle_matrix)
# Plot traces
res$trace(plot=TRUE)
# Plot pairwise membership and probability matrices
res$matrices(plot=TRUE)
# Plot clustering
res$clustering(plot=TRUE)
```

## References

[1] Radford M. Neal (2000), "Markov Chain Sampling Methods for Dirichlet Process Mixture Models". Journal of Computational and Graphical Statistics, Vol. 9, No. 2.

[2] Jeffrey W. Miller, Matthew T. Harrison (2018), "Mixture Models with a Prior on the Number of Components". Journal of the American Statistical Association, Vol. 113, Issue 521.

[3] Jon D. McAuliffe, David M. Blei, Michael I. Jordan (2006), "Nonparametric empirical Bayes for the Dirichlet process mixture model". Statistics and Computing, Vol. 16, Issue 1.

[4] David B. Dahl (2006), "Model-Based Clustering for Expression Data via a Dirichlet Process Mixture Model". Bayesian Inference for Gene Expression and Proteomics.
