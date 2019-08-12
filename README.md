# Bayesian Markov Chain Monte Carlo Clustering

Implementation of Markov Chain Monte Carlo Bayesian Clustering techniques, including DPM (Dirichlet Process Mixture Models; [Neal, 2000](https://github.com/thetianshuhuang/bmcc/wiki/References#1-neal-2000)) and MFM (Mixture of Finite Mixtures; [Miller & Harrison, 2018](References#2-miller--harrison-2018))) mixture models, with an abstract Mixture Model and Component Model API.

Both Gibbs samplers [Neal, 2000](https://github.com/thetianshuhuang/bmcc/wiki/References#1-neal-2000), Split Merge samplers [Jain & Neal, 2012] are implemented.

Hyperparameter updates for DPM are (optionally) implemented using an Empirical Bayes update procedure [(McAuliffe et. al., 2006)](References#3-mcauliffe-et-al-2006).

Final configuration selection is implemented using Least Squares clustering [(Dahl, 2006)](References#4-dahl-2006).

[[Wiki / Documentation]](https://github.com/thetianshuhuang/bmcc/wiki)
[[References]](https://github.com/thetianshuhuang/bmcc/wiki/References)

![](https://github.com/thetianshuhuang/bmcc/blob/master/preview/scatter.png)

## Installation

### Python

```bmcc``` is on PyPI (the Python Package Index):

```shell
pip install bmcc
```
(or ```pip3 install bmcc```, depending on your environment.)

**NOTE**: Only Python 3 is supported. Python 2 may not work.

### R

First, make sure Python 3 is installed, and install with ```pip install bmcc``` as per above. Then, install the R package with 

```R
library(devtools)
install_github("https://github.com/thetianshuhuang/bmcc_r")
```

To use, load the package ```bmcc```. You will also need to load ```reticulate``` in order to deal with type conversions.

```R
library(bmcc)
library(reticulate)
```

See [the wiki page](https://github.com/thetianshuhuang/bmcc/wiki/Installation-and-Basic-Usage) for more details.

## Usage

See the [wiki](https://github.com/thetianshuhuang/bmcc/wiki) for documentation and usage instructions.

Three [examples](https://github.com/thetianshuhuang/bmcc/tree/master/examples) are available.
