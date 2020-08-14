# BetterViolinPlots [![Build Status](https://travis-ci.com/tmcclintock/BetterViolinPlots.svg?branch=master)](https://travis-ci.com/tmcclintock/BetterViolinPlots.svg?branch=master)

Better violin plots for common scenarios.

This package allows for violin plots of any and all of:

- analytic distributions
- KDE estimates from samples
- box plots
- single or double sided violin plots

Packages like `matplotlib` or `seaborn` are limited in that they
do not have all of these options.

## Installation

After cloning, make sure the requirements are installed with
```bash
pip install -r requirements.txt
```
then do the actual install of this package by performing
```bash
python setup.py install
```
and run the tests
```bash
pytest
```
Please report any issues
[here](https://github.com/tmcclintock/BetterViolinPlots/issues).

To contribute to the development, there are a few more requirements
found in `environment.yml`. To install, the environment, follow the directions
on [creating and environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file), with
```bash
conda env create -f environment.yml
```
Followed by the installation instructions above.

## Usage

Import the package or a single routine and use it with the appropriate kinds of data (either a `scipy`-like distribution or samples of points).
```python
from bvp import analytic_violin

from scipy.stats import norm

# Five normal distributions
distributions = [norm() for _ in range(5)]

fig, ax = analytic_violin(distributions)
```