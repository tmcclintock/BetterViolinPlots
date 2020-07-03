# BetterViolinPlots [![Build Status](https://travis-ci.com/tmcclintock/BetterViolinPlots.svg?branch=master)](https://travis-ci.com/tmcclintock/BetterViolinPlots.svg?branch=master)

Better violin plots for common scenarios.

This package allows for violin plots of any and all of:

- analytic distributions
- sample/point estimates
- KDE estimates from samples
- histogram representations
- single or double sided violin plots

In contrast, in traditional packages like `matplotlib` or `seaborn` you will find a single version.

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
Please report any issues [here](https://github.com/tmcclintock/BetterViolinPlots/issues).

## Usage

Import the package or a single routine and use it with the appropriate kinds of data (either a `scipy`-like distribution or samples of points).
```python
from bvp import analytic_violin

from scipy.stats import norm

# Five normal distributions
distributions = [norm() for _ in range(5)]

fig, ax = analytic_violin(distributions)
```