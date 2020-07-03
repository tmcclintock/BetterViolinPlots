"""
Tests of violin plot functions.
"""

from unittest import TestCase

import numpy as np
import pytest
import scipy.stats as ss

from bvp import analytic_violin, kde_violin
from bvp.plots import _xy_order


class analytic_violin_test(TestCase):
    def setUp(self):
        super().setUp()
        self.dists = [ss.norm(loc=0, scale=1), ss.norm(loc=0, scale=1)]
        self.disc_dists = [ss.poisson(mu=3, loc=0), ss.poisson(mu=3, loc=0)]

    def test_smoke_continuous(self):
        fig, ax = analytic_violin(self.dists)
        assert ax is not None
        assert fig is not None

    def test_smoke_discrete(self):
        fig, ax = analytic_violin(self.disc_dists)
        assert ax is not None
        assert fig is not None

    def test_asserts(self):
        with pytest.raises(AssertionError):
            analytic_violin(self.dists, positions=[0])
        with pytest.raises(AssertionError):
            analytic_violin(self.dists, positions=[0, 1, 2])
        with pytest.raises(AssertionError):
            analytic_violin(self.dists, plot_kwargs=[{}, {}, {}])


class kde_violin_test(TestCase):
    def setUp(self):
        super().setUp()
        self.dists = [ss.norm(loc=0, scale=1), ss.norm(loc=0, scale=1)]
        self.samples = [d.rvs(size=1000) for d in self.dists]

    def test_smoke_continuous(self):
        fig, ax = kde_violin(self.samples)
        assert ax is not None
        assert fig is not None

    def test_asserts(self):
        with pytest.raises(AssertionError):
            kde_violin(self.samples, positions=[0])
        with pytest.raises(AssertionError):
            kde_violin(self.samples, positions=[0, 1, 2])
        with pytest.raises(AssertionError):
            kde_violin(self.samples, plot_kwargs=[{}, {}, {}])


class _xy_order_test(TestCase):
    def test_correct_orders(self):
        x = np.arange(10)
        y = np.arange(10) + 5
        xout, yout = _xy_order(x, y, vertical_violin=False)
        assert np.all(x == xout)
        assert np.all(y == yout)
        xout, yout = _xy_order(x, y, vertical_violin=True)
        assert np.all(x == yout)
        assert np.all(y == xout)


if __name__ == "__main__":
    avt = analytic_violin_test()
    avt.setUp()
    avt.test_smoke_discrete()
