"""
Tests of violin plot functions.
"""

from unittest import TestCase

import pytest
import scipy.stats as ss

from bvp import analytic_violin


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


if __name__ == "__main__":
    avt = analytic_violin_test()
    avt.setUp()
    avt.test_smoke_discrete()
