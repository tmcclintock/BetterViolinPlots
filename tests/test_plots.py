"""
Tests of violin plot functions.
"""

from unittest import TestCase

import scipy.stats as ss


class analytic_plot_test(TestCase):
    def setUp(self):
        super().setUp()
        self.dist = ss.norm(loc=0, scale=1)
