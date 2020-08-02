"""
A set of functions that should not be publically accessible.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def _preamble(
    data, axis, plot_kwargs, positions, vertical_violins, sides="both"
):
    if vertical_violins is True:
        assert sides in ["both", "left", "right"]
    else:  # horizontal violins
        assert sides in ["both", "top", "bottom"]

    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()

    if isinstance(plot_kwargs, list):
        assert len(data) == len(plot_kwargs)

    if positions is not None:
        assert len(data) == len(positions)
    else:
        # Horizontal positions of the centers of the violins
        positions = np.arange(0, len(data))

    # Center positions between integers
    if vertical_violins:
        axis.set_xlim(positions.min() - 0.5, positions.max() + 0.5)
    else:
        axis.set_ylim(positions.min() - 0.5, positions.max() + 0.5)
    return fig, axis, positions


def _xy_order(domain: List, dist: List, vertical_violin: bool):
    if vertical_violin:
        return dist, domain
    return domain, dist


def _plot_from_x_dist(
    axis, x, y, index, kwargs, vertical_violins, sides="both", fill=False
):
    scale = 0.4 / y.max()
    if sides in ["both", "left", "top"]:
        axis.plot(
            *_xy_order(x, index - y * scale, vertical_violins), **kwargs,
        )
    if sides in ["both", "right", "bottom"]:
        axis.plot(
            *_xy_order(x, index + y * scale, vertical_violins), **kwargs,
        )
    return


def _inner_from_x_and_kde(
    axis, x, y, index, inner, scale, vertical_violins, sides="both"
):
    for i, (xi, yi) in enumerate(zip(x, y)):
        if sides in ["both", "left", "top"]:
            xii, yii = _xy_order(
                [xi, xi], [index, index - yi * scale], vertical_violins
            )
            if inner in ["stick", "quartiles"]:
                if inner == "quartiles" and i == 1:
                    axis.plot(xii, yii, c="k", alpha=0.5)
                else:
                    axis.plot(xii, yii, c="k", alpha=0.5, ls=":")
        if sides in ["both", "right", "bottom"]:
            xii, yii = _xy_order(
                [xi, xi], [index, index + yi * scale], vertical_violins
            )
            if inner in ["stick", "quartiles"]:
                if inner == "quartiles" and i == 1:
                    axis.plot(xii, yii, c="k", alpha=0.5)
                else:
                    axis.plot(xii, yii, c="k", alpha=0.5, ls=":")
    return
