"""
A set of functions that should not be publically accessible.
"""

from typing import List


def _xy_order(domain: List, dist: List, vertical_violin: bool):
    if vertical_violin:
        return dist, domain
    else:
        return domain, dist


def _plot_from_x_dist(
    axis, x, y, index, kwargs, vertical_violins, sides="both"
):
    scale = 0.4 / y.max()
    # left side
    if sides in ["both", "left", "top"]:
        axis.plot(
            *_xy_order(x, index - y * scale, vertical_violins), **kwargs,
        )
    if sides in ["both", "right", "bottom"]:
        # right side
        axis.plot(
            *_xy_order(x, index + y * scale, vertical_violins), **kwargs,
        )
    return


def _inner_from_x_and_kde(
    axis, x, y, index, inner, scale, vertical_violins, sides="both"
):
    for i, (xi, yi) in enumerate(zip(x, y)):
        # left side
        if sides in ["both", "left", "top"]:
            xii, yii = _xy_order(
                [xi, xi], [index, index - yi * scale], vertical_violins
            )
            if inner == "stick":
                axis.plot(xii, yii, c="k", alpha=0.5)
            if inner == "quartiles":
                if i == 1:
                    axis.plot(xii, yii, c="k", alpha=0.5)
                else:
                    axis.plot(xii, yii, c="k", alpha=0.5, ls="--")
        # right side
        if sides in ["both", "right", "bottom"]:
            xii, yii = _xy_order(
                [xi, xi], [index, index + yi * scale], vertical_violins
            )
            if inner == "stick":
                axis.plot(xii, yii, c="k", alpha=0.5)
            if inner == "quartiles":
                if i == 1:
                    axis.plot(xii, yii, c="k", alpha=0.5)
                else:
                    axis.plot(xii, yii, c="k", alpha=0.5, ls="--")
    return
