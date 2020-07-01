"""
Better violin plots than usual.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_discrete, rv_continuous


def _xy_order(domain: List, dist: List, vertical_violin: bool) -> Dict:
    if vertical_violin:
        return dist, domain  # {"x": dist, "y": domain}
    else:
        return domain, dist  # {"x": domain, "y": dist}


def analytic_violin(
    dists: List,
    positions: Optional[List[int]] = None,
    axis: Optional["mpl.axes.Axes"] = None,
    vertical_violins: bool = True,
    plot_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = {},
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create violin plots of the analytic distributions.

    Note: can accept either discrete or continuous distributions.

    Args:
        dists (List): analytic distributions
        axis (mpl.axes.Axes): axis to use for plotting, default `None`
        vertical_violins (bool): flag to indicate orientation
        plot_kwargs (Dict or List): if Dict, a dictionary of keyword-value
            pairs to pass to each plot routine. If List, it is a list of
            Dict objects to pass, one for each plot routine
    """
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()

    if plot_kwargs is not None:
        if isinstance(plot_kwargs, list):
            assert len(dists) == len(plot_kwargs)

    if positions is not None:
        assert len(dists) == len(positions)
    else:
        # Horizontal positions of the centers of the violins
        positions = np.arange(0, len(dists))

    # Center positions between integers
    if vertical_violins:
        axis.set_xlim(positions.min() - 0.5, positions.max() + 0.5)
    else:
        axis.set_ylim(positions.min() - 0.5, positions.max() + 0.5)

    # Loop over all distributions and draw the violin
    for i, d in zip(positions, dists):
        interval = d.interval(0.99999)  # 5-sigma

        if isinstance(plot_kwargs, list):
            kwargs = plot_kwargs[i]
        else:
            kwargs = plot_kwargs

        # Handle continuous vs discrete cases differently
        if hasattr(d, "dist"):

            if isinstance(d.dist, rv_discrete):
                x = np.arange(min(interval), max(interval) + 1)
                y = d.pmf(x)
                scale = 0.4 / y.max()
                for xi, yi in zip(x, y):
                    # right side
                    axis.plot(
                        _xy_order(
                            [xi, xi + 1],
                            [i - yi * scale, i - yi * scale],
                            vertical_violins,
                        ),
                        **kwargs,
                    )
                    # left side
                    axis.plot(
                        _xy_order(
                            [xi, xi + 1],
                            [i + yi * scale, i + yi * scale],
                            vertical_violins,
                        ),
                        **kwargs,
                    )
            elif isinstance(d.dist, rv_continuous):
                x = np.linspace(min(interval), max(interval), 100)
                y = d.dist.pdf(x)
                scale = 0.4 / y.max()
                # left side
                axis.plot(
                    _xy_order(x, i - y * scale, vertical_violins), **kwargs,
                )
                print(_xy_order(x, i - y * scale, vertical_violins))
                # right side
                axis.plot(
                    _xy_order(x, i + y * scale, vertical_violins), **kwargs,
                )
            else:  # need to do random draws
                raise NotImplementedError(
                    "only scipy.stats distributions supported"
                )  # pragma: no cover
        else:
            raise NotImplementedError(
                "only scipy.stats distributions supported"
            )  # pragma: no cover

    return fig, axis


def point_violin():
    pass


def histogram_violin():
    pass
