"""
Better violin plots than usual.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_discrete, rv_continuous, gaussian_kde


def _xy_order(domain: List, dist: List, vertical_violin: bool) -> Dict:
    if vertical_violin:
        return dist, domain  # {"x": dist, "y": domain}
    else:
        return domain, dist  # {"x": domain, "y": dist}


def analytic_violin(
    distributions: List,
    positions: Optional[List[int]] = None,
    axis: Optional["mpl.axes.Axes"] = None,
    vertical_violins: bool = True,
    plot_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = {
        "color": "black",
    },
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create violin plots of analytic distributions.

    .. note::

       can accept either discrete or continuous distributions.

    .. note::

       the default plot keywords are

       .. code-bloc:: python

          plot_kwargs = {
             "color": "black",
          }

    Args:
        distributions (List): analytic distributions
        positions (Optional[List[int]]): locations to plot the violins
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

    if isinstance(plot_kwargs, list):
        assert len(distributions) == len(plot_kwargs)

    if positions is not None:
        assert len(distributions) == len(positions)
    else:
        # Horizontal positions of the centers of the violins
        positions = np.arange(0, len(distributions))

    # Center positions between integers
    if vertical_violins:
        axis.set_xlim(positions.min() - 0.5, positions.max() + 0.5)
    else:
        axis.set_ylim(positions.min() - 0.5, positions.max() + 0.5)

    # Loop over all distributions and draw the violin
    for i, d in zip(positions, distributions):
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
                        *_xy_order(
                            [xi, xi + 1],
                            [i - yi * scale, i - yi * scale],
                            vertical_violins,
                        ),
                        **kwargs,
                    )
                    # left side
                    axis.plot(
                        *_xy_order(
                            [xi, xi + 1],
                            [i + yi * scale, i + yi * scale],
                            vertical_violins,
                        ),
                        **kwargs,
                    )
            elif isinstance(d.dist, rv_continuous):
                x = np.linspace(min(interval), max(interval), 1000)
                y = d.pdf(x)
                scale = 0.4 / y.max()
                # left side
                axis.plot(
                    *_xy_order(x, i - y * scale, vertical_violins), **kwargs,
                )
                # right side
                axis.plot(
                    *_xy_order(x, i + y * scale, vertical_violins), **kwargs,
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


def kde_violin(
    points: Union[List, np.ndarray],
    positions: Optional[List[int]] = None,
    axis: Optional["mpl.axes.Axes"] = None,
    vertical_violins: bool = True,
    plot_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = {
        "color": "black",
    },
    kde_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = {
        "bw_method": "scott",
        "weights": None,
    },
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create violin plots of Gaussian kernel density estimations (KDE)
    of points.

    .. note::

       the default plot keywords are

       .. code-bloc:: python

          plot_kwargs = {
             "color": "black",
          }

    Args:
        points (List): samples of an unknown distribution or list of samples
        positions (Optional[List[int]]): locations to plot the violins
        axis (Optional[mpl.axes.Axes]): axis to use for plotting,
            default `None`
        vertical_violins (Optional[bool]): flag to indicate orientation
        plot_kwargs (Optional[Union[Dict, List]]): if `Dict`, a dictionary
            of keyword-value pairs to pass to each plot routine.
            If `List`, it is a list of `Dict` objects to pass, one for
            each plot routine
        kde_kwargs (Optional[Dict]): keywords to pass to the
            `scipy.stats.gaussian_kde` constructor
    """
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()

    assert np.ndim(points) < 3
    points = np.atleast_2d(points)

    if isinstance(plot_kwargs, list):
        assert len(points) == len(plot_kwargs)

    if positions is not None:
        assert len(points) == len(positions)
    else:
        # Horizontal positions of the centers of the violins
        positions = np.arange(0, len(points))

    # Center positions between integers
    if vertical_violins:
        axis.set_xlim(positions.min() - 0.5, positions.max() + 0.5)
    else:
        axis.set_ylim(positions.min() - 0.5, positions.max() + 0.5)

    # Loop over all distributions and draw the violin
    for i, pi in zip(positions, points):
        mean = np.mean(pi)
        std = np.std(pi)
        interval = np.array([mean - 5 * std, mean + 5 * std])  # 5-sigma

        if isinstance(plot_kwargs, list):
            kwargs = plot_kwargs[i]
        else:
            kwargs = plot_kwargs

        # Create the KDE
        kde = gaussian_kde(pi, **kde_kwargs)

        # Make the domain and range
        x = np.linspace(min(interval), max(interval), 1000)
        y = kde(x)
        scale = 0.4 / y.max()
        # left side
        axis.plot(
            *_xy_order(x, i - y * scale, vertical_violins), **kwargs,
        )
        # right side
        axis.plot(
            *_xy_order(x, i + y * scale, vertical_violins), **kwargs,
        )

    return fig, axis


def point_violin():
    pass


def histogram_violin():
    pass
