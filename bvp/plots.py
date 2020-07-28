"""
Better violin plots than usual.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_discrete, rv_continuous, gaussian_kde, norm


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
    for xi, yi in zip(x, y):
        # left side
        if sides in ["both", "left", "top"]:
            xii, yii = _xy_order(
                [xi, xi], [index, index - yi * scale], vertical_violins
            )
            if inner == "stick":
                axis.plot(xii, yii, c="k", alpha=0.5)
        # right side
        if sides in ["both", "right", "bottom"]:
            xii, yii = _xy_order(
                [xi, xi], [index, index + yi * scale], vertical_violins
            )
            if inner == "stick":
                axis.plot(xii, yii, c="k", alpha=0.5)
    return


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


def analytic_violin(
    distributions: List,
    positions: Optional[List[int]] = None,
    axis: Optional["mpl.axes.Axes"] = None,
    vertical_violins: bool = True,
    sides: str = "both",
    plot_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = {
        "color": "black",
    },
    sigma: float = 5.0,
    interval: Optional[List] = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create violin plots of analytic distributions.

    .. note::

       can accept either discrete or continuous distributions.

    .. note::

       the default plot keywords are

       .. code-block:: python

          plot_kwargs = {
             "color": "black",
          }

    Args:
        distributions (List): analytic distributions
        positions (Optional[List[int]]): locations to plot the violins
        axis (mpl.axes.Axes): axis to use for plotting, default `None`
        vertical_violins (bool): flag to indicate orientation
        sides (str): string to indicate where to put the plot
        plot_kwargs (Dict or List): if Dict, a dictionary of keyword-value
            pairs to pass to each plot routine. If List, it is a list of
            Dict objects to pass, one for each plot routine
        sigma (float): symmetric sigma level to plot; mutually
            exclusiive with the `interval` argument
        interval (Optional[List[float]]): plotting interval; default `None`
    """
    fig, axis, positions = _preamble(
        distributions, axis, plot_kwargs, positions, vertical_violins, sides,
    )

    if sigma is not None and interval is not None:
        raise ValueError("`sigma` and `interval` are mutually exclusive")
    if sigma is not None:
        assert np.isscalar(sigma)
        normal_prob_interval = norm.cdf(sigma) - norm.cdf(-sigma)
    elif interval is not None:
        assert np.shape(interval) == (2,)
        assert interval[0] < interval[1]
        normal_prob_interval = None
    else:  # sigma and interval are None
        raise ValueError("one of `sigma` and `interval` must be specified")

    # Loop over all distributions and draw the violin
    for i, d in zip(positions, distributions):
        if normal_prob_interval is not None:
            interval = d.interval(normal_prob_interval)

        if isinstance(plot_kwargs, list):
            kwargs = plot_kwargs[i]
        else:
            kwargs = plot_kwargs

        # Handle continuous vs discrete cases differently
        if hasattr(d, "dist"):

            if isinstance(d.dist, rv_discrete):
                xs = np.arange(min(interval), max(interval) + 1)
                ys = d.pmf(xs)
                scale = 0.4 / ys.max()
                x = np.array([xs[0], xs[0], xs[0] + 1])
                y = np.array([0, ys[0] * scale, ys[0] * scale])
                for j in range(1, len(xs)):
                    x = np.hstack((x, [xs[j], xs[j] + 1]))
                    y = np.hstack((y, [ys[j] * scale, ys[j] * scale]))
                _plot_from_x_dist(
                    axis, x, y, i, kwargs, vertical_violins, sides
                )
            elif isinstance(d.dist, rv_continuous):
                x = np.linspace(min(interval), max(interval), 1000)
                y = d.pdf(x)
                _plot_from_x_dist(
                    axis, x, y, i, kwargs, vertical_violins, sides
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
    sides: str = "both",
    plot_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = {
        "color": "black",
    },
    kde_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = {
        "bw_method": "scott",
        "weights": None,
    },
    sigma: float = 5.0,
    interval: Optional[List] = None,
    inner: str = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create violin plots of Gaussian kernel density estimations (KDE)
    of points.

    .. note::

       the default plot keywords are

       .. code-block:: python

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
        sigma (Optional[float]): symmetric sigma level to plot; mutually
            exclusiive with the `interval` argument
        interval (Optional[List[float]]): plotting interval; default `None`
        inner (Optional[str]): Representation of the datapoints in the violin
            interior. If `quartiles`, draw the quartiles of the distribution.
            If `point` or `stick`, show each underlying datapoint. Using
            `None` will draw unadorned violins.
    """
    assert np.ndim(points) < 3
    points = np.atleast_2d(points)

    if inner is not None:
        assert inner in ["quartilse", "point", "stick"]

    fig, axis, positions = _preamble(
        points, axis, plot_kwargs, positions, vertical_violins, sides
    )

    if sigma is not None and interval is not None:
        raise ValueError("`sigma` and `interval` are mutually exclusive")
    if sigma is not None:
        assert np.isscalar(sigma)
        compute_interval = True
    elif interval is not None:
        assert np.shape(interval) == (2,)
        assert interval[0] < interval[1]
        compute_interval = False
    else:  # sigma and interval are None
        raise ValueError("one of `sigma` and `interval` must be specified")

    # Loop over all distributions and draw the violin
    for i, pi in zip(positions, points):
        mean = np.mean(pi)
        std = np.std(pi)
        if compute_interval:
            interval = np.array([mean - sigma * std, mean + sigma * std])

        if isinstance(plot_kwargs, list):
            kwargs = plot_kwargs[i]
        else:
            kwargs = plot_kwargs

        # Create the KDE
        kde = gaussian_kde(pi, **kde_kwargs)

        # Make the domain and range
        x = np.linspace(min(interval), max(interval), 1000)
        y = kde(x)
        _plot_from_x_dist(axis, x, y, i, kwargs, vertical_violins, sides)

        # Make the inner sticks
        # Note - to make inner=="quartile" work I need to reorganize
        # the logic
        if inner is not None:
            scale = 0.4 / y.max()
            _inner_from_x_and_kde(
                axis, pi, kde(pi), i, inner, scale, vertical_violins, sides
            )

    return fig, axis


def boxplot(
    points: Union[List, np.ndarray],
    positions: Optional[List[int]] = None,
    axis: Optional["mpl.axes.Axes"] = None,
    vertical_violins: bool = True,
    boxplot_kwargs: Dict[str, Dict[str, Any]] = {
        "boxprops": {"color": "black"},
    },
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Create boxplots of the points using the implementation
    from `matplotlib`.

    Args:
        points (List): samples of an unknown distribution or list of samples
        positions (Optional[List[int]]): locations to plot the violins
        axis (Optional[mpl.axes.Axes]): axis to use for plotting,
            default `None`
        vertical_violins (Optional[bool]): flag to indicate orientation
        boxplot_kwargs (Dict[str, Dict[str, Any]]): keyword-value pairs
            to pass to each of the artists in the boxplot.
            See `this SO <https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color>`_  # noqa: E501
            link for more information.
    """

    assert np.ndim(points) < 3
    points = np.atleast_2d(points)

    fig, axis, positions = _preamble(
        points, axis, None, positions, vertical_violins
    )

    axis.boxplot(
        points.T,
        positions=positions,
        vert=vertical_violins,
        patch_artist=True,
        **boxplot_kwargs,
    )
    return fig, axis


# May not implement this one
def histogram_violin():
    pass
