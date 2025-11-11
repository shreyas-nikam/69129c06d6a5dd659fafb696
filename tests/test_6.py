import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from definition_bc5495e2366742ea8de1eff11e687f26 import plot_histogram_distribution_matplotlib


def _as_fig_ax(result):
    # Normalize return to (Figure, Axes)
    if isinstance(result, Figure):
        return result, result.gca()
    if isinstance(result, Axes):
        return result.figure, result
    if isinstance(result, tuple) and len(result) == 2:
        fig, ax = result
        # Handle possible ndarray of axes
        if not isinstance(fig, Figure):
            raise AssertionError("First element of tuple is not a Matplotlib Figure")
        if isinstance(ax, Axes):
            return fig, ax
        try:
            import numpy as _np
            if isinstance(ax, _np.ndarray) and ax.size > 0:
                ax0 = ax.flat[0]
                if isinstance(ax0, Axes):
                    return fig, ax0
        except Exception:
            pass
        raise AssertionError("Second element of tuple is not an Axes or array of Axes")
    raise AssertionError("Unexpected return type from function")


def test_basic_plot_figsize_and_logscale():
    df = pd.DataFrame({'data': np.random.randn(500)})
    result = plot_histogram_distribution_matplotlib(
        df=df,
        column='data',
        bins=20,
        density=False,
        log_scale=True,
        figsize=(7, 3)
    )
    fig, ax = _as_fig_ax(result)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_yscale() == 'log'
    w, h = fig.get_size_inches()
    assert w == pytest.approx(7, rel=0.05)
    assert h == pytest.approx(3, rel=0.05)
    plt.close(fig)


def test_bins_as_sequence_works():
    df = pd.DataFrame({'x': np.linspace(-1, 2, 50)})
    bins = [-1, 0, 1, 2]
    result = plot_histogram_distribution_matplotlib(
        df=df,
        column='x',
        bins=bins,
        density=True,
        log_scale=False,
        figsize=(4, 3)
    )
    fig, ax = _as_fig_ax(result)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_invalid_column_raises():
    df = pd.DataFrame({'x': np.arange(10)})
    with pytest.raises((KeyError, ValueError, TypeError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column='missing',
            bins=10,
            density=False,
            log_scale=False,
            figsize=(4, 3)
        )


def test_non_numeric_column_raises():
    df = pd.DataFrame({'label': list('abcdef')})
    with pytest.raises((TypeError, ValueError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column='label',
            bins=5,
            density=False,
            log_scale=False,
            figsize=(4, 3)
        )


def test_invalid_bins_raises():
    df = pd.DataFrame({'x': np.random.randn(20)})
    with pytest.raises((ValueError, TypeError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column='x',
            bins=-5,
            density=False,
            log_scale=False,
            figsize=(4, 3)
        )