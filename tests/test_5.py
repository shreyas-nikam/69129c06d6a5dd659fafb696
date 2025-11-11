import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from definition_897cf09a3ac6404e9ae6aef90151ed68 import plot_boxplot_outliers_matplotlib


def _get_fig_axes(ret):
    if isinstance(ret, Figure):
        fig = ret
        axes = fig.axes
    elif isinstance(ret, tuple) and len(ret) >= 1 and isinstance(ret[0], Figure):
        fig = ret[0]
        if len(ret) >= 2:
            ax = ret[1]
            if isinstance(ax, Axes):
                axes = [ax]
            else:
                try:
                    axes = list(ax)
                except Exception:
                    axes = fig.axes
        else:
            axes = fig.axes
    else:
        raise AssertionError("Expected a Matplotlib Figure or (Figure, Axes) return.")
    return fig, axes


def test_plot_boxplot_outliers_matplotlib_basic():
    np.random.seed(0)
    df = pd.DataFrame({
        "value": np.concatenate([np.random.normal(0, 1, 100), np.array([10, -9])])
    })
    figsize = (6, 4)

    ret = plot_boxplot_outliers_matplotlib(df, column="value", by=None, figsize=figsize)
    fig, axes = _get_fig_axes(ret)

    try:
        assert isinstance(fig, Figure)
        assert len(axes) >= 1
        w, h = fig.get_size_inches()
        assert pytest.approx(w) == figsize[0]
        assert pytest.approx(h) == figsize[1]
    finally:
        plt.close(fig)


def test_plot_boxplot_outliers_matplotlib_grouped():
    np.random.seed(1)
    df = pd.DataFrame({
        "value": np.random.randn(120),
        "group": np.repeat(list("ABCD"), 30)
    })
    figsize = (7, 3)

    ret = plot_boxplot_outliers_matplotlib(df, column="value", by="group", figsize=figsize)
    fig, axes = _get_fig_axes(ret)

    try:
        assert isinstance(fig, Figure)
        assert len(axes) >= 1
        w, h = fig.get_size_inches()
        assert pytest.approx(w) == figsize[0]
        assert pytest.approx(h) == figsize[1]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("df,column,by,figsize,expected_exc", [
    (pd.DataFrame({"value": [1, 2, 3]}), "not_a_column", None, (5, 4), (KeyError, ValueError)),
    (pd.DataFrame({"text": ["a", "b", "c"]}), "text", None, (5, 4), (TypeError, ValueError)),
    (pd.DataFrame({"value": [1, 2, 3]}), "value", None, "invalid_figsize", (TypeError, ValueError)),
])
def test_plot_boxplot_outliers_matplotlib_invalid_inputs(df, column, by, figsize, expected_exc):
    with pytest.raises(expected_exc):
        ret = plot_boxplot_outliers_matplotlib(df, column=column, by=by, figsize=figsize)
        # In case the function unexpectedly returns without error, close any figures to avoid leaks.
        try:
            fig, _ = _get_fig_axes(ret)
            plt.close(fig)
        except Exception:
            pass