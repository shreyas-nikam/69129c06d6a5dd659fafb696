import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from definition_118287afed974b81b427e09767800788 import plot_histogram_distribution_matplotlib


def _assert_fig_axes(result):
    # Accept either a Figure alone or (Figure, Axes) / (Figure, axes-like)
    if isinstance(result, Figure):
        plt.close(result)
        return
    if isinstance(result, (tuple, list)) and len(result) >= 1 and isinstance(result[0], Figure):
        fig = result[0]
        plt.close(fig)
        return
    pytest.fail("Function should return a matplotlib Figure or (Figure, Axes).")


def test_valid_basic_histogram():
    df = pd.DataFrame({"value": [1, 2, 2, 3, 4, 5, 5, 6, 7, 8]})
    res = plot_histogram_distribution_matplotlib(
        df=df,
        column="value",
        bins=5,
        density=False,
        log_scale=False,
        figsize=(6, 4),
    )
    _assert_fig_axes(res)


def test_invalid_column_name():
    df = pd.DataFrame({"value": np.arange(10)})
    with pytest.raises((KeyError, ValueError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column="unknown_column",
            bins=10,
            density=False,
            log_scale=False,
            figsize=(6, 4),
        )


def test_non_numeric_column():
    df = pd.DataFrame({"category": ["a", "b", "c", "a", "b"]})
    with pytest.raises((TypeError, ValueError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column="category",
            bins=3,
            density=False,
            log_scale=False,
            figsize=(6, 4),
        )


def test_invalid_bins_value():
    df = pd.DataFrame({"value": np.random.randn(100)})
    with pytest.raises((TypeError, ValueError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column="value",
            bins=0,  # invalid: must be positive integer or valid bin spec
            density=False,
            log_scale=False,
            figsize=(6, 4),
        )


def test_invalid_density_type():
    df = pd.DataFrame({"value": np.random.randn(50)})
    with pytest.raises((TypeError, ValueError)):
        plot_histogram_distribution_matplotlib(
            df=df,
            column="value",
            bins=10,
            density="yes",  # invalid: should be boolean
            log_scale=False,
            figsize=(6, 4),
        )