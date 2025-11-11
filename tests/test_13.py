import pytest
import numpy as np
import pandas as pd

plotly = pytest.importorskip("plotly")
from plotly import graph_objects as go

from definition_19b45d12c3ed48efaaaad8c76f1e5c20 import plot_interactive_correlation_matrix_plotly


def _get_heatmap_trace(fig):
    for tr in fig.data:
        if getattr(tr, "type", "").lower() in ("heatmap", "heatmapgl"):
            return tr
    return None


def test_returns_heatmap_with_correct_values_pearson():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 6, 8],
        "c": [4, 1, 0, -1],  # extra column to ensure selection works
    })
    cols = ["a", "b"]
    fig = plot_interactive_correlation_matrix_plotly(df, cols, method="pearson")
    assert isinstance(fig, go.Figure)

    trace = _get_heatmap_trace(fig)
    assert trace is not None, "No heatmap trace found in the returned figure."

    z = np.array(trace.z, dtype=float)
    assert z.shape == (2, 2)
    expected = df[cols].corr(method="pearson").values
    assert np.allclose(z, expected, equal_nan=True)


def test_single_column_produces_1x1_identity():
    df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
    cols = ["x"]
    fig = plot_interactive_correlation_matrix_plotly(df, cols, method="pearson")
    assert isinstance(fig, go.Figure)

    trace = _get_heatmap_trace(fig)
    assert trace is not None
    z = np.array(trace.z, dtype=float)
    assert z.shape == (1, 1)
    assert np.isclose(z[0, 0], 1.0, equal_nan=False)


def test_handles_nans_with_spearman():
    df = pd.DataFrame({
        "a": [1, np.nan, 3, 4, 5],
        "b": [2, 3, np.nan, 5, 6],
    })
    cols = ["a", "b"]
    method = "spearman"
    fig = plot_interactive_correlation_matrix_plotly(df, cols, method=method)
    assert isinstance(fig, go.Figure)

    trace = _get_heatmap_trace(fig)
    assert trace is not None
    z = np.array(trace.z, dtype=float)
    expected = df[cols].corr(method=method).values
    assert z.shape == expected.shape
    assert np.allclose(z, expected, equal_nan=True)


def test_invalid_method_raises_value_error():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    with pytest.raises(ValueError):
        plot_interactive_correlation_matrix_plotly(df, ["a", "b"], method="invalid")


def test_non_numeric_column_raises_value_error():
    df = pd.DataFrame({
        "a": [1.0, 2.5, 3.1, 4.2],
        "text": ["x", "y", "z", "w"],
    })
    with pytest.raises(ValueError):
        plot_interactive_correlation_matrix_plotly(df, ["a", "text"], method="pearson")