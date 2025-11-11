import pytest
import pandas as pd

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from definition_d9aeff4c46ef45a681b071a39f3d07e2 import plot_interactive_scatter_plotly


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40],
        "group": ["A", "B", "A", "B"],
        "weight": [1.0, 2.5, 0.5, 3.0],
        "age": [23, 35, 29, 41],
    })


def test_returns_plotly_figure_with_expected_data(sample_df):
    fig = plot_interactive_scatter_plotly(
        df=sample_df, x="feature1", y="feature2",
        color=None, size=None, hover_data=["age"], trendline=False
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Validate that the first trace maps x and y correctly
    trace = fig.data[0]
    assert list(trace.x) == sample_df["feature1"].tolist()
    assert list(trace.y) == sample_df["feature2"].tolist()


def test_trendline_adds_additional_trace(sample_df):
    fig_no_trend = plot_interactive_scatter_plotly(
        df=sample_df, x="feature1", y="feature2",
        color=None, size=None, hover_data=None, trendline=False
    )
    fig_with_trend = plot_interactive_scatter_plotly(
        df=sample_df, x="feature1", y="feature2",
        color=None, size=None, hover_data=None, trendline=True
    )
    assert isinstance(fig_no_trend, go.Figure)
    assert isinstance(fig_with_trend, go.Figure)
    # Expect additional trace(s) for the trendline
    assert len(fig_with_trend.data) > len(fig_no_trend.data)


def test_missing_columns_raise(sample_df):
    with pytest.raises((KeyError, ValueError)):
        plot_interactive_scatter_plotly(
            df=sample_df, x="not_a_col", y="feature2",
            color=None, size=None, hover_data=None, trendline=False
        )


def test_non_numeric_xy_raises(sample_df):
    df_bad = sample_df.copy()
    df_bad["feature1"] = ["a", "b", "c", "d"]  # non-numeric x
    with pytest.raises((TypeError, ValueError)):
        plot_interactive_scatter_plotly(
            df=df_bad, x="feature1", y="feature2",
            color=None, size=None, hover_data=None, trendline=False
        )


def test_empty_dataframe_raises():
    empty_df = pd.DataFrame(columns=["feature1", "feature2", "group", "weight", "age"])
    with pytest.raises((ValueError, RuntimeError)):
        plot_interactive_scatter_plotly(
            df=empty_df, x="feature1", y="feature2",
            color=None, size=None, hover_data=None, trendline=False
        )