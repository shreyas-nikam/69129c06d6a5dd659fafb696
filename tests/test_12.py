import pytest
import pandas as pd
import plotly.graph_objects as go
from definition_8c30488a02db49f699269264dc51318b import plot_interactive_scatter_plotly


def test_returns_plotly_figure_basic():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40],
        "income": [50000, 60000, 45000, 80000],
        "group": ["A", "B", "A", "B"],
        "weight": [1.0, 2.5, 1.2, 3.3],
        "id": [101, 102, 103, 104],
    })
    fig = plot_interactive_scatter_plotly(
        df=df,
        x="age",
        y="income",
        color="group",
        size="weight",
        hover_data=["id"],
        trendline=False,
    )
    assert isinstance(fig, go.Figure)


def test_missing_x_column_raises():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40],
        "income": [50000, 60000, 45000, 80000],
    })
    with pytest.raises((KeyError, ValueError)):
        plot_interactive_scatter_plotly(
            df=df,
            x="nonexistent",
            y="income",
            color=None,
            size=None,
            hover_data=None,
            trendline=False,
        )


def test_invalid_df_type_raises():
    with pytest.raises(TypeError):
        plot_interactive_scatter_plotly(
            df=123,  # not a DataFrame
            x="age",
            y="income",
            color=None,
            size=None,
            hover_data=None,
            trendline=False,
        )


def test_invalid_hover_data_raises():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40],
        "income": [50000, 60000, 45000, 80000],
        "group": ["A", "B", "A", "B"],
    })
    with pytest.raises((KeyError, ValueError, TypeError)):
        plot_interactive_scatter_plotly(
            df=df,
            x="age",
            y="income",
            color="group",
            size=None,
            hover_data=["missing_col"],
            trendline=False,
        )


def test_empty_dataframe_raises():
    df = pd.DataFrame(columns=["age", "income", "group", "weight", "id"])
    with pytest.raises(ValueError):
        plot_interactive_scatter_plotly(
            df=df,
            x="age",
            y="income",
            color="group",
            size="weight",
            hover_data=["id"],
            trendline=False,
        )