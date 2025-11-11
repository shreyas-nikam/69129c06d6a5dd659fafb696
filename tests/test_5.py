import pytest
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from definition_3d56a688915b4bc58dbefda75b154a30 import plot_boxplot_outliers_matplotlib


def _unpack_result(result):
    # Accept either fig or (fig, ax)
    if isinstance(result, tuple):
        fig = result[0]
        ax = result[1] if len(result) > 1 else (result[0].axes[0] if result[0].axes else None)
    else:
        fig = result
        ax = fig.axes[0] if fig.axes else None
    return fig, ax


@pytest.mark.parametrize(
    "df,column,by,figsize,expected_num_groups",
    [
        # Basic, no grouping
        (pd.DataFrame({"value": [1, 2, 3, 4, 5]}), "value", None, (7, 5), None),
        # Grouped by categorical column with 3 groups
        (
            pd.DataFrame({
                "value": [1, 2, 3, 4, 5, 6],
                "group": pd.Categorical(["A", "A", "B", "B", "C", "C"])
            }),
            "value",
            "group",
            (6, 4),
            3
        ),
    ],
)
def test_plot_boxplot_success(df, column, by, figsize, expected_num_groups):
    fig = None
    try:
        result = plot_boxplot_outliers_matplotlib(df, column, by, figsize)
        fig, ax = _unpack_result(result)

        assert isinstance(fig, Figure)
        assert ax is not None
        assert ax.has_data()

        # Check figsize applied (approx to account for float conversions)
        w, h = fig.get_size_inches()
        assert pytest.approx(w) == figsize[0]
        assert pytest.approx(h) == figsize[1]

        # If grouped, ensure tick labels match number of groups
        if expected_num_groups is not None:
            fig.canvas.draw()
            labels = [tick.get_text() for tick in ax.get_xticklabels() if tick.get_text()]
            assert len(set(labels)) == expected_num_groups
    finally:
        if fig is not None:
            plt.close(fig)


@pytest.mark.parametrize(
    "df,column,by,figsize,expected_exc",
    [
        # Missing numeric column
        (pd.DataFrame({"x": [1, 2, 3]}), "value", None, (6, 4), (KeyError, ValueError)),
        # 'by' column does not exist
        (pd.DataFrame({"value": [1, 2, 3]}), "value", "missing_group", (6, 4), (KeyError, ValueError)),
        # Non-numeric data in column
        (pd.DataFrame({"value": ["a", "b", "c"]}), "value", None, (6, 4), (TypeError, ValueError)),
    ],
)
def test_plot_boxplot_errors(df, column, by, figsize, expected_exc):
    with pytest.raises(expected_exc):
        result = plot_boxplot_outliers_matplotlib(df, column, by, figsize)
        # If it returns a fig despite error expectation, close it to avoid resource leak
        try:
            fig, _ = _unpack_result(result)
            if isinstance(fig, Figure):
                plt.close(fig)
        except Exception:
            pass