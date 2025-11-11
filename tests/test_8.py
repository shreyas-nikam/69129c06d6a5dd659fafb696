import pytest
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from definition_28055febe7e049e6b4e7b15b912884f4 import plot_demographic_pie_chart_matplotlib


def _get_all_texts(fig):
    texts = []
    for ax in fig.axes:
        texts.extend([t.get_text() for t in ax.texts])
    return texts


def _count_wedges(fig):
    count = 0
    for ax in fig.axes:
        count += sum(isinstance(p, Wedge) for p in ax.patches)
    return count


def test_returns_figure_and_figsize():
    df = pd.DataFrame({"group": ["A", "A", "B", "B", "C", "C", "C"]})
    figsize = (6, 4)
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="group",
        label_map=None,
        top_n=None,
        other_label="Other",
        figsize=figsize,
    )
    try:
        assert isinstance(fig, Figure)
        # Size may be float; allow small tolerance
        w, h = fig.get_size_inches()
        assert abs(w - figsize[0]) < 0.51 and abs(h - figsize[1]) < 0.51
        # Expect one wedge per unique category
        assert _count_wedges(fig) == df["group"].nunique()
    finally:
        plt.close(fig)


def test_top_n_aggregation_and_other_label():
    # A:5, B:4, C:3, D:2, E:1 -> top_n=2 => expect 3 slices (A, B, Other)
    df = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2 + ["E"]})
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="group",
        label_map=None,
        top_n=2,
        other_label="Other",
        figsize=(5, 5),
    )
    try:
        assert isinstance(fig, Figure)
        assert _count_wedges(fig) == 3
        texts = " ".join(_get_all_texts(fig))
        assert "Other" in texts
        assert "A" in texts and "B" in texts
    finally:
        plt.close(fig)


def test_label_map_applied():
    df = pd.DataFrame({"group": ["A", "A", "B", "B"]})
    label_map = {"A": "Alpha", "B": "Beta"}
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="group",
        label_map=label_map,
        top_n=None,
        other_label="Other",
        figsize=(4, 4),
    )
    try:
        texts = " ".join(_get_all_texts(fig))
        assert "Alpha" in texts and "Beta" in texts
        # Ensure original labels are not used when remapped
        assert "A" not in texts and "B" not in texts
    finally:
        plt.close(fig)


@pytest.mark.parametrize("bad_column, expected_exc", [
    ("missing", (KeyError, ValueError)),
    (None, (TypeError, ValueError, AttributeError)),
])
def test_invalid_demographic_column_raises(bad_column, expected_exc):
    df = pd.DataFrame({"group": ["A", "B", "C"]})
    with pytest.raises(expected_exc):
        plot_demographic_pie_chart_matplotlib(
            df=df,
            demographic_column=bad_column,
            label_map=None,
            top_n=None,
            other_label="Other",
            figsize=(4, 4),
        )


def test_empty_dataframe_raises():
    df = pd.DataFrame({"group": []})
    with pytest.raises(Exception):
        plot_demographic_pie_chart_matplotlib(
            df=df,
            demographic_column="group",
            label_map=None,
            top_n=None,
            other_label="Other",
            figsize=(4, 4),
        )