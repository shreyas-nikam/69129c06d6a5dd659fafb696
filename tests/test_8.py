import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from definition_fff2865cb1704e45b1733a677c2a29d3 import plot_demographic_pie_chart_matplotlib


def _get_wedge_count(fig: Figure) -> int:
    axes = fig.get_axes()
    assert len(axes) >= 1, "Expected at least one Axes in the returned Figure."
    ax = axes[0]
    return sum(1 for p in ax.patches if p.__class__.__name__.lower().endswith("wedge"))


def _get_all_label_texts(fig: Figure) -> set:
    texts = set()
    ax = fig.get_axes()[0]
    texts.update(t.get_text() for t in ax.texts if t.get_text())
    leg = ax.get_legend()
    if leg is not None:
        texts.update(t.get_text() for t in leg.get_texts() if t.get_text())
    return texts


def test_basic_wedge_count_no_topn():
    df = pd.DataFrame({"demo": ["A", "B", "B", "C", "C", "C"]})
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="demo",
        label_map=None,
        top_n=None,
        other_label="Other",
        figsize=(6, 4),
    )
    try:
        assert isinstance(fig, Figure)
        assert _get_wedge_count(fig) == 3
    finally:
        plt.close(fig)


def test_topn_aggregation_creates_other_wedge():
    df = pd.DataFrame({"demo": ["A", "B", "B", "C", "C", "C", "D"]})
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="demo",
        label_map=None,
        top_n=2,
        other_label="Other",
        figsize=(6, 4),
    )
    try:
        # Expect top 2 categories + 1 "Other"
        assert _get_wedge_count(fig) == 3
        # "Other" label should appear in either texts or legend
        labels = _get_all_label_texts(fig)
        assert any(lbl.strip().lower() == "other" for lbl in labels)
    finally:
        plt.close(fig)


def test_nan_values_are_ignored_in_counts():
    df = pd.DataFrame({"demo": ["A", np.nan, "B", "B", None]})
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="demo",
        label_map=None,
        top_n=None,
        other_label="Other",
        figsize=(6, 4),
    )
    try:
        # Only "A" and "B" should be counted => 2 wedges
        assert _get_wedge_count(fig) == 2
    finally:
        plt.close(fig)


def test_missing_column_raises_keyerror():
    df = pd.DataFrame({"not_demo": ["A", "B"]})
    with pytest.raises(KeyError):
        plot_demographic_pie_chart_matplotlib(
            df=df,
            demographic_column="demo",
            label_map=None,
            top_n=None,
            other_label="Other",
            figsize=(6, 4),
        )


def test_figsize_applied():
    df = pd.DataFrame({"demo": ["A", "A", "B"]})
    target_size = (7, 4)
    fig = plot_demographic_pie_chart_matplotlib(
        df=df,
        demographic_column="demo",
        label_map=None,
        top_n=None,
        other_label="Other",
        figsize=target_size,
    )
    try:
        assert isinstance(fig, Figure)
        width, height = fig.get_size_inches()
        assert pytest.approx(width, rel=1e-6, abs=1e-6) == target_size[0]
        assert pytest.approx(height, rel=1e-6, abs=1e-6) == target_size[1]
    finally:
        plt.close(fig)