import pytest
import pandas as pd
from definition_c68bfcb2a2d84442ac2a2e9bd47a201f import detect_representation_bias_by_threshold


def _get_group_row(df_out, demographic_column, group_val):
    # Try index-based lookup
    if df_out.index.name == demographic_column or (df_out.index.name is None and group_val in df_out.index):
        return df_out.loc[group_val]
    # Try column-based lookup
    if demographic_column in df_out.columns:
        rows = df_out[df_out[demographic_column] == group_val]
        assert len(rows) == 1, "Expected exactly one row per group"
        return rows.iloc[0]
    # Fallback: if exactly one non-numeric column exists, treat as label column
    non_numeric_cols = [c for c in df_out.columns if not pd.api.types.is_numeric_dtype(df_out[c])]
    if non_numeric_cols:
        col = non_numeric_cols[0]
        rows = df_out[df_out[col] == group_val]
        assert len(rows) == 1, "Expected exactly one row per group"
        return rows.iloc[0]
    raise AssertionError("Could not identify group label column or index")


def _find_columns(df_out):
    cols = df_out.columns
    # Identify count column
    count_candidates = [c for c in cols if "count" in c.lower()]
    count_col = count_candidates[0] if count_candidates else None
    # Identify share column
    share_candidates = [c for c in cols if "share" in c.lower()]
    share_col = share_candidates[0] if share_candidates else None
    # Identify boolean flag columns
    bool_cols = [c for c in cols if df_out[c].dtype == bool]
    under_share_candidates = [c for c in bool_cols if "share" in c.lower()]
    under_count_candidates = [c for c in bool_cols if "count" in c.lower()]
    under_share_col = under_share_candidates[0] if under_share_candidates else None
    under_count_col = under_count_candidates[0] if under_count_candidates else None
    return count_col, share_col, under_share_col, under_count_col


def test_detect_representation_basic():
    df = pd.DataFrame({"demo": ["A"] * 5 + ["B"] * 2 + ["C"] * 3})
    out = detect_representation_bias_by_threshold(df, "demo", min_share=0.25, min_count=3)

    assert isinstance(out, pd.DataFrame)
    count_col, share_col, under_share_col, under_count_col = _find_columns(out)
    assert count_col is not None, "Count column not found"
    assert share_col is not None, "Share column not found"
    assert under_share_col is not None and under_count_col is not None, "Underrepresentation flag columns not found"

    # Shares should sum to ~1.0
    assert pytest.approx(out[share_col].sum(), rel=1e-9, abs=1e-9) == 1.0

    # Validate each group's metrics and flags
    for grp, exp_count, exp_share, exp_under_share, exp_under_count in [
        ("A", 5, 0.5, False, False),
        ("B", 2, 0.2, True, True),
        ("C", 3, 0.3, False, False),
    ]:
        row = _get_group_row(out, "demo", grp)
        assert row[count_col] == exp_count
        assert pytest.approx(float(row[share_col]), rel=1e-9, abs=1e-9) == exp_share
        assert bool(row[under_share_col]) is exp_under_share
        assert bool(row[under_count_col]) is exp_under_count


def test_detect_representation_threshold_boundaries_equal_not_flagged():
    # Y has share exactly 0.2 and count exactly 1
    df = pd.DataFrame({"demo": ["X"] * 4 + ["Y"] * 1})
    out = detect_representation_bias_by_threshold(df, "demo", min_share=0.2, min_count=1)

    count_col, share_col, under_share_col, under_count_col = _find_columns(out)
    assert all([count_col, share_col, under_share_col, under_count_col]), "Expected required columns"

    y = _get_group_row(out, "demo", "Y")
    assert pytest.approx(float(y[share_col]), rel=1e-9, abs=1e-9) == 0.2
    assert y[count_col] == 1
    # Equal to threshold should not be flagged
    assert bool(y[under_share_col]) is False
    assert bool(y[under_count_col]) is False


def test_detect_representation_missing_column_raises():
    df = pd.DataFrame({"group": ["A", "B", "A"]})
    with pytest.raises(KeyError):
        detect_representation_bias_by_threshold(df, "demo", min_share=0.2, min_count=1)


def test_detect_representation_empty_df():
    df = pd.DataFrame({"demo": []})
    out = detect_representation_bias_by_threshold(df, "demo", min_share=0.2, min_count=1)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 0  # no groups to report


def test_detect_representation_invalid_min_share_raises():
    df = pd.DataFrame({"demo": ["A", "B", "A", "C"]})
    with pytest.raises(ValueError):
        detect_representation_bias_by_threshold(df, "demo", min_share=1.5, min_count=1)