import pytest
import pandas as pd
import numpy as np
from definition_e8262e1a6bde42eb8ae93d1d461fcd02 import flag_statistical_parity_difference


def _get_group_row(out_df, group, demographic_column):
    # Try to access by column; else by index
    if demographic_column in out_df.columns:
        rows = out_df[out_df[demographic_column] == group]
        assert not rows.empty, f"Group {group} not found in output"
        return rows.iloc[0]
    else:
        try:
            return out_df.loc[group]
        except Exception as e:
            raise AssertionError(f"Could not locate group {group} in output by index") from e


def _get_col_name(out_df, candidates):
    for c in candidates:
        if c in out_df.columns:
            return c
    raise AssertionError(f"None of the expected columns found: {candidates}")


def test_spd_basic_two_groups_flagging():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "outcome": [1, 1, 1, 0, 1, 0, 0, 0],
    })
    threshold = 0.2
    out = flag_statistical_parity_difference(
        df, demographic_column="group", target_column="outcome", positive_label=1, threshold=threshold
    )

    # Expected
    overall = 4 / 8
    expected = {
        "A": {"rate": 0.75, "spd": 0.75 - overall, "flagged": True},
        "B": {"rate": 0.25, "spd": 0.25 - overall, "flagged": True},
    }

    rate_col = _get_col_name(out, ["selection_rate"])
    spd_col = _get_col_name(out, ["spd"])
    flagged_col = _get_col_name(out, ["flagged"])

    for g in ["A", "B"]:
        row = _get_group_row(out, g, "group")
        assert pytest.approx(row[rate_col], rel=1e-9) == expected[g]["rate"]
        assert pytest.approx(row[spd_col], rel=1e-9) == expected[g]["spd"]
        assert bool(row[flagged_col]) is expected[g]["flagged"]


def test_threshold_boundary_not_flagged():
    df = pd.DataFrame({
        "group": ["A", "A", "A,","A", "B", "B", "B", "B"]
    })
    # Fix accidental comma in group label
    df["group"] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    df["outcome"] = [1, 1, 1, 0, 1, 0, 0, 0]

    threshold = 0.25  # Exactly equals abs(SPD), should NOT flag if using "exceeds" semantics
    out = flag_statistical_parity_difference(
        df, demographic_column="group", target_column="outcome", positive_label=1, threshold=threshold
    )

    rate_col = _get_col_name(out, ["selection_rate"])
    spd_col = _get_col_name(out, ["spd"])
    flagged_col = _get_col_name(out, ["flagged"])

    # Validate presence and values at boundary
    for g in ["A", "B"]:
        row = _get_group_row(out, g, "group")
        assert isinstance(row[flagged_col], (bool, np.bool_))
        assert abs(row[spd_col]) == pytest.approx(0.25)
        assert bool(row[flagged_col]) is False


def test_non_numeric_positive_label_strings():
    df = pd.DataFrame({
        "sex": ["F", "F", "F", "M", "M", "M", "M"],
        "admit": ["yes", "no", "yes", "no", "no", "yes", "no"],
    })
    threshold = 0.2
    out = flag_statistical_parity_difference(
        df, demographic_column="sex", target_column="admit", positive_label="yes", threshold=threshold
    )

    overall = 3 / 7
    expected = {
        "F": {"rate": 2/3, "spd": (2/3) - overall, "flagged": True},   # ~0.238 > 0.2
        "M": {"rate": 1/4, "spd": (1/4) - overall, "flagged": False},  # ~-0.179
    }

    rate_col = _get_col_name(out, ["selection_rate"])
    spd_col = _get_col_name(out, ["spd"])
    flagged_col = _get_col_name(out, ["flagged"])

    for g in ["F", "M"]:
        row = _get_group_row(out, g, "sex")
        assert pytest.approx(row[rate_col], rel=1e-9) == expected[g]["rate"]
        assert pytest.approx(row[spd_col], rel=1e-9) == expected[g]["spd"]
        assert bool(row[flagged_col]) is expected[g]["flagged"]


def test_handles_missing_values_ignored_in_computation():
    df = pd.DataFrame({
        "demo": ["A", "B", "B", np.nan, np.nan],
        "y":    [1,   0,   1,   np.nan, 1],
    })
    # Clean rows are: (A,1), (B,0), (B,1)
    # rates: A=1.0, B=0.5, overall=2/3, spd: A≈0.333, B≈-0.167
    threshold = 0.3
    out = flag_statistical_parity_difference(
        df, demographic_column="demo", target_column="y", positive_label=1, threshold=threshold
    )

    rate_col = _get_col_name(out, ["selection_rate"])
    spd_col = _get_col_name(out, ["spd"])
    flagged_col = _get_col_name(out, ["flagged"])

    # Ensure only groups A and B are present
    present_groups = set(out["demo"].tolist()) if "demo" in out.columns else set(out.index.tolist())
    assert present_groups == {"A", "B"}

    rowA = _get_group_row(out, "A", "demo")
    rowB = _get_group_row(out, "B", "demo")

    assert pytest.approx(rowA[rate_col], rel=1e-9) == 1.0
    assert pytest.approx(rowB[rate_col], rel=1e-9) == 0.5
    assert bool(rowA[flagged_col]) is True  # 0.333... > 0.3
    assert bool(rowB[flagged_col]) is False


def test_invalid_demographic_column_raises():
    df = pd.DataFrame({
        "group": ["A", "B"],
        "outcome": [1, 0],
    })
    with pytest.raises(Exception):
        flag_statistical_parity_difference(
            df, demographic_column="nonexistent", target_column="outcome", positive_label=1, threshold=0.1
        )