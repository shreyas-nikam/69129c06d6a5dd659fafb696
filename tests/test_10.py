import pytest
import pandas as pd
import numpy as np
from definition_28a9566de2954532b01607a974dee9a3 import compute_disparate_impact_ratio


def _get_group_row(res: pd.DataFrame, demographic_column: str, group):
    if demographic_column in res.columns:
        matches = res[res[demographic_column] == group]
        assert len(matches) == 1, "Result should have exactly one row per group"
        return matches.iloc[0]
    try:
        return res.loc[group]
    except KeyError:
        raise AssertionError(f"Group {group} not found in result index")


def _get_groups_in_result(res: pd.DataFrame, demographic_column: str):
    if demographic_column in res.columns:
        return set(res[demographic_column].tolist())
    return set(res.index.tolist())


def _find_di_column(res: pd.DataFrame, demographic_column: str, reference_group):
    ref_row = _get_group_row(res, demographic_column, reference_group)
    numeric_cols = [c for c in res.columns if pd.api.types.is_numeric_dtype(res[c])]
    candidates = [c for c in numeric_cols if np.isfinite(ref_row[c]) and np.isclose(ref_row[c], 1.0, atol=1e-8)]
    if not candidates:
        candidates = [c for c in numeric_cols if any(k in c.lower() for k in ["di", "ratio"])]
        if candidates:
            return candidates[0]
        raise AssertionError("Could not identify DI ratio column in results")
    preferred = [c for c in candidates if any(k in c.lower() for k in ["di", "ratio"])]
    return preferred[0] if preferred else candidates[0]


def test_compute_di_with_reference_group():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "B", "C", "C"],
        "outcome": [1, 1, 0, 1, 0, 0, 1, 0, 0],
    })
    res = compute_disparate_impact_ratio(df, "group", "outcome", 1, "A")
    assert isinstance(res, pd.DataFrame)

    # Ensure all groups are present
    res_groups = _get_groups_in_result(res, "group")
    assert {"A", "B", "C"}.issubset(res_groups)

    # Expected positive rates
    rate_A = 2 / 3
    rate_B = 2 / 4
    rate_C = 0 / 2

    di_col = _find_di_column(res, "group", "A")
    row_A = _get_group_row(res, "group", "A")
    row_B = _get_group_row(res, "group", "B")
    row_C = _get_group_row(res, "group", "C")

    assert np.isclose(row_A[di_col], 1.0, atol=1e-8)
    assert np.isclose(row_B[di_col], rate_B / rate_A, atol=1e-8)  # 0.75
    assert np.isclose(row_C[di_col], rate_C / rate_A, atol=1e-8)  # 0.0


def test_compute_di_with_none_reference_uses_largest_group():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "B", "C", "C"],
        "outcome": [1, 1, 0, 1, 0, 0, 1, 0, 0],
    })
    res = compute_disparate_impact_ratio(df, "group", "outcome", 1, None)
    largest_group = df["group"].value_counts().idxmax()  # "B"
    di_col = _find_di_column(res, "group", largest_group)

    row_largest = _get_group_row(res, "group", largest_group)
    assert np.isclose(row_largest[di_col], 1.0, atol=1e-8)

    # Check another group's DI relative to largest group
    rate_A = (df.loc[df.group == "A", "outcome"] == 1).mean()
    rate_B = (df.loc[df.group == "B", "outcome"] == 1).mean()
    row_A = _get_group_row(res, "group", "A")
    assert np.isclose(row_A[di_col], rate_A / rate_B, atol=1e-8)


def test_reference_group_not_found_raises():
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "outcome": [1, 0, 1, 1],
    })
    with pytest.raises((KeyError, ValueError, AssertionError)):
        compute_disparate_impact_ratio(df, "group", "outcome", 1, "NON_EXISTENT_GROUP")


def test_invalid_target_column_raises():
    df = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "outcome": [1, 0, 1, 1],
    })
    with pytest.raises((KeyError, ValueError, AttributeError)):
        compute_disparate_impact_ratio(df, "group", "bad_target_column", 1, "A")


def test_non_numeric_positive_label():
    df = pd.DataFrame({
        "grp": ["A", "A", "B", "B", "B"],
        "result": ["Y", "N", "Y", "N", "Y"],
    })
    # Rates: A=0.5, B=2/3 -> DI(B)=1.3333..., DI(A)=1
    res = compute_disparate_impact_ratio(df, "grp", "result", "Y", "A")
    assert isinstance(res, pd.DataFrame)

    di_col = _find_di_column(res, "grp", "A")
    row_A = _get_group_row(res, "grp", "A")
    row_B = _get_group_row(res, "grp", "B")

    assert np.isclose(row_A[di_col], 1.0, atol=1e-8)
    assert np.isclose(row_B[di_col], (2/3) / (1/2), atol=1e-8)
