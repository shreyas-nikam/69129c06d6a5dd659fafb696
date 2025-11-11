import pytest
import pandas as pd
import numpy as np
from definition_09dc81602dbe43348e13c142cc1435a0 import compute_demographic_distribution

def _get_group_labels(result, demographic_column):
    if isinstance(result, pd.Series):
        return set(result.index.tolist())
    elif isinstance(result, pd.DataFrame):
        if demographic_column in result.columns:
            return set(result[demographic_column].tolist())
        else:
            return set(result.index.tolist())
    return set()

def _extract_numeric_series_with_sum(result, target_sum, demographic_column, tol=1e-8):
    if isinstance(result, pd.Series):
        if pd.api.types.is_numeric_dtype(result) and np.isfinite(result.sum()) and np.isclose(result.sum(), target_sum, atol=tol):
            return result
        return None
    elif isinstance(result, pd.DataFrame):
        if demographic_column in result.columns:
            idx = result[demographic_column]
        else:
            idx = result.index
        numeric_cols = result.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            series = pd.Series(result[col].to_numpy(), index=idx)
            s = series.sum()
            if np.isfinite(s) and np.isclose(s, target_sum, atol=tol):
                return series
    return None

def _extract_proportions_series(result, demographic_column, tol=1e-8):
    # Finds a numeric vector whose sum is ~1 and values are in [0,1]
    if isinstance(result, pd.Series):
        s = result.sum()
        if pd.api.types.is_numeric_dtype(result) and np.isfinite(s) and np.isclose(s, 1.0, atol=tol) and ((result >= -tol) & (result <= 1 + tol)).all():
            return result
        return None
    elif isinstance(result, pd.DataFrame):
        if demographic_column in result.columns:
            idx = result[demographic_column]
        else:
            idx = result.index
        for col in result.select_dtypes(include=np.number).columns:
            series = pd.Series(result[col].to_numpy(), index=idx)
            s = series.sum()
            if np.isfinite(s) and np.isclose(s, 1.0, atol=tol) and ((series >= -tol) & (series <= 1 + tol)).all():
                return series
    return None

def test_counts_basic():
    df = pd.DataFrame({"group": ["A", "B", "A", "C", "B", "B"]})
    res = compute_demographic_distribution(df, "group", normalize=False)
    expected_counts = {"A": 2, "B": 3, "C": 1}
    total = len(df)

    # Groups present
    assert _get_group_labels(res, "group") == set(expected_counts.keys())

    # Verify counts via a numeric vector summing to total
    counts_series = _extract_numeric_series_with_sum(res, target_sum=total, demographic_column="group")
    assert counts_series is not None, "Expected a counts vector summing to total"
    for k, v in expected_counts.items():
        assert counts_series.loc[k] == v

def test_proportions_basic():
    df = pd.DataFrame({"group": ["A", "B", "A", "C", "B", "B"]})
    res = compute_demographic_distribution(df, "group", normalize=True)
    counts = {"A": 2, "B": 3, "C": 1}
    total = sum(counts.values())
    expected_props = {k: v / total for k, v in counts.items()}

    # Groups present
    assert _get_group_labels(res, "group") == set(expected_props.keys())

    # Verify proportions via a numeric vector summing to 1
    prop_series = _extract_proportions_series(res, demographic_column="group")
    assert prop_series is not None, "Expected a proportions vector summing to 1"
    for k, v in expected_props.items():
        assert np.isclose(prop_series.loc[k], v, atol=1e-8)

def test_invalid_column_raises():
    df = pd.DataFrame({"group": ["A", "B", "A"]})
    with pytest.raises((KeyError, ValueError, AttributeError, TypeError)):
        compute_demographic_distribution(df, "missing_column", normalize=False)

@pytest.mark.parametrize("normalize", [False, True])
def test_empty_dataframe_returns_empty(normalize):
    df = pd.DataFrame({"group": pd.Series(dtype="object")})
    res = compute_demographic_distribution(df, "group", normalize=normalize)
    if isinstance(res, pd.Series):
        assert res.empty
    elif isinstance(res, pd.DataFrame):
        assert res.empty
        assert len(res) == 0
    else:
        pytest.fail("Function should return a Pandas Series or DataFrame.")

def test_single_group_proportions():
    df = pd.DataFrame({"group": ["A"] * 5})
    res = compute_demographic_distribution(df, "group", normalize=True)

    # Validate groups
    assert _get_group_labels(res, "group") == {"A"}

    # Extract proportions and validate it's 1.0 for 'A'
    prop_series = _extract_proportions_series(res, demographic_column="group")
    assert prop_series is not None, "Expected a proportions vector"
    assert np.isclose(prop_series.loc["A"], 1.0, atol=1e-12)
    assert np.isclose(prop_series.sum(), 1.0, atol=1e-12)