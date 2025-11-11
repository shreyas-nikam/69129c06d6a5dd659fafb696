import pytest
import pandas as pd
import numpy as np
from definition_430fcd9b4e454343b8f8809fe6e08577 import compute_demographic_distribution

def _extract_series(res, kind):
    # kind: 'count' or 'proportion'
    import pandas as pd
    if isinstance(res, pd.Series):
        return res
    assert isinstance(res, pd.DataFrame), "Result must be a pandas Series or DataFrame"
    preferred_cols = {
        'proportion': ['proportion', 'proportions', 'prop', 'ratio', 'fraction', 'share', 'pct', 'percentage'],
        'count': ['count', 'counts', 'n', 'freq', 'frequency']
    }[kind]
    for col in preferred_cols:
        if col in res.columns:
            return res[col]
    # Heuristics by dtype
    if kind == 'proportion':
        float_cols = [c for c in res.columns if res[c].dtype.kind == 'f']
        if float_cols:
            return res[float_cols[0]]
        # If only one column, return it
        if res.shape[1] == 1:
            return res.iloc[:, 0]
        # Try sum-to-one heuristic
        for c in res.columns:
            try:
                if pytest.approx(res[c].dropna().sum(), rel=1e-6) == 1:
                    return res[c]
            except Exception:
                continue
    else:
        int_cols = [c for c in res.columns if res[c].dtype.kind in ('i', 'u')]
        if int_cols:
            return res[int_cols[0]]
        if res.shape[1] == 1:
            return res.iloc[:, 0]
    raise AssertionError(f"Could not determine appropriate column for {kind}")

def test_basic_counts():
    df = pd.DataFrame({'group': ['A', 'B', 'A', 'C', 'B', 'B']})
    res = compute_demographic_distribution(df, 'group', normalize=False)
    counts = _extract_series(res, kind='count')
    expected = {'B': 3, 'A': 2, 'C': 1}
    assert counts.to_dict() == expected

def test_normalize_true_proportions_sum_and_values():
    df = pd.DataFrame({'group': ['A', 'B', 'A', 'C', 'B', 'B']})
    res = compute_demographic_distribution(df, 'group', normalize=True)
    props = _extract_series(res, kind='proportion')
    expected = {'B': 3/6, 'A': 2/6, 'C': 1/6}
    # Compare values approximately
    for k, v in expected.items():
        assert props[k] == pytest.approx(v, rel=1e-9)
    # Sum to 1
    assert props.sum() == pytest.approx(1.0, rel=1e-9)

def test_missing_column_raises_keyerror():
    df = pd.DataFrame({'group': ['A', 'B', 'C']})
    with pytest.raises(KeyError):
        compute_demographic_distribution(df, 'nonexistent_column', normalize=False)

def test_empty_dataframe_returns_empty_counts():
    df = pd.DataFrame({'group': []})
    res = compute_demographic_distribution(df, 'group', normalize=False)
    series = _extract_series(res, kind='count')
    assert len(series) == 0

def test_all_missing_values_returns_empty_proportions():
    df = pd.DataFrame({'group': [None, np.nan, None]})
    res = compute_demographic_distribution(df, 'group', normalize=True)
    series = _extract_series(res, kind='proportion')
    assert len(series) == 0 or series.dropna().sum() == pytest.approx(0.0, rel=1e-9)