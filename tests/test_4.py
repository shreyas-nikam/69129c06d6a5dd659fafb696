import pytest
import pandas as pd
import numpy as np
from definition_676083cf055047d6aeb9018c97d75c0f import detect_outliers_iqr


def _extract_mask(item):
    # Accept several plausible return shapes: Series, dict with mask, or tuple/list containing a mask
    if isinstance(item, pd.Series) and item.dtype == bool:
        return item
    if isinstance(item, dict):
        for key in ("mask", "outliers", "outlier_mask"):
            val = item.get(key)
            if isinstance(val, pd.Series) and val.dtype == bool:
                return val
    if isinstance(item, (tuple, list)):
        for val in item:
            if isinstance(val, pd.Series) and val.dtype == bool:
                return val
    raise AssertionError("Could not extract boolean mask from result item")


def test_detect_outliers_basic():
    df = pd.DataFrame({
        "A": [1, 1, 1, 1, 100],   # clear upper outlier
        "B": [10, 12, 11, 10, 12] # no outliers
    })
    cols = ["A", "B"]
    result = detect_outliers_iqr(df, cols, 1.5)

    # Keys should match requested columns
    assert isinstance(result, dict)
    assert set(result.keys()) == set(cols)

    mask_a = _extract_mask(result["A"])
    mask_b = _extract_mask(result["B"])

    # Masks should align to df index and be boolean
    assert mask_a.index.equals(df.index) and mask_a.dtype == bool
    assert mask_b.index.equals(df.index) and mask_b.dtype == bool

    # 'A' flags only the last value as outlier
    assert mask_a.tolist() == [False, False, False, False, True]
    # 'B' should have no outliers
    assert mask_b.sum() == 0


@pytest.mark.parametrize("bad_columns,expected_exc", [
    (["C"], (KeyError, ValueError)),
    (["A", "C"], (KeyError, ValueError)),
])
def test_detect_outliers_invalid_column(bad_columns, expected_exc):
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(expected_exc):
        detect_outliers_iqr(df, bad_columns, 1.5)


def test_detect_outliers_non_numeric_column():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "C": ["x", "y", "z", "w"]  # non-numeric
    })
    with pytest.raises((TypeError, ValueError)):
        detect_outliers_iqr(df, ["A", "C"], 1.5)


@pytest.mark.parametrize("bad_multiplier", [-1, 0, "bad"])
def test_detect_outliers_invalid_multiplier(bad_multiplier):
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    with pytest.raises((TypeError, ValueError)):
        detect_outliers_iqr(df, ["A"], bad_multiplier)


def test_detect_outliers_empty_columns_returns_empty_mapping():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = detect_outliers_iqr(df, [], 1.5)
    assert isinstance(result, dict)
    assert len(result) == 0