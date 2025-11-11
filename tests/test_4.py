import pytest
import pandas as pd
import numpy as np
from definition_835680fa99034a97b3350f10e4d8a78c import detect_outliers_iqr


def _extract_mask(value):
    # Helper to normalize the return structure:
    # - Accepts pd.Series of bools
    # - Or a dict containing one of common keys for the mask
    # - Or a numpy array / list convertible to a boolean Series
    if isinstance(value, pd.Series):
        return value.astype(bool)
    if isinstance(value, dict):
        for key in ("mask", "is_outlier", "outliers", "flags"):
            if key in value:
                m = value[key]
                if isinstance(m, pd.Series):
                    return m.astype(bool)
                try:
                    return pd.Series(m, dtype=bool)
                except Exception:
                    pass
    if isinstance(value, (list, tuple, np.ndarray)):
        return pd.Series(value, dtype=bool)
    raise AssertionError("Unable to extract outlier mask from result value")


def test_detect_outliers_iqr_single_column_flags_expected_outlier():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 100]})
    res = detect_outliers_iqr(df, ["a"], 1.5)
    assert isinstance(res, dict)
    assert "a" in res
    mask = _extract_mask(res["a"])
    assert len(mask) == len(df)
    assert list(mask) == [False, False, False, False, True]


def test_detect_outliers_iqr_multiple_columns_mixed():
    df = pd.DataFrame({
        "a": [1, 1, 1, 1, 100],     # last is outlier
        "b": [10, 11, 12, 13, 14],  # no outliers
        "c": [0, 0, 1000, 0, 0],    # index 2 is outlier
    })
    res = detect_outliers_iqr(df, ["a", "b", "c"], 1.5)
    assert set(res.keys()) >= {"a", "b", "c"}

    mask_a = _extract_mask(res["a"])
    mask_b = _extract_mask(res["b"])
    mask_c = _extract_mask(res["c"])

    assert len(mask_a) == len(df) and len(mask_b) == len(df) and len(mask_c) == len(df)

    assert list(mask_a) == [False, False, False, False, True]
    assert list(mask_b) == [False, False, False, False, False]
    assert list(mask_c) == [False, False, True, False, False]


def test_detect_outliers_iqr_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    with pytest.raises((KeyError, ValueError)):
        detect_outliers_iqr(df, ["a", "missing"], 1.5)


def test_detect_outliers_iqr_non_numeric_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "s": ["x", "y", "z", "w", "v"]})
    with pytest.raises((TypeError, ValueError)):
        detect_outliers_iqr(df, ["a", "s"], 1.5)


@pytest.mark.parametrize("multiplier", [0, -1, None, "1.5"])
def test_detect_outliers_iqr_invalid_multiplier_raises(multiplier):
    df = pd.DataFrame({"a": [1, 1, 1, 1, 100]})
    with pytest.raises((TypeError, ValueError)):
        detect_outliers_iqr(df, ["a"], multiplier)
