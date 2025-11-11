import pytest
import pandas as pd
import numpy as np
from definition_a34d04be8d344dada98aaa38ba1bcdfd import impute_missing_values_with_strategies


def test_mean_and_median_imputation_numeric():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.nan],
        "b": [np.nan, 4.0, np.nan, 8.0],
    })
    strategies = {"a": "mean", "b": "median"}

    imputed_df, report = impute_missing_values_with_strategies(df, strategies, fill_values={})

    # Expected: mean of a = (1 + 3) / 2 = 2; median of b = (4 + 8) / 2 = 6
    expected_a = pd.Series([1.0, 2.0, 3.0, 2.0], name="a")
    expected_b = pd.Series([6.0, 4.0, 6.0, 8.0], name="b")

    pd.testing.assert_series_equal(imputed_df["a"], expected_a)
    pd.testing.assert_series_equal(imputed_df["b"], expected_b)

    assert isinstance(report, dict)
    assert set(strategies.keys()).issubset(set(report.keys()))


def test_most_frequent_imputation_categorical():
    df = pd.DataFrame({
        "cat": ["x", None, "y", "x", np.nan],
    })
    strategies = {"cat": "most_frequent"}

    imputed_df, report = impute_missing_values_with_strategies(df, strategies, fill_values={})

    # Mode should be 'x'
    assert imputed_df["cat"].isna().sum() == 0
    assert imputed_df.loc[1, "cat"] == "x"
    assert imputed_df.loc[4, "cat"] == "x"

    assert isinstance(report, dict)
    assert "cat" in report


def test_constant_imputation_with_fill_values():
    df = pd.DataFrame({
        "c": [np.nan, np.nan, "a"],
        "d": [1.0, np.nan, 2.0],
    })
    strategies = {"c": "constant", "d": "constant"}
    fill_values = {"c": "missing", "d": 0.0}

    imputed_df, report = impute_missing_values_with_strategies(df, strategies, fill_values=fill_values)

    expected_c = pd.Series(["missing", "missing", "a"], name="c", dtype=object)
    expected_d = pd.Series([1.0, 0.0, 2.0], name="d", dtype=float)

    pd.testing.assert_series_equal(imputed_df["c"], expected_c)
    pd.testing.assert_series_equal(imputed_df["d"], expected_d)

    assert isinstance(report, dict)
    assert set(strategies.keys()).issubset(set(report.keys()))


def test_constant_imputation_missing_fill_value_raises():
    df = pd.DataFrame({"x": [np.nan, 1]})
    strategies = {"x": "constant"}

    with pytest.raises((ValueError, KeyError, TypeError)):
        impute_missing_values_with_strategies(df, strategies, fill_values=None)


def test_mean_imputation_all_missing_raises():
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    strategies = {"a": "mean"}

    with pytest.raises((ValueError, ZeroDivisionError)):
        impute_missing_values_with_strategies(df, strategies, fill_values={})