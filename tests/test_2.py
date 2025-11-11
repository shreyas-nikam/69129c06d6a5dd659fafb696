import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from definition_5af1c7630a874b839f8ab9b5d58bf3e4 import impute_missing_values_with_strategies


def test_mixed_strategies_imputation():
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 5.0],                  # mean -> (1+2+5)/3 = 2.666...
        "B": [10.0, np.nan, 30.0, 40.0],               # median -> 30.0
        "C": ["x", "y", np.nan, "y"],                  # most_frequent -> "y"
        "D": [np.nan, "b", np.nan, "d"],               # constant -> "unknown"
    })
    strategies = {"A": "mean", "B": "median", "C": "most_frequent", "D": "constant"}
    fill_values = {"D": "unknown"}

    imputed_df, report = impute_missing_values_with_strategies(df, strategies, fill_values)

    # Check imputed values
    assert pd.isna(imputed_df).sum().sum() == 0  # no NaNs remain
    assert imputed_df.loc[2, "A"] == pytest.approx((1 + 2 + 5) / 3)
    assert imputed_df.loc[1, "B"] == 30.0
    assert imputed_df.loc[2, "C"] == "y"
    assert imputed_df.loc[0, "D"] == "unknown" and imputed_df.loc[2, "D"] == "unknown"

    # Check report structure and contents
    assert isinstance(report, dict)
    for col in ["A", "B", "C", "D"]:
        assert col in report
        assert report[col].get("strategy") == strategies[col]
    assert report["A"].get("imputed_count") == 1
    assert report["B"].get("imputed_count") == 1
    assert report["C"].get("imputed_count") == 1
    assert report["D"].get("imputed_count") == 2
    # If implementation provides fill value details, validate them
    if "fill_value" in report["D"]:
        assert report["D"]["fill_value"] == "unknown"


def test_no_missing_values_returns_same_and_empty_or_zero_report():
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0],
        "B": ["x", "y", "z"],
    })
    strategies = {"A": "mean", "B": "most_frequent"}

    imputed_df, report = impute_missing_values_with_strategies(df, strategies, fill_values=None)

    assert_frame_equal(imputed_df, df)

    # Accept either an empty report or a report with zero imputed_count per column
    if report:
        assert isinstance(report, dict)
        for col in strategies:
            assert col in report
            assert report[col].get("strategy") == strategies[col]
            assert report[col].get("imputed_count") in (0, None)
    else:
        assert report == {}


def test_constant_without_fill_value_raises():
    df = pd.DataFrame({"A": [1, np.nan, 3]})
    strategies = {"A": "constant"}
    # No fill_values provided -> should raise an error
    try:
        imputed_missing = impute_missing_values_with_strategies(df, strategies, fill_values=None)
        pytest.fail(f"Expected an exception, got {imputed_missing}")
    except Exception as e:
        assert isinstance(e, (KeyError, ValueError))


def test_invalid_strategy_raises():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    strategies = {"A": "invalid_strategy_name"}
    try:
        impute_missing_values_with_strategies(df, strategies, fill_values=None)
        pytest.fail("Expected ValueError for invalid strategy.")
    except Exception as e:
        assert isinstance(e, ValueError)


def test_mean_on_non_numeric_raises():
    df = pd.DataFrame({"A": ["a", np.nan, "c"]})
    strategies = {"A": "mean"}
    try:
        impute_missing_values_with_strategies(df, strategies, fill_values=None)
        pytest.fail("Expected error when applying mean to non-numeric data.")
    except Exception as e:
        assert isinstance(e, (TypeError, ValueError))