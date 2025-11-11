import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from definition_3bdee57304fe483d8798cf20f9ecfcef import standardize_categorical_values


def test_standardize_with_casefold_and_map():
    df = pd.DataFrame({
        "Gender": ["Male", "FEMALE", "f", "M", None],
        "Age": [25, 30, 22, 40, 28],
    })
    normalization_map = {
        "Gender": {"male": "Male", "female": "Female", "f": "Female", "m": "Male"}
    }

    df_out, applied_maps = standardize_categorical_values(
        df, categorical_columns=["Gender"], normalization_map=normalization_map, casefold=True
    )

    expected = pd.Series(["Male", "Female", "Female", "Male", np.nan], name="Gender", dtype="object")
    assert_series_equal(df_out["Gender"], expected, check_dtype=False)
    assert isinstance(applied_maps, dict)
    assert "Gender" in applied_maps
    assert isinstance(applied_maps["Gender"], dict)


def test_standardize_without_casefold_map_case_sensitive():
    df = pd.DataFrame({
        "Gender": ["Male", "FEMALE", "f", "M", None],
    })
    normalization_map = {
        "Gender": {"male": "Male", "female": "Female", "f": "Female", "m": "Male"}
    }

    df_out, applied_maps = standardize_categorical_values(
        df, categorical_columns=["Gender"], normalization_map=normalization_map, casefold=False
    )

    # Only "f" should map; others remain unchanged due to case sensitivity
    expected = pd.Series(["Male", "FEMALE", "Female", "M", np.nan], name="Gender", dtype="object")
    assert_series_equal(df_out["Gender"], expected, check_dtype=False)
    assert isinstance(applied_maps, dict)


def test_standardize_casefold_only_no_map():
    df = pd.DataFrame({
        "Ethnicity": ["Asian", "Hispanic", "UNKNOWN", None, 42],
    })

    df_out, applied_maps = standardize_categorical_values(
        df, categorical_columns=["Ethnicity"], normalization_map=None, casefold=True
    )

    # Only strings should be lowercased; non-strings and None should remain as-is
    expected = pd.Series(["asian", "hispanic", "unknown", np.nan, 42], name="Ethnicity", dtype="object")
    assert_series_equal(df_out["Ethnicity"], expected, check_dtype=False)
    assert isinstance(applied_maps, dict)
    assert "Ethnicity" in applied_maps
    assert isinstance(applied_maps["Ethnicity"], dict)


def test_standardize_invalid_inputs():
    with pytest.raises(TypeError):
        standardize_categorical_values(
            df=["not", "a", "dataframe"],
            categorical_columns=["col"],
            normalization_map=None,
            casefold=True
        )
    with pytest.raises(TypeError):
        standardize_categorical_values(
            df=pd.DataFrame({"col": ["a", "b"]}),
            categorical_columns="col",  # should be a list
            normalization_map=None,
            casefold=True
        )


def test_does_not_mutate_input_dataframe():
    df = pd.DataFrame({
        "City": ["NY", "ny", "SF", None],
        "Age": [21, 22, 23, 24],
    })
    df_before = df.copy(deep=True)

    normalization_map = {"City": {"ny": "New York", "sf": "San Francisco"}}
    df_out, _ = standardize_categorical_values(
        df, categorical_columns=["City"], normalization_map=normalization_map, casefold=True
    )

    # Input should remain unchanged
    assert df.equals(df_before)
    # Returned DataFrame should not be the same object
    assert df_out is not df
    # Output should reflect normalization
    expected_city = pd.Series(["New York", "New York", "San Francisco", np.nan], name="City", dtype="object")
    assert_series_equal(df_out["City"], expected_city, check_dtype=False)