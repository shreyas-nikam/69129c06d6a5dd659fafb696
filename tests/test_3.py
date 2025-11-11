import pytest
import pandas as pd
from definition_14fd6d9f8d8144098b8e442fd4221f89 import standardize_categorical_values


def test_standardize_casefold_and_mapping_basic():
    df = pd.DataFrame({"Gender": ["Male", "FEMALE", "Unknown", None]})
    normalization_map = {"Gender": {"male": "M", "female": "F"}}

    out_df, applied_maps = standardize_categorical_values(
        df, categorical_columns=["Gender"], normalization_map=normalization_map, casefold=True
    )

    assert out_df["Gender"].tolist() == ["M", "F", "unknown", None]
    assert isinstance(applied_maps, dict)
    assert "Gender" in applied_maps
    assert applied_maps["Gender"].get("male") == "M"
    assert applied_maps["Gender"].get("female") == "F"


def test_standardize_no_casefold_case_sensitive_mapping():
    df = pd.DataFrame({"City": ["NY", "ny", "Ny"]})
    normalization_map = {"City": {"ny": "New York"}}

    out_df, applied_maps = standardize_categorical_values(
        df, categorical_columns=["City"], normalization_map=normalization_map, casefold=False
    )

    assert out_df["City"].tolist() == ["NY", "New York", "Ny"]
    assert "City" in applied_maps
    assert applied_maps["City"].get("ny") == "New York"


def test_standardize_map_none_casefold_only():
    df = pd.DataFrame({"Dept": ["Sales", "engineering", None]})

    out_df, applied_maps = standardize_categorical_values(
        df, categorical_columns=["Dept"], normalization_map=None, casefold=True
    )

    # Expect only case-folding when no map is provided
    assert out_df["Dept"].tolist() == ["sales", "engineering", None]
    assert isinstance(applied_maps, dict)
    assert "Dept" not in applied_maps or applied_maps.get("Dept") == {}


def test_standardize_raises_for_missing_column():
    df = pd.DataFrame({"Existing": ["A", "B"]})
    with pytest.raises((KeyError, ValueError)):
        standardize_categorical_values(
            df, categorical_columns=["Missing"], normalization_map=None, casefold=True
        )


def test_standardize_preserves_non_categorical_and_applied_maps():
    df = pd.DataFrame({"Gender": ["f", "M"], "Score": [1, 2]})
    normalization_map = {"Gender": {"f": "Female", "m": "Male"}}

    out_df, applied_maps = standardize_categorical_values(
        df, categorical_columns=["Gender"], normalization_map=normalization_map, casefold=True
    )

    assert out_df["Gender"].tolist() == ["Female", "Male"]
    assert out_df["Score"].tolist() == [1, 2]  # non-categorical column preserved
    pd.testing.assert_index_equal(out_df.index, df.index)

    assert "Gender" in applied_maps
    assert applied_maps["Gender"].get("f") == "Female"
    assert applied_maps["Gender"].get("m") == "Male"
