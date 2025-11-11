import pytest
import pandas as pd
from definition_0c82fa16c47944ba99b4496040904c09 import load_training_dataset_from_csv


def test_load_training_dataset_from_csv_dtype_and_na(tmp_path):
    # Create a CSV with NA tokens and values requiring dtype enforcement
    p = tmp_path / "data.csv"
    p.write_text(
        "id,age,score\n"
        "1,25,88.5\n"
        "2,NA,91.0\n"
        "3,30,missing\n",
        encoding="utf-8",
    )

    df = load_training_dataset_from_csv(
        filepath=str(p),
        sep=",",
        encoding="utf-8",
        dtype_map={"id": "Int64", "age": "Int64", "score": "float64"},
        na_values=["NA", "missing"],
    )

    assert list(df.columns) == ["id", "age", "score"]
    assert df.shape == (3, 3)
    # Dtype enforcement
    assert str(df.dtypes["id"]) == "Int64"
    assert str(df.dtypes["age"]) == "Int64"
    assert str(df.dtypes["score"]) == "float64"
    # NA parsing
    assert pd.isna(df.loc[1, "age"])
    assert pd.isna(df.loc[2, "score"])


def test_load_training_dataset_from_csv_custom_sep_and_na_dict(tmp_path):
    p = tmp_path / "custom_sep.csv"
    p.write_text(
        "id;age;note\n"
        "10;N/A;hello\n"
        "11;42;world\n",
        encoding="utf-8",
    )

    df = load_training_dataset_from_csv(
        filepath=str(p),
        sep=";",
        encoding="utf-8",
        dtype_map={"id": "Int64", "age": "Int64"},
        na_values={"age": ["N/A"]},
    )

    assert df.shape == (2, 3)
    assert list(df.columns) == ["id", "age", "note"]
    assert str(df.dtypes["id"]) == "Int64"
    assert str(df.dtypes["age"]) == "Int64"
    assert pd.isna(df.loc[0, "age"])
    assert df.loc[1, "age"] == 42


def test_load_training_dataset_from_csv_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_training_dataset_from_csv(
            filepath=str(missing),
            sep=",",
            encoding="utf-8",
            dtype_map=None,
            na_values=None,
        )


def test_load_training_dataset_from_csv_invalid_sep_type(tmp_path):
    p = tmp_path / "invalid_sep.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_training_dataset_from_csv(
            filepath=str(p),
            sep=123,  # invalid sep type
            encoding="utf-8",
            dtype_map=None,
            na_values=None,
        )


def test_load_training_dataset_from_csv_dtype_conflict_raises(tmp_path):
    p = tmp_path / "dtype_conflict.csv"
    p.write_text(
        "id,age\n"
        "1,ok\n"
        "2,30\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_training_dataset_from_csv(
            filepath=str(p),
            sep=",",
            encoding="utf-8",
            dtype_map={"age": "Int64"},
            na_values=None,
        )