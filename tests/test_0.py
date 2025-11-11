import pytest
import pandas as pd
import numpy as np
from pandas.api.types import is_integer_dtype, is_float_dtype

from definition_3e60c8abce164188bdbd197e06736a52 import load_training_dataset_from_csv


def test_basic_load_with_dtype_and_na(tmp_path):
    # Prepare CSV
    csv_content = "id,score,note\n1,10.5,ok\n2,NA,missing\n3,7.0,NA\n"
    file_path = tmp_path / "basic.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    # Parameters
    sep = ","
    encoding = "utf-8"
    dtype_map = {"id": "int64", "score": "float64", "note": "object"}
    na_values = ["NA"]

    df = load_training_dataset_from_csv(str(file_path), sep, encoding, dtype_map, na_values)

    assert df.shape == (3, 3)
    assert list(df.columns) == ["id", "score", "note"]
    assert is_integer_dtype(df["id"])
    assert is_float_dtype(df["score"])
    assert pd.isna(df.loc[1, "score"])  # 'NA' parsed as NaN
    assert pd.isna(df.loc[2, "note"])   # 'NA' parsed as NaN


def test_custom_separator_and_encoding_unicode(tmp_path):
    # Prepare CSV with semicolon separator and unicode text
    csv_content = "name;value\ncafé;42\n"
    file_path = tmp_path / "unicode.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    df = load_training_dataset_from_csv(str(file_path), sep=";", encoding="utf-8", dtype_map=None, na_values=None)

    assert df.shape == (1, 2)
    assert df.loc[0, "name"] == "café"
    assert df.loc[0, "value"] == 42


def test_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_training_dataset_from_csv("non_existent_file.csv", ",", "utf-8", None, None)


def test_invalid_dtype_map_raises(tmp_path):
    # Prepare simple CSV
    csv_content = "id\n1\n2\n"
    file_path = tmp_path / "invalid_dtype.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    # Invalid dtype should raise
    with pytest.raises((TypeError, ValueError)):
        load_training_dataset_from_csv(str(file_path), ",", "utf-8", {"id": "not_a_dtype"}, None)


def test_na_values_dict_per_column(tmp_path):
    # Prepare CSV with per-column NA markers
    csv_content = "age,income\nunknown,1000\n30,-\n40,2000\n"
    file_path = tmp_path / "na_per_column.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    na_values = {"age": ["unknown"], "income": ["-"]}
    dtype_map = {"age": "float64", "income": "float64"}

    df = load_training_dataset_from_csv(str(file_path), ",", "utf-8", dtype_map, na_values)

    assert df.shape == (3, 2)
    assert is_float_dtype(df["age"]) and is_float_dtype(df["income"])
    assert pd.isna(df.loc[0, "age"])
    assert pd.isna(df.loc[1, "income"])
    assert df.loc[2, "age"] == 40.0 and df.loc[2, "income"] == 2000.0