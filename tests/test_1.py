import pytest
import pandas as pd
import numpy as np
from definition_405a56644c2f4c0b86cb54c87599f64b import generate_dataset_overview


def _normalize_mapping_like(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise AssertionError(f"Expected dict-like object, got {type(obj)}")


def _normalize_dtypes(obj):
    m = _normalize_mapping_like(obj)
    return {k: str(v) for k, v in m.items()}


def _is_empty_stats(obj):
    if obj is None:
        return True
    try:
        import pandas as pd  # local import for type checking
        if isinstance(obj, pd.DataFrame):
            return obj.empty
    except Exception:
        pass
    if isinstance(obj, dict):
        return len(obj) == 0
    if hasattr(obj, "size"):
        return getattr(obj, "size") == 0
    return False


def test_basic_structure_and_content_with_object_stats():
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 22, 40],
        "income": [50000, 60000, 55000, None, 70000],
        "cat": ["A", "B", "A", "A", None],
    })
    result = generate_dataset_overview(df, include_object_stats=True)

    assert isinstance(result, dict)

    # Required keys
    for key in ("dtypes", "non_null_counts", "memory_usage", "numeric_stats"):
        assert key in result, f"Missing key: {key}"

    # dtypes matches df.dtypes
    expected_dtypes = {k: str(v) for k, v in df.dtypes.items()}
    got_dtypes = _normalize_dtypes(result["dtypes"])
    assert got_dtypes == expected_dtypes

    # non-null counts correct
    expected_non_null = df.notna().sum().to_dict()
    got_non_null = _normalize_mapping_like(result["non_null_counts"])
    assert got_non_null == expected_non_null

    # memory usage is a reasonable number: accept shallow or deep or any between
    mem = result["memory_usage"]
    assert isinstance(mem, (int, float, np.integer, np.floating))
    shallow = int(df.memory_usage().sum())
    deep = int(df.memory_usage(deep=True).sum())
    lo, hi = min(shallow, deep), max(shallow, deep)
    assert lo <= int(mem) <= hi

    # numeric_stats covers numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stats = result["numeric_stats"]
    if isinstance(numeric_stats, pd.DataFrame):
        # Expect common describe indices
        assert {"count", "mean"}.issubset(set(map(str, numeric_stats.index)))
        assert set(numeric_cols).issubset(set(numeric_stats.columns))
    elif isinstance(numeric_stats, dict):
        assert set(numeric_cols).issubset(set(numeric_stats.keys()))
    else:
        pytest.fail(f"Unexpected type for numeric_stats: {type(numeric_stats)}")

    # object_stats present and references object column(s)
    assert "object_stats" in result
    object_stats = result["object_stats"]
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if isinstance(object_stats, pd.DataFrame):
        # describe(include=['object']) yields columns as object columns
        for c in obj_cols:
            assert c in object_stats.columns
        # Common indices: unique, top, freq
        assert {"unique", "top", "freq"}.issubset(set(map(str, object_stats.index)))
    elif isinstance(object_stats, dict):
        for c in obj_cols:
            assert c in object_stats
    else:
        pytest.fail(f"Unexpected type for object_stats: {type(object_stats)}")


def test_object_stats_excluded_when_flag_false():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "a"],
    })
    result = generate_dataset_overview(df, include_object_stats=False)
    assert isinstance(result, dict)
    assert "object_stats" not in result


def test_no_numeric_columns_numeric_stats_empty():
    df = pd.DataFrame({
        "a": ["x", "y", "z"],
        "b": ["u", None, "v"],
    })
    result = generate_dataset_overview(df, include_object_stats=True)
    assert "numeric_stats" in result
    assert _is_empty_stats(result["numeric_stats"])


def test_empty_dataframe_handling():
    df = pd.DataFrame({"a": pd.Series(dtype="float"), "cat": pd.Series(dtype="object")})
    result = generate_dataset_overview(df, include_object_stats=True)

    # non-null counts should be zeros
    non_null = _normalize_mapping_like(result["non_null_counts"])
    assert non_null == {"a": 0, "cat": 0}

    # dtypes preserved
    got_dtypes = _normalize_dtypes(result["dtypes"])
    assert got_dtypes == {"a": "float64", "cat": "object"}

    # object_stats may be empty but should exist since flag is True
    assert "object_stats" in result
    assert _is_empty_stats(result["object_stats"]) or (
        isinstance(result["object_stats"], pd.DataFrame) and "cat" in result["object_stats"].columns
    )


def test_invalid_input_type_raises():
    with pytest.raises((TypeError, ValueError)):
        generate_dataset_overview(["not", "a", "dataframe"], include_object_stats=True)
