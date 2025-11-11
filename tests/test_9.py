import pytest
import pandas as pd
from definition_20d7b7638d834f919eaaf99c56fdaefb import detect_representation_bias_by_threshold


def _get_group_row(result_df, group_col, group_value):
    if group_col in result_df.columns:
        matches = result_df[result_df[group_col] == group_value]
        assert not matches.empty, f"Group {group_value} not found in result."
        return matches.iloc[0]
    # else assume groups are index labels
    try:
        return result_df.loc[group_value]
    except Exception as e:
        raise AssertionError("Result must expose demographic groups either as a column or as the index.") from e


def test_detect_representation_bias_basic():
    # Prepare sample data with clear counts and shares
    df = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
    min_share = 0.25  # 25%
    min_count = 25

    result = detect_representation_bias_by_threshold(df, "group", min_share, min_count)

    # Basic structure checks
    assert isinstance(result, pd.DataFrame)
    assert set(["count", "share", "underrepresented_by_share", "underrepresented_by_count"]).issubset(result.columns), \
        "Result must include required columns."

    # There should be 3 groups
    assert len(result) == 3

    # Shares should sum approximately to 1.0
    assert pytest.approx(float(result["share"].sum()), rel=1e-9) == 1.0

    # Check per-group metrics and flags
    row_a = _get_group_row(result, "group", "A")
    row_b = _get_group_row(result, "group", "B")
    row_c = _get_group_row(result, "group", "C")

    # Counts
    assert int(row_a["count"]) == 50
    assert int(row_b["count"]) == 30
    assert int(row_c["count"]) == 20

    # Shares
    assert pytest.approx(float(row_a["share"]), rel=1e-9) == 0.50
    assert pytest.approx(float(row_b["share"]), rel=1e-9) == 0.30
    assert pytest.approx(float(row_c["share"]), rel=1e-9) == 0.20

    # Flags: only C should be underrepresented by both share and count
    assert bool(row_a["underrepresented_by_share"]) is False
    assert bool(row_a["underrepresented_by_count"]) is False

    assert bool(row_b["underrepresented_by_share"]) is False
    assert bool(row_b["underrepresented_by_count"]) is False

    assert bool(row_c["underrepresented_by_share"]) is True
    assert bool(row_c["underrepresented_by_count"]) is True


@pytest.mark.parametrize("bad_min_share", [-0.1, 1.1])
def test_invalid_min_share_raises_value_error(bad_min_share):
    df = pd.DataFrame({"group": ["A", "A", "B"]})
    with pytest.raises(ValueError):
        detect_representation_bias_by_threshold(df, "group", bad_min_share, 1)


def test_negative_min_count_raises_value_error():
    df = pd.DataFrame({"group": ["A", "B"]})
    with pytest.raises(ValueError):
        detect_representation_bias_by_threshold(df, "group", 0.2, -5)


def test_missing_demographic_column_raises():
    df = pd.DataFrame({"not_group": ["A", "B"]})
    with pytest.raises((KeyError, ValueError)):
        detect_representation_bias_by_threshold(df, "group", 0.2, 1)