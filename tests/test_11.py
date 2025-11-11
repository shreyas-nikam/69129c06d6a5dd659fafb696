import pytest
import pandas as pd
from definition_2bc00c1aefd449db9bc56dbe4d4f0c4c import flag_statistical_parity_difference


def test_equal_rates_no_flags():
    # Two groups with identical selection rates -> SPD should be ~0 and no flags
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'target': [1, 0, 1, 0],
    })
    res = flag_statistical_parity_difference(
        df=df,
        demographic_column='group',
        target_column='target',
        positive_label=1,
        threshold=0.01
    )

    # Structure checks
    assert hasattr(res, 'columns')
    assert set(['selection_rate', 'spd', 'flagged']).issubset(res.columns)
    assert len(res) == df['group'].nunique()

    # Value checks
    assert res['selection_rate'].nunique() == 1
    assert res['selection_rate'].unique()[0] == pytest.approx(0.5)
    assert res['spd'].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert (~res['flagged']).all()


def test_spd_difference_invariant_and_flag_presence():
    # Two groups with different selection rates (0.8 and 0.2)
    # The difference between max and min SPD equals the difference in selection rates (invariant to reference choice)
    df = pd.DataFrame({
        'group': ['A'] * 5 + ['B'] * 5,
        'target': [1, 1, 1, 1, 0,  1, 0, 0, 0, 0],  # A:0.8, B:0.2
    })
    res = flag_statistical_parity_difference(
        df=df,
        demographic_column='group',
        target_column='target',
        positive_label=1,
        threshold=0.25
    )

    assert set(['selection_rate', 'spd', 'flagged']).issubset(res.columns)
    assert len(res) == df['group'].nunique()

    # SPD difference invariant: max(spd) - min(spd) == 0.6
    spd_range = res['spd'].max() - res['spd'].min()
    assert spd_range == pytest.approx(0.6, abs=1e-12)

    # At least one group must be flagged with threshold 0.25 (regardless of reference)
    assert res['flagged'].sum() >= 1


def test_string_positive_label_and_large_threshold():
    # Positive label is a string; with a large threshold, no flags expected
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'target': ['Y', 'N', 'Y', 'Y'],  # A:0.5, B:1.0
    })
    res = flag_statistical_parity_difference(
        df=df,
        demographic_column='group',
        target_column='target',
        positive_label='Y',
        threshold=1.0  # very large threshold -> no flags
    )

    assert set(['selection_rate', 'spd', 'flagged']).issubset(res.columns)
    assert len(res) == df['group'].nunique()

    # Selection rates computed correctly
    sel_rates = set(res['selection_rate'].round(10).tolist())
    assert sel_rates == {0.5, 1.0}

    # No flags due to large threshold
    assert not res['flagged'].any()


def test_missing_columns_raise():
    df = pd.DataFrame({
        'g': ['A', 'B'],
        'y': [1, 0],
    })
    with pytest.raises((KeyError, ValueError, AttributeError, TypeError)):
        flag_statistical_parity_difference(
            df=df,
            demographic_column='group',  # does not exist
            target_column='y',
            positive_label=1,
            threshold=0.1
        )


def test_zero_and_full_selection_rates_extremes():
    # One group with 0% positives, another with 100% positives
    df = pd.DataFrame({
        'group': ['A'] * 4 + ['B'] * 4,
        'target': [0, 0, 0, 0,  1, 1, 1, 1],  # A:0.0, B:1.0
    })
    res = flag_statistical_parity_difference(
        df=df,
        demographic_column='group',
        target_column='target',
        positive_label=1,
        threshold=0.4
    )

    assert set(['selection_rate', 'spd', 'flagged']).issubset(res.columns)
    assert len(res) == df['group'].nunique()

    # Selection rate extremes present
    assert res['selection_rate'].min() == pytest.approx(0.0)
    assert res['selection_rate'].max() == pytest.approx(1.0)

    # SPD difference invariant equals 1.0
    spd_range = res['spd'].max() - res['spd'].min()
    assert spd_range == pytest.approx(1.0, abs=1e-12)

    # With threshold 0.4, at least one group must be flagged
    assert res['flagged'].sum() >= 1