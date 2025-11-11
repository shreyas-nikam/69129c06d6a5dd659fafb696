import pytest
import pandas as pd
import numpy as np

# definition_a2e96f2d1b3e49919b818225c6f1fa07 block
# This block must be kept as is. DO NOT REPLACE or REMOVE.
from definition_a2e96f2d1b3e49919b818225c6f1fa07 import compute_disparate_impact_ratio
# End definition_a2e96f2d1b3e49919b818225c6f1fa07 block

# Helper function to create a basic DataFrame for tests
def _create_test_df():
    data = {
        'demographic': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'],
        'outcome': ['Y', 'N', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N', 'N']
    }
    return pd.DataFrame(data)

# Test Case 1: Standard functionality with disparity and a specified reference group
def test_compute_disparate_impact_ratio_standard_case():
    df = _create_test_df()
    demographic_column = 'demographic'
    target_column = 'outcome'
    positive_label = 'Y'
    reference_group = 'A'

    # Expected calculations based on data:
    # Group A: 2 'Y' out of 4 total -> rate = 0.5
    # Group B: 1 'Y' out of 3 total -> rate = 1/3 (~0.333)
    # Group C: 2 'Y' out of 5 total -> rate = 0.4

    # Reference rate (Group A) = 0.5
    # DI(A) = 0.5 / 0.5 = 1.0
    # DI(B) = (1/3) / 0.5 = 2/3 (~0.666)
    # DI(C) = 0.4 / 0.5 = 0.8
    # meets_four_fifths: True if di_ratio >= 0.8
    # A: 1.0 >= 0.8 (True)
    # B: 0.666 < 0.8 (False)
    # C: 0.8 >= 0.8 (True)

    expected_data = {
        'group': ['A', 'B', 'C'],
        'positive_rate': [0.5, 1/3, 0.4],
        'di_ratio': [1.0, 2/3, 0.8],
        'meets_four_fifths': [True, False, True]
    }
    expected_df = pd.DataFrame(expected_data).set_index('group')
    # Ensure dtypes match what pandas would produce for consistency
    expected_df['positive_rate'] = expected_df['positive_rate'].astype(float)
    expected_df['di_ratio'] = expected_df['di_ratio'].astype(float)
    expected_df['meets_four_fifths'] = expected_df['meets_four_fifths'].astype(bool)

    result_df = compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True, check_exact=False)


# Test Case 2: Reference group is None, implying the largest group should be used
def test_compute_disparate_impact_ratio_reference_none():
    df = _create_test_df()
    demographic_column = 'demographic'
    target_column = 'outcome'
    positive_label = 'Y'
    reference_group = None # Group C is the largest (5 members), so it should be the reference.

    # Group counts: A=4, B=3, C=5. So, C is the largest group.
    # Group A: rate = 0.5
    # Group B: rate = 1/3
    # Group C: rate = 0.4

    # Reference rate (Group C) = 0.4
    # DI(A) = 0.5 / 0.4 = 1.25
    # DI(B) = (1/3) / 0.4 = 0.833...
    # DI(C) = 0.4 / 0.4 = 1.0
    # meets_four_fifths: True if di_ratio >= 0.8
    # A: 1.25 >= 0.8 (True)
    # B: 0.833 >= 0.8 (True)
    # C: 1.0 >= 0.8 (True)

    expected_data = {
        'group': ['A', 'B', 'C'],
        'positive_rate': [0.5, 1/3, 0.4],
        'di_ratio': [1.25, (1/3)/0.4, 1.0],
        'meets_four_fifths': [True, True, True]
    }
    expected_df = pd.DataFrame(expected_data).set_index('group')
    expected_df['positive_rate'] = expected_df['positive_rate'].astype(float)
    expected_df['di_ratio'] = expected_df['di_ratio'].astype(float)
    expected_df['meets_four_fifths'] = expected_df['meets_four_fifths'].astype(bool)

    result_df = compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True, check_exact=False)


# Test Case 3: No disparity, all DI ratios should be 1.0
def test_compute_disparate_impact_ratio_no_disparity():
    data = {
        'demographic': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
        'outcome': ['Y', 'N', 'Y', 'N', 'Y', 'N']
    }
    df = pd.DataFrame(data)
    demographic_column = 'demographic'
    target_column = 'outcome'
    positive_label = 'Y'
    reference_group = 'X' # Any group will do as reference, since all rates are equal

    # All groups: 1 'Y' out of 2 total -> rate = 0.5
    # Reference rate (Group X) = 0.5
    # All DI ratios = 0.5 / 0.5 = 1.0
    # meets_four_fifths: True for all

    expected_data = {
        'group': ['X', 'Y', 'Z'],
        'positive_rate': [0.5, 0.5, 0.5],
        'di_ratio': [1.0, 1.0, 1.0],
        'meets_four_fifths': [True, True, True]
    }
    expected_df = pd.DataFrame(expected_data).set_index('group')
    expected_df['positive_rate'] = expected_df['positive_rate'].astype(float)
    expected_df['di_ratio'] = expected_df['di_ratio'].astype(float)
    expected_df['meets_four_fifths'] = expected_df['meets_four_fifths'].astype(bool)

    result_df = compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True, check_exact=False)


# Test Case 4: Edge case - Reference group has 0 positive outcomes
def test_compute_disparate_impact_ratio_zero_reference_rate():
    data = {
        'demographic': ['Ref', 'Ref', 'Ref', 'GroupA', 'GroupA'],
        'outcome': ['N', 'N', 'N', 'Y', 'N'] # 'Ref' has 0 'Y's, 'GroupA' has 1 'Y'
    }
    df = pd.DataFrame(data)
    demographic_column = 'demographic'
    target_column = 'outcome'
    positive_label = 'Y'
    reference_group = 'Ref'

    # Group 'Ref': 0 'Y' out of 3 total -> rate = 0.0
    # Group 'GroupA': 1 'Y' out of 2 total -> rate = 0.5

    # Reference rate (Group 'Ref') = 0.0
    # DI(GroupA) = 0.5 / 0.0 = np.inf
    # DI(Ref) = 0.0 / 0.0 = np.nan (typically, or 1.0 based on specific definition)
    # meets_four_fifths: np.inf and np.nan indicate non-compliance or undefined state, thus False.

    expected_data = {
        'group': ['GroupA', 'Ref'], # Order might vary; assert_frame_equal handles index alignment
        'positive_rate': [0.5, 0.0],
        'di_ratio': [np.inf, np.nan],
        'meets_four_fifths': [False, False]
    }
    expected_df = pd.DataFrame(expected_data).set_index('group')
    expected_df['positive_rate'] = expected_df['positive_rate'].astype(float)
    expected_df['di_ratio'] = expected_df['di_ratio'].astype(float)
    expected_df['meets_four_fifths'] = expected_df['meets_four_fifths'].astype(bool)

    result_df = compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group)

    # Use check_exact=False and manually check for NaNs/Infs as pandas assert_frame_equal
    # can be very strict for these specific float values even with check_exact=False.
    # Compare all columns except 'di_ratio' first
    pd.testing.assert_frame_equal(result_df.drop(columns=['di_ratio']), expected_df.drop(columns=['di_ratio']),
                                  check_dtype=True, check_exact=False)
    # Manually check 'di_ratio' column for inf and nan
    assert np.isinf(result_df.loc['GroupA', 'di_ratio'])
    assert np.isnan(result_df.loc['Ref', 'di_ratio'])


# Test Case 5: Edge case - Column not found in DataFrame
def test_compute_disparate_impact_ratio_column_not_found():
    df = _create_test_df()
    demographic_column = 'non_existent_demographic'
    target_column = 'outcome'
    positive_label = 'Y'
    reference_group = 'A'

    # Test for non-existent demographic_column
    with pytest.raises(KeyError, match=f"'{demographic_column}' not found in DataFrame"):
        compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group)

    demographic_column = 'demographic'
    target_column = 'non_existent_outcome'
    # Test for non-existent target_column
    with pytest.raises(KeyError, match=f"'{target_column}' not found in DataFrame"):
        compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group)