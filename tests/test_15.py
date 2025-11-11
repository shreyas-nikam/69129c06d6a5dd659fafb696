import pytest
import pandas as pd
import numpy as np

# This block must remain as is, DO NOT REPLACE or REMOVE.
from definition_0ef1821c30c94d57bece7d9c80a63b1a import summarize_performance_variability
# End of DO NOT TOUCH block

@pytest.mark.parametrize("input_df_data, group_by_col, expected_outcome", [
    # Test Case 1: Standard functionality with multiple groups and metrics
    # Expected output contains means and standard deviations for each group and metric.
    (pd.DataFrame({
        'group_col': ['A', 'A', 'B', 'B', 'A'],
        'metric1': [10, 12, 5, 7, 11],
        'metric2': [100, 110, 50, 60, 105]
    }), 'group_col', { # Expected values for verification (mean, std)
        'A': {'metric1': {'mean': 11.0, 'std': 1.0}, 'metric2': {'mean': 105.0, 'std': 5.0}},
        'B': {'metric1': {'mean': 6.0, 'std': pytest.approx(np.sqrt(2.0))}, 'metric2': {'mean': 55.0, 'std': pytest.approx(np.sqrt(50.0))}}
    }),

    # Test Case 2: `group_by` column not found in the DataFrame
    # Expects a KeyError as pandas.DataFrame.groupby() would raise it.
    (pd.DataFrame({'metric_only': [1,2,3]}), 'non_existent_col', KeyError),

    # Test Case 3: Empty `results_df`
    # Expects an empty DataFrame as a result, assuming graceful handling of empty inputs.
    (pd.DataFrame(columns=['group_col', 'metric1', 'metric2'], dtype='object'), 'group_col', 'empty_df_expected'),

    # Test Case 4: `results_df` is not a pandas DataFrame (e.g., None)
    # Expects an AttributeError because .groupby() would be called on a non-DataFrame object.
    (None, 'group_col', AttributeError), 

    # Test Case 5: Single group scenario (all data belongs to one group)
    # Verifies correct summary statistics, including std dev for a small number of samples.
    (pd.DataFrame({
        'group_col': ['overall', 'overall', 'overall'],
        'metric1': [10, 20, 30],
        'metric2': [1, 2, 3] 
    }), 'group_col', { # Expected values for verification
        'overall': {'metric1': {'mean': 20.0, 'std': pytest.approx(10.0)}, 'metric2': {'mean': 2.0, 'std': pytest.approx(1.0)}}
    }),
])
def test_summarize_performance_variability(input_df_data, group_by_col, expected_outcome):
    if isinstance(expected_outcome, type) and issubclass(expected_outcome, Exception):
        # If expected_outcome is an exception type, assert that the function raises it.
        with pytest.raises(expected_outcome):
            summarize_performance_variability(input_df_data, group_by_col)
    elif expected_outcome == 'empty_df_expected':
        # If expecting an empty DataFrame, call the function and check if the result is an empty DataFrame.
        actual_df = summarize_performance_variability(input_df_data, group_by_col)
        assert isinstance(actual_df, pd.DataFrame)
        assert actual_df.empty
    else: 
        # For successful cases, where expected_outcome is a dictionary of expected values.
        actual_df = summarize_performance_variability(input_df_data, group_by_col)
        assert isinstance(actual_df, pd.DataFrame)
        assert not actual_df.empty

        # Iterate through the expected outcomes to verify specific values.
        # This assumes the summary DataFrame has a MultiIndex for columns (metric, stat).
        for group, metrics_expected in expected_outcome.items():
            for metric, stats_expected in metrics_expected.items():
                for stat, expected_value in stats_expected.items():
                    # Access the actual value from the DataFrame using .loc with MultiIndex for columns
                    # The docstring mentions mean, std, CI, and effect size. For this test, we verify mean and std.
                    # Other stats would need similar checks.
                    assert actual_df.loc[group, (metric, stat)] == expected_value