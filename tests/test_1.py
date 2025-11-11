import pytest
import pandas as pd
import numpy as np
from definition_549047dad22b40f0ab84744586fb3f6f import generate_dataset_overview

@pytest.mark.parametrize("df_input, include_object_stats, expected_output_keys, expected_error", [
    # Test Case 1: Mixed DataFrame with numeric and object columns, include_object_stats=True
    # Expect all relevant stats to be present and non-empty.
    (pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0], 'C': ['x', 'y', 'z']}), True,
     ['dtypes', 'non_null_counts', 'memory_usage', 'numeric_stats', 'object_stats'], None),

    # Test Case 2: Mixed DataFrame with numeric and object columns, include_object_stats=False
    # Expect 'object_stats' key to be absent.
    (pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0], 'C': ['x', 'y', 'z']}), False,
     ['dtypes', 'non_null_counts', 'memory_usage', 'numeric_stats'], None),

    # Test Case 3: Empty DataFrame (no columns, no rows), include_object_stats=True
    # dtypes and non_null_counts should be empty Series. numeric_stats and object_stats should be empty DataFrames.
    (pd.DataFrame(), True,
     ['dtypes', 'non_null_counts', 'memory_usage', 'numeric_stats', 'object_stats'], None),

    # Test Case 4: DataFrame with only numeric columns, include_object_stats=True
    # object_stats should be an empty DataFrame.
    (pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]}), True,
     ['dtypes', 'non_null_counts', 'memory_usage', 'numeric_stats', 'object_stats'], None),

    # Test Case 5: Invalid df_input type (not a pandas DataFrame)
    ("this is not a DataFrame", True, None, TypeError),
])
def test_generate_dataset_overview(df_input, include_object_stats, expected_output_keys, expected_error):
    if expected_error:
        with pytest.raises(expected_error):
            generate_dataset_overview(df_input, include_object_stats)
    else:
        result = generate_dataset_overview(df_input, include_object_stats)

        # 1. Assert result is a dictionary
        assert isinstance(result, dict), "Result should be a dictionary."

        # 2. Assert all expected keys are present
        for key in expected_output_keys:
            assert key in result, f"Expected key '{key}' not found in result."
        
        # 3. Assert no unexpected keys (e.g., 'object_stats' when include_object_stats is False)
        if 'object_stats' not in expected_output_keys:
            assert 'object_stats' not in result, "Unexpected 'object_stats' key found when not expected."
        
        # 4. Assert types and content consistency for present keys

        # Check 'dtypes'
        assert isinstance(result.get('dtypes'), pd.Series), "Value for 'dtypes' should be a pandas Series."
        if not df_input.empty:
            pd.testing.assert_series_equal(result['dtypes'], df_input.dtypes, check_names=False)
        else: # For empty DataFrame, dtypes Series should be empty
            assert result['dtypes'].empty

        # Check 'non_null_counts'
        assert isinstance(result.get('non_null_counts'), pd.Series), "Value for 'non_null_counts' should be a pandas Series."
        if not df_input.empty:
            pd.testing.assert_series_equal(result['non_null_counts'], df_input.notnull().sum(), check_names=False)
        else: # For empty DataFrame, non_null_counts Series should be empty
            assert result['non_null_counts'].empty

        # Check 'memory_usage'
        assert isinstance(result.get('memory_usage'), (int, float)), "Value for 'memory_usage' should be an int or float."
        # Not asserting exact value as it can vary, only type.

        # Check 'numeric_stats'
        assert isinstance(result.get('numeric_stats'), pd.DataFrame), "Value for 'numeric_stats' should be a pandas DataFrame."
        expected_numeric_describe = df_input.select_dtypes(include=np.number).describe()
        
        # Check if the empty status matches for numeric stats
        assert result['numeric_stats'].empty == expected_numeric_describe.empty, \
            "numeric_stats empty status mismatch with expected describe output."
        
        if not result['numeric_stats'].empty:
            # For non-empty numeric stats, compare structure and approximately values
            pd.testing.assert_frame_equal(result['numeric_stats'], expected_numeric_describe, check_dtype=True, rtol=1e-5, atol=1e-5)

        # Check 'object_stats' if expected to be present
        if 'object_stats' in expected_output_keys:
            assert isinstance(result.get('object_stats'), pd.DataFrame), "Value for 'object_stats' should be a pandas DataFrame."
            expected_object_describe = df_input.select_dtypes(include='object').describe()

            # Check if the empty status matches for object stats
            assert result['object_stats'].empty == expected_object_describe.empty, \
                "object_stats empty status mismatch with expected describe output."
            
            if not result['object_stats'].empty:
                pd.testing.assert_frame_equal(result['object_stats'], expected_object_describe, check_dtype=True)

