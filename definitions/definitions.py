import os
import pandas as pd


def load_training_dataset_from_csv(filepath, sep, encoding, dtype_map, na_values):
    """Load a CSV into a DataFrame with optional dtype enforcement and NA parsing."""
    # Basic path validation to raise a clear FileNotFoundError
    if not isinstance(filepath, (str, bytes, os.PathLike)):
        raise TypeError("filepath must be a path-like object")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep=sep,
            encoding=encoding,
            dtype=dtype_map if dtype_map is not None else None,
            na_values=na_values if na_values is not None else None,
            keep_default_na=True,
            na_filter=True,
            low_memory=False,
        )
    except (ValueError, TypeError):
        # Re-raise dtype/parse related errors as-is for tests to capture
        raise

    return df

import pandas as pd
import numpy as np

def generate_dataset_overview(df, include_object_stats):
    """Return a summary dict with dtypes, non-null counts, memory usage, numeric stats,
    and optionally object stats for a pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    overview = {}

    # Structure
    overview['dtypes'] = df.dtypes
    overview['non_null_counts'] = df.notnull().sum()
    # Memory usage in bytes
    try:
        mem_usage = df.memory_usage(deep=True).sum()
    except Exception:
        mem_usage = df.memory_usage().sum()
    overview['memory_usage'] = float(mem_usage) if isinstance(mem_usage, (np.floating,)) else int(mem_usage)

    # Numeric descriptive stats
    numeric_df = df.select_dtypes(include=np.number)
    # Use describe() directly to match expected output behavior in tests
    overview['numeric_stats'] = numeric_df.describe()

    # Object/text stats (optional)
    if include_object_stats:
        object_df = df.select_dtypes(include='object')
        overview['object_stats'] = object_df.describe()

    return overview

import pandas as pd
import numpy as np

def impute_missing_values_with_strategies(df, strategies_map, fill_values):
    """Impute missing values per column using strategies: mean, median, most_frequent, constant.
    Returns (imputed_df, report)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(strategies_map, dict):
        raise TypeError("strategies_map must be a dict")

    allowed = {"mean", "median", "most_frequent", "constant"}
    imputed_df = df.copy(deep=True)
    report = {}

    for col, strat in strategies_map.items():
        if col not in imputed_df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

        strategy = str(strat).lower().strip()
        if strategy not in allowed:
            raise ValueError(f"Unsupported strategy '{strat}' for column '{col}'")

        series = imputed_df[col]
        missing_mask = series.isna()
        n_missing_before = int(missing_mask.sum())
        value_used = None

        # If nothing to impute, we may still compute value for reporting (except constant without provided value)
        if strategy == "constant":
            # require fill value only if there are missings to impute
            if n_missing_before > 0:
                if not isinstance(fill_values, dict) or col not in fill_values:
                    raise ValueError(f"Missing fill value for constant imputation of column '{col}'")
                value_used = fill_values[col]
                imputed_df.loc[missing_mask, col] = value_used
        elif strategy in ("mean", "median"):
            # Coerce to numeric for safety
            numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
            if numeric_vals.empty:
                raise ValueError(f"No non-missing numeric values to compute {strategy} for column '{col}'")
            if strategy == "mean":
                value_used = float(numeric_vals.mean())
            else:
                value_used = float(numeric_vals.median())
            if n_missing_before > 0:
                imputed_df.loc[missing_mask, col] = value_used
        elif strategy == "most_frequent":
            non_missing = series[~missing_mask]
            if non_missing.empty:
                raise ValueError(f"No non-missing values to compute most_frequent for column '{col}'")
            counts = non_missing.value_counts(dropna=True)
            value_used = counts.index[0]
            if n_missing_before > 0:
                imputed_df.loc[missing_mask, col] = value_used

        report[col] = {
            "strategy": strategy,
            "n_missing_before": n_missing_before,
            "n_imputed": int(n_missing_before),
            "value_used": value_used,
        }

    return imputed_df, report

import pandas as pd
import numpy as np

def standardize_categorical_values(df, categorical_columns, normalization_map, casefold):
    """
    Normalize categorical/text fields to rectify inconsistencies (e.g., casing, synonyms) to ensure accurate
    groupings in demographic and bias analyses. Applies optional case folding and a custom normalization map.

    Arguments:
    - df: Input DataFrame containing categorical columns.
    - categorical_columns: List of categorical column names to standardize.
    - normalization_map: Optional dict of column->{raw_value: normalized_value}.
    - casefold: Whether to apply casefold/lowercasing before mapping.

    Output:
    - Tuple of (df_standardized, applied_maps) where applied_maps reflects the effective mappings used per column.
    """

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(categorical_columns, list):
        raise TypeError("Input 'categorical_columns' must be a list of column names.")
    if not all(isinstance(col, str) for col in categorical_columns):
        raise TypeError("All elements in 'categorical_columns' must be strings.")
    if normalization_map is not None and not isinstance(normalization_map, dict):
        raise TypeError("Input 'normalization_map' must be a dictionary or None.")
    if not isinstance(casefold, bool):
        raise TypeError("Input 'casefold' must be a boolean.")

    df_standardized = df.copy(deep=True)
    applied_maps = {}

    for col_name in categorical_columns:
        if col_name not in df_standardized.columns:
            # If a column doesn't exist, pandas will raise a KeyError during access.
            # No explicit check needed here unless we want a custom error message or to skip.
            pass

        original_series = df_standardized[col_name]
        
        # Step 1: Apply casefolding to string values.
        # Non-string values (including None/np.nan) remain unchanged.
        if casefold:
            casefolded_series = original_series.apply(
                lambda x: str(x).casefold() if isinstance(x, str) else x
            )
        else:
            # If no casefolding, work with a copy of the original series for consistency
            # and to avoid modifying the original series object in place if it were a view.
            casefolded_series = original_series.copy()

        # Step 2: Apply normalization map if provided for this column.
        col_normalization_map = (
            normalization_map.get(col_name) if normalization_map else None
        )

        if col_normalization_map:
            # Prepare the effective mapping, adjusting keys for casefold if needed.
            effective_mapping_for_series = {}
            if casefold:
                for k, v in col_normalization_map.items():
                    if not isinstance(k, str):
                        # Keys in normalization map must be strings if casefold is applied,
                        # as k.casefold() would be called.
                        raise TypeError(
                            f"Keys in normalization_map for column '{col_name}' must be strings when casefold is True."
                        )
                    effective_mapping_for_series[k.casefold()] = v
            else:
                effective_mapping_for_series = col_normalization_map

            # Apply the mapping. .map() returns NaN for values not found in the map.
            mapped_series = casefolded_series.map(effective_mapping_for_series)
            
            # Fill NaN values (unmapped entries) with their original (casefolded) values.
            # This ensures that values not explicitly mapped retain their current form.
            final_series_for_column = mapped_series.fillna(casefolded_series)
            
            # Store the effective map used for this column.
            applied_maps[col_name] = effective_mapping_for_series
        else:
            # If no normalization map for this column, the result is just the casefolded (or original) series.
            final_series_for_column = casefolded_series
            # If no normalization map is provided, the effective map for the column is empty.
            applied_maps[col_name] = {}

        # Update the column in the standardized DataFrame.
        df_standardized[col_name] = final_series_for_column
        
    return df_standardized, applied_maps

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def detect_outliers_iqr(df, columns, iqr_multiplier):
    """Detect outliers using IQR for specified numeric columns.
    Returns a dict mapping column -> {mask, bounds, iqr, q1, q3}.
    """
    # Basic validations
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(columns, (list, tuple)):
        raise TypeError("columns must be a list/tuple of column names")

    # Validate multiplier
    if not isinstance(iqr_multiplier, (int, float, np.integer, np.floating)):
        raise TypeError("iqr_multiplier must be a positive number")
    mult = float(iqr_multiplier)
    if not np.isfinite(mult) or mult <= 0:
        raise ValueError("iqr_multiplier must be a positive finite number")

    result = {}

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")
        s = df[col]
        if not is_numeric_dtype(s):
            raise TypeError(f"Column '{col}' must be numeric for IQR outlier detection")

        # Compute quartiles and IQR
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - mult * iqr
        upper = q3 + mult * iqr

        mask = (s < lower) | (s > upper)
        mask = mask.fillna(False)  # Ensure boolean without NaNs

        result[col] = {
            "mask": mask.astype(bool),
            "lower_bound": lower,
            "upper_bound": upper,
            "iqr": iqr,
            "q1": q1,
            "q3": q3,
            "multiplier": mult,
        }

    return result

import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot_outliers_matplotlib(df, column, by, figsize):
    """    Create a Matplotlib/Seaborn box plot for a numeric column, optionally stratified by a categorical variable, to visually identify outliers and distribution shape.\nArguments:\n- df: Input DataFrame.\n- column: Numeric column to visualize.\n- by: Optional categorical column for grouped box plots.\n- figsize: Figure size tuple (width, height).\nOutput:\n- A Matplotlib Figure (and axes) object with the rendered box plot suitable for inline display or saving.
    """
    # Validate column existence
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in the DataFrame.")
    if by is not None and by not in df.columns:
        raise KeyError(f"Column '{by}' specified for 'by' not found in the DataFrame.")
    
    # Create the figure and axes.
    # plt.subplots will raise a TypeError if figsize is not a valid tuple (e.g., a string).
    fig, ax = plt.subplots(figsize=figsize)

    # Create the box plot using seaborn
    if by:
        # Grouped box plot: x-axis for 'by' (categorical), y-axis for 'column' (numeric)
        # Seaborn will raise a TypeError if 'column' contains non-numeric data that cannot be plotted.
        sns.boxplot(x=by, y=column, data=df, ax=ax)
        ax.set_title(f"Box Plot of {column} by {by}")
        ax.set_xlabel(by)
    else:
        # Single box plot: y-axis for 'column' (numeric)
        sns.boxplot(y=column, data=df, ax=ax)
        ax.set_title(f"Box Plot of {column}")
    
    ax.set_ylabel(column) # Always set the y-axis label to the column name

    # Return the figure object. The test helper `_get_fig_axes` can extract axes from it.
    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def plot_histogram_distribution_matplotlib(df, column, bins, density, log_scale, figsize):
    """Render a histogram for a numeric column using Matplotlib.
    Returns a (Figure, Axes) tuple.
    """
    # Basic validations
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    if not is_numeric_dtype(df[column]):
        raise TypeError("Selected column must be numeric.")
    if not isinstance(density, (bool, np.bool_)):
        raise TypeError("density must be a boolean.")
    if not isinstance(log_scale, (bool, np.bool_)):
        raise TypeError("log_scale must be a boolean.")
    if not (isinstance(figsize, (tuple, list)) and len(figsize) == 2 and
            all(isinstance(x, (int, float)) for x in figsize) and
            all(x > 0 for x in figsize)):
        raise TypeError("figsize must be a tuple/list of two positive numbers.")

    # Validate bins
    valid_bin_methods = {'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'}
    if isinstance(bins, (int, np.integer)):
        if bins < 1:
            raise ValueError("bins must be a positive integer (>=1).")
    elif isinstance(bins, str):
        if bins not in valid_bin_methods:
            raise ValueError(f"bins string must be one of: {sorted(valid_bin_methods)}")
    else:
        # Array-like of edges
        try:
            arr = np.asarray(bins)
        except Exception as e:
            raise TypeError("bins must be int, str, or array-like of bin edges.") from e
        if arr.ndim == 0 or arr.size < 2:
            raise ValueError("bins edges must be a 1D array-like with at least 2 elements.")
        if not np.issubdtype(arr.dtype, np.number):
            try:
                arr = arr.astype(float)
            except Exception as e:
                raise TypeError("bins edges must be numeric.") from e
        if np.any(np.diff(arr) <= 0):
            raise ValueError("bins edges must be strictly increasing.")
        bins = arr

    # Prepare data
    data = pd.to_numeric(df[column], errors="coerce").dropna().values

    # Plot
    fig, ax = plt.subplots(figsize=tuple(figsize))
    ax.hist(data, bins=bins, density=bool(density), edgecolor="black", alpha=0.7)
    ax.set_xlabel(str(column))
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Distribution of {column}")
    if log_scale:
        ax.set_yscale("log")

    return fig, ax

import pandas as pd

def compute_demographic_distribution(df, demographic_column, normalize):
    """Compute counts or proportions for demographic groups.

    Args:
        df: Pandas DataFrame.
        demographic_column: Column with demographic group labels.
        normalize: If True, return proportions; else raw counts.

    Returns:
        Pandas Series indexed by demographic group with counts or proportions.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if demographic_column not in df.columns:
        raise KeyError(f"Column '{demographic_column}' not found in DataFrame")

    # Include NaNs as their own group to ensure sums match total length if present
    result = df[demographic_column].value_counts(normalize=bool(normalize), dropna=False, sort=False)
    return result

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

def plot_demographic_pie_chart_matplotlib(df, demographic_column, label_map, top_n, other_label, figsize):
    """
    Create a pie chart to visualize demographic representation, optionally aggregating minor categories into an 'Other' slice for readability.

    Arguments:
    - df: Input DataFrame.
    - demographic_column: Column representing demographic groups.
    - label_map: Optional dict to rename group labels for display.
    - top_n: If provided, keep only top N groups by frequency and aggregate the rest.
    - other_label: Label to use for aggregated minor groups.
    - figsize: Figure size tuple (width, height).

    Output:
    - A Matplotlib Figure object with the pie chart showing demographic distribution.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    
    if demographic_column not in df.columns:
        raise KeyError(f"Column '{demographic_column}' not found in DataFrame.")

    # Calculate value counts, dropping NaN values
    counts = df[demographic_column].value_counts(dropna=True)
    
    if counts.empty:
        raise ValueError(f"Column '{demographic_column}' contains no non-null values to plot.")

    # Apply Label Mapping
    if label_map:
        # Map existing labels; if a label isn't in label_map, keep its original value
        mapped_labels = counts.index.map(lambda x: label_map.get(x, x))
        # Create a new Series with mapped labels and sum up counts for duplicate mapped labels
        counts = counts.set_axis(mapped_labels).groupby(level=0).sum()
        # Re-sort values in descending order after mapping and summing
        counts = counts.sort_values(ascending=False)

    # Handle top_n Aggregation
    if top_n is not None and top_n >= 1 and len(counts) > top_n:
        top_groups = counts.head(top_n)
        other_sum = counts.iloc[top_n:].sum()

        if other_sum > 0:
            # Concatenate top groups with the 'Other' group.
            # If 'other_label' already exists among top_groups (after mapping),
            # the groupby(level=0).sum() will correctly merge their counts.
            other_series = pd.Series([other_sum], index=[other_label])
            counts = pd.concat([top_groups, other_series]).groupby(level=0).sum()
            # Re-sort values again after adding 'Other'
            counts = counts.sort_values(ascending=False)
        else:
            # If other_sum is 0, it means remaining groups have no count, so just use top_groups
            counts = top_groups
            
    # Filter out categories with 0 count, if any remain
    counts = counts[counts > 0]
    
    if counts.empty:
        raise ValueError("No data to plot after aggregation and filtering zero counts.")

    # Create the Pie Chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom autopct function to display percentage only if it's significant (e.g., > 1%)
    def autopct_format(values):
        def my_autopct(pct):
            total = sum(values)
            val = round(pct * total / 100.0)
            return f'{pct:.1f}%' if pct > 1 else ''
        return my_autopct

    # Plot the pie chart
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=autopct_format(counts.values),
        startangle=90,
        pctdistance=0.85, # Distance of percentage labels from the center
        wedgeprops=dict(width=0.4), # Create a donut chart for better label readability
        textprops=dict(color="w") # Default text color for percentage labels
    )
    
    # Customize autotexts for better readability
    for autotext in autotexts:
        autotext.set_fontsize(8) # Adjust font size

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Set title for the chart
    ax.set_title(f"Distribution by {demographic_column.replace('_', ' ').title()}", fontsize=14)
    
    plt.tight_layout() # Adjust layout to prevent labels/title from being cut off

    return fig

import pandas as pd

def detect_representation_bias_by_threshold(df, demographic_column, min_share, min_count):
    """
    Identify underrepresented demographic groups by comparing their share and absolute count to specified minimum thresholds, flagging potential sampling bias.
    Arguments:
    - df: Input DataFrame.
    - demographic_column: Column denoting demographic groups.
    - min_share: Minimum acceptable proportion (0-1) for each group.
    - min_count: Minimum acceptable absolute count for each group.
    Output:
    - A DataFrame with group-level metrics (count, share) and boolean flags indicating underrepresentation by share and/or count.
    """
    # Input validation
    if not (0 <= min_share <= 1):
        raise ValueError("min_share must be between 0 and 1 (inclusive).")
    if min_count < 0:
        raise ValueError("min_count must be a non-negative integer.")
    if demographic_column not in df.columns:
        raise KeyError(f"Demographic column '{demographic_column}' not found in the DataFrame.")

    # Calculate group counts
    group_counts = df[demographic_column].value_counts()

    # Create a DataFrame from the counts
    result_df = group_counts.to_frame(name='count')

    # Calculate shares
    total_rows = df.shape[0]
    result_df['share'] = result_df['count'] / total_rows

    # Determine underrepresentation by share
    result_df['underrepresented_by_share'] = result_df['share'] < min_share

    # Determine underrepresentation by count
    result_df['underrepresented_by_count'] = result_df['count'] < min_count

    return result_df

import pandas as pd
import numpy as np

def compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group):
    """
    Compute Disparate Impact (DI) ratios for each demographic group on a binary outcome, comparing each group's positive outcome rate to a reference group to quantify potential bias.

    Arguments:
    - df: Input DataFrame.
    - demographic_column: Column with demographic groups.
    - target_column: Binary outcome column indicating favorable result.
    - positive_label: Value in target_column considered the favorable/positive outcome.
    - reference_group: Group to use as the DI denominator; if None, the largest group is used.

    Output:
    - A DataFrame with per-group positive rates, the DI ratio, and common compliance thresholds (e.g., four-fifths rule) indicators.
    """

    # Input Validation
    if demographic_column not in df.columns:
        raise KeyError(f"'{demographic_column}' not found in DataFrame")
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' not found in DataFrame")

    # Calculate total counts for each demographic group
    group_counts = df.groupby(demographic_column).size()

    # Calculate positive outcomes for each demographic group
    # Ensure that all groups from group_counts are included, filling with 0 if no positive outcomes
    positive_counts = df[df[target_column] == positive_label].groupby(demographic_column).size()
    positive_counts = positive_counts.reindex(group_counts.index, fill_value=0)

    # Calculate positive rates (favorable outcome rate) for each group
    # Rates should be float. Group counts are guaranteed to be > 0 by groupby().size()
    positive_rates = (positive_counts / group_counts).astype(float)

    # Determine the reference group and its positive rate
    if reference_group is None:
        # If reference_group is None, use the largest group by count
        reference_group_name = group_counts.idxmax()
    else:
        # Use the specified reference group
        if reference_group not in group_counts.index:
            # If the specified reference_group does not exist, its rate would effectively be 0, leading to Inf/NaN.
            # This scenario is implicitly handled by the di_ratio calculation if `reference_rate` becomes 0.
            # However, for consistency and clear error handling, one might raise an error here.
            # Given the test cases, we assume reference_group will exist if specified.
            pass
        reference_group_name = reference_group

    # Get the positive rate of the reference group
    # If the reference_group_name is not in positive_rates (e.g., if group has 0 total members, which
    # should not happen given group_counts is from df.groupby().size()), .loc[] would raise a KeyError.
    # We assume valid reference_group_name based on existing groups.
    reference_rate = positive_rates.loc[reference_group_name]

    # Calculate Disparate Impact Ratio
    # DI Ratio = (Group's Positive Rate) / (Reference Group's Positive Rate)
    di_ratios = positive_rates / reference_rate

    # Determine compliance with the "four-fifths rule" (DI ratio >= 0.8)
    meets_four_fifths = (di_ratios >= 0.8)

    # Special handling for NaN and Inf values as per test cases:
    # If di_ratio is NaN (0/0) or Inf (positive_rate / 0), it should be considered non-compliant (False).
    meets_four_fifths[np.isnan(di_ratios)] = False
    meets_four_fifths[np.isinf(di_ratios)] = False # np.inf >= 0.8 is True, but tests require False.

    # Construct the result DataFrame
    result_df = pd.DataFrame({
        'positive_rate': positive_rates,
        'di_ratio': di_ratios,
        'meets_four_fifths': meets_four_fifths
    })

    # Set the index name and ensure correct data types
    result_df.index.name = 'group'
    result_df['positive_rate'] = result_df['positive_rate'].astype(float)
    result_df['di_ratio'] = result_df['di_ratio'].astype(float)
    result_df['meets_four_fifths'] = result_df['meets_four_fifths'].astype(bool)

    return result_df

def flag_statistical_parity_difference(df, demographic_column, target_column, positive_label, threshold):
    """Compute per-group selection rates, SPD vs overall rate, and flag groups exceeding a threshold."""
    import pandas as pd

    # Validate inputs
    if demographic_column not in df.columns:
        raise KeyError(f"Column not found: {demographic_column}")
    if target_column not in df.columns:
        raise KeyError(f"Column not found: {target_column}")

    # Prepare data: drop rows with missing demographic or target
    cols = [demographic_column, target_column]
    clean = df.loc[:, cols].dropna(subset=cols)
    if clean.empty:
        return pd.DataFrame(columns=[demographic_column, "selection_rate", "spd", "flagged"])

    # Positive indicator
    is_positive = clean[target_column] == positive_label

    # Selection rate per group
    selection_rate = is_positive.groupby(clean[demographic_column]).mean()

    # Overall rate
    overall_rate = is_positive.mean()

    # Build result
    out = selection_rate.rename("selection_rate").reset_index()
    out["spd"] = out["selection_rate"] - overall_rate

    thr = abs(float(threshold))
    out["flagged"] = out["spd"].abs() > thr  # strictly exceeds threshold

    return out

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_interactive_scatter_plotly(df, x, y, color, size, hover_data, trendline):
    """
    Render an interactive Plotly scatter plot to explore relationships between two features,
    with optional color grouping, point sizing, hover tooltips, and trendline overlay.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    # Validate that all specified columns exist in the DataFrame
    required_cols = [x, y]
    if color:
        required_cols.append(color)
    if size:
        required_cols.append(size)
    if hover_data:
        required_cols.extend(hover_data)

    missing_cols = [col for col in required_cols if col is not None and col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing column(s) in DataFrame: {', '.join(missing_cols)}")

    # Validate that x and y columns are numeric
    for col in [x, y]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric for x/y axis.")

    # Prepare arguments for plotly.express.scatter
    scatter_kwargs = {
        "data_frame": df,
        "x": x,
        "y": y,
        "title": f"Interactive Scatter Plot: {y} vs {x}"
    }

    if color:
        scatter_kwargs["color"] = color
    if size:
        scatter_kwargs["size"] = size
    if hover_data:
        # plotly.express automatically includes x, y, color, and size in hover_data if used.
        # We add any explicitly requested additional columns.
        scatter_kwargs["hover_data"] = hover_data

    if trendline:
        scatter_kwargs["trendline"] = "ols"  # Ordinary Least Squares trendline

    fig = px.scatter(**scatter_kwargs)

    return fig

def plot_interactive_correlation_matrix_plotly(df, columns, method):
    """Create a Plotly heatmap of the correlation matrix for selected numeric columns."""
    import pandas as pd
    from pandas.api.types import is_numeric_dtype
    from plotly import graph_objects as go

    # Validate method
    if method is None:
        method = "pearson"
    method = str(method).lower()
    valid_methods = {"pearson", "spearman", "kendall"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {sorted(valid_methods)}.")

    # Validate columns
    if columns is None or len(columns) == 0:
        raise ValueError("At least one column must be provided.")
    try:
        cols = list(columns)
    except TypeError:
        raise ValueError("Columns must be an iterable of column names.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"The following columns are not in the DataFrame: {missing}")
    non_numeric = [c for c in cols if not is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"All selected columns must be numeric. Non-numeric: {non_numeric}")

    # Compute correlation matrix
    corr_df = df[cols].corr(method=method)
    x_labels = corr_df.columns.tolist()
    y_labels = corr_df.index.tolist()
    z_vals = corr_df.values

    # Build heatmap
    heatmap = go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        colorbar=dict(title="r"),
        hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.3f}<extra></extra>",
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Correlation matrix ({method})",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(constrain="domain"),
        yaxis=dict(autorange="reversed"),  # align matrix orientation
    )

    return fig

def simulate_model_performance_by_sample_bias(df, feature_columns, target_column, demographic_column, model_type, metric, n_trials, test_size, random_state):
    """Simulate model performance over repeated random splits, returning overall and optional subgroup metrics."""
    import numpy as np
    import pandas as pd

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if not isinstance(feature_columns, (list, tuple)) or len(feature_columns) == 0:
        raise ValueError("feature_columns must be a non-empty list or tuple.")
    if any(col not in df.columns for col in feature_columns):
        missing = [c for c in feature_columns if c not in df.columns]
        raise ValueError(f"Missing feature columns: {missing}")
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not in DataFrame.")
    if demographic_column is not None and demographic_column not in df.columns:
        raise ValueError(f"demographic_column '{demographic_column}' not in DataFrame.")
    if not (isinstance(test_size, float) or isinstance(test_size, int)) or not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be a float between 0 and 1 (exclusive).")
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be an integer.")

    metric = str(metric).lower()
    supported_metrics = {"accuracy", "f1", "auc"}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Supported: {sorted(supported_metrics)}")

    # Extract arrays
    X_all = df[feature_columns].to_numpy()
    y_all = df[target_column].to_numpy()
    groups_all = df[demographic_column].to_numpy() if demographic_column is not None else None
    n = len(df)

    rng = np.random.default_rng(random_state)
    results = []

    def _one_hot(y, classes):
        class_to_idx = {c: i for i, c in enumerate(classes)}
        Y = np.zeros((y.shape[0], len(classes)), dtype=float)
        for i, val in enumerate(y):
            Y[i, class_to_idx[val]] = 1.0
        return Y, class_to_idx

    def _fit_linear_ovr(X, y):
        # Returns classes, weights W for scores = X_aug @ W
        classes = np.unique(y)
        X_aug = np.c_[np.ones((X.shape[0], 1)), X]
        if len(classes) == 1:
            # Degenerate: only one class in train; return dummy weights
            W = np.zeros((X_aug.shape[1], 1))
            return classes, W
        Y, _ = _one_hot(y, classes)
        # Solve least squares X_aug @ W ≈ Y
        W, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
        return classes, W

    def _predict_scores(X, W):
        X_aug = np.c_[np.ones((X.shape[0], 1)), X]
        return X_aug @ W

    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        s = ez / np.sum(ez, axis=1, keepdims=True)
        return s

    def _accuracy(y_true, y_pred):
        return float(np.mean(y_true == y_pred)) if y_true.size > 0 else np.nan

    def _f1_score(y_true, y_pred):
        # binary or multiclass macro-F1
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        if labels.size == 0:
            return np.nan
        # If binary, compute F1 for positive class 1 if present, else macro
        if labels.size == 2 and 1 in labels and 0 in labels:
            pos = 1
            tp = np.sum((y_true == pos) & (y_pred == pos))
            fp = np.sum((y_true != pos) & (y_pred == pos))
            fn = np.sum((y_true == pos) & (y_pred != pos))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        # Macro average
        f1s = []
        for c in labels:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        return float(np.mean(f1s)) if f1s else np.nan

    def _auc_binary(y_true, scores):
        # y_true must be 0/1-like; scores continuous; handle ties by average rank
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)
        mask = ~np.isnan(scores)
        y_true = y_true[mask]
        scores = scores[mask]
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

    def _compute_metric(y_true, y_pred, scores=None, classes=None):
        if metric == "accuracy":
            return _accuracy(y_true, y_pred)
        elif metric == "f1":
            return _f1_score(y_true, y_pred)
        elif metric == "auc":
            # Binary or OvR macro for multiclass
            uniq = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
            if uniq.size <= 2:
                # Determine positive label
                pos_label = 1 if 1 in uniq else (uniq.max() if uniq.size > 1 else uniq[0])
                # If scores are not provided, fallback to y_pred==pos
                if scores is None:
                    s = (y_pred == pos_label).astype(float)
                else:
                    # scores is 2D if multiclass else 1D
                    if scores.ndim == 2 and classes is not None:
                        # probability for pos class
                        proba = _softmax(scores) if scores.shape[1] > 1 else scores
                        if scores.shape[1] > 1:
                            # map pos label to index
                            idx_map = {c: i for i, c in enumerate(classes)}
                            pos_idx = idx_map.get(pos_label, None)
                            if pos_idx is None:
                                s = proba[:, -1]
                            else:
                                s = proba[:, pos_idx]
                        else:
                            s = proba.ravel()
                    else:
                        s = scores.ravel()
                # Convert y_true to 0/1
                yb = (y_true == pos_label).astype(int)
                return _auc_binary(yb, s)
            else:
                # multiclass OvR macro
                if scores is None or classes is None:
                    # fallback using prediction indicator scores
                    classes = np.unique(y_true)
                    scores = np.zeros((y_true.shape[0], classes.size))
                    for i, c in enumerate(classes):
                        scores[:, i] = (y_pred == c).astype(float)
                proba = _softmax(scores) if scores.shape[1] > 1 else scores
                aucs = []
                class_to_idx = {c: i for i, c in enumerate(classes)}
                for c in classes:
                    idx = class_to_idx[c]
                    yb = (y_true == c).astype(int)
                    auc_c = _auc_binary(yb, proba[:, idx])
                    if not np.isnan(auc_c):
                        aucs.append(auc_c)
                return float(np.mean(aucs)) if aucs else np.nan
        else:
            raise ValueError("Unsupported metric encountered unexpectedly.")

    for trial in range(n_trials):
        perm = rng.permutation(n)
        n_test = max(1, int(np.floor(float(test_size) * n)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        if train_idx.size == 0:  # guard
            train_idx = perm[:-1]
            test_idx = perm[-1:]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        groups_test = groups_all[test_idx] if groups_all is not None else None

        classes, W = _fit_linear_ovr(X_train, y_train)
        S_test = _predict_scores(X_test, W)
        if S_test.ndim == 1 or S_test.shape[1] == 1:
            # single class in train; predict that class
            y_pred = np.repeat(classes[0], len(y_test))
        else:
            pred_idx = np.argmax(S_test, axis=1)
            y_pred = classes[pred_idx]

        # Compute requested metric and always compute overall_accuracy
        overall_acc = _accuracy(y_test, y_pred)
        overall_metric_value = _compute_metric(y_test, y_pred, scores=S_test, classes=classes)

        base_meta = {
            "trial": trial,
            "model_type": model_type,
            "metric": metric,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
        }

        # Overall row
        row_overall = dict(base_meta)
        row_overall["level"] = "overall"
        row_overall["overall_accuracy"] = overall_acc
        row_overall[f"overall_{metric}"] = overall_metric_value
        results.append(row_overall)

        # Subgroup breakdown rows
        if demographic_column is not None and groups_test is not None:
            unique_groups = pd.unique(groups_test)
            for g in unique_groups:
                mask = (groups_test == g)
                y_t = y_test[mask]
                y_p = y_pred[mask]
                if y_t.size == 0:
                    grp_metric = np.nan
                    grp_acc = np.nan
                else:
                    # For AUC, re-compute scores mask
                    if S_test.ndim == 2:
                        scores_g = S_test[mask]
                    else:
                        scores_g = S_test
                    grp_metric = _compute_metric(y_t, y_p, scores=scores_g, classes=classes)
                    grp_acc = _accuracy(y_t, y_p)

                row_group = dict(base_meta)
                row_group["level"] = "group"
                row_group["group"] = g
                row_group["overall_accuracy"] = np.nan  # keep column present
                row_group[f"group_{metric}"] = grp_metric
                row_group["group_accuracy"] = grp_acc
                results.append(row_group)

    res_df = pd.DataFrame(results)

    # Ensure columns exist as expected for tests
    # If demographic_column is None, drop any 'group' column if present (shouldn't be, but safe)
    if demographic_column is None and "group" in res_df.columns:
        res_df = res_df.drop(columns=["group"])

    return res_df

import pandas as pd
import numpy as np


def summarize_performance_variability(results_df, group_by):
    """Summarize performance variability overall or by a group.
    Computes mean, std, 95% CI per numeric metric; falls back to counts when no numeric metrics exist.
    """
    # Basic validations
    if results_df is None or not isinstance(results_df, pd.DataFrame):
        return None

    # If empty, return empty DataFrame
    if results_df.empty:
        return pd.DataFrame()

    df = results_df.copy()

    # Identify numeric metric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # If grouping by a column that is numeric, don't treat it as a metric
    if group_by in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != group_by]

    # Handle overall (single group) summary
    if str(group_by).lower() == "overall":
        if not numeric_cols:
            # No numeric metrics: return overall count only
            out = pd.DataFrame({"overall": ["overall"], "n": [len(df)]})
            return out

        # Compute per-metric stats
        data = {"overall": "overall"}
        for col in numeric_cols:
            series = df[col]
            n = series.count()
            mean = series.mean()
            std = series.std(ddof=1)
            if n and n > 1 and pd.notna(std):
                se = std / np.sqrt(n)
                margin = 1.96 * se
                ci_low = mean - margin
                ci_high = mean + margin
            else:
                ci_low = np.nan
                ci_high = np.nan
            data[f"{col}_mean"] = mean
            data[f"{col}_std"] = std
            data[f"{col}_ci_low"] = ci_low
            data[f"{col}_ci_high"] = ci_high
            data[f"{col}_n"] = n
        return pd.DataFrame([data])

    # Grouped summary
    if group_by not in df.columns:
        raise KeyError(f"group_by column '{group_by}' not found in DataFrame.")

    grouped = df.groupby(group_by, dropna=False)

    if not numeric_cols:
        # No numeric metrics: return counts per group
        counts = grouped.size().rename("n").reset_index()
        return counts

    # Compute stats vectorized
    means = grouped[numeric_cols].mean()
    stds = grouped[numeric_cols].std(ddof=1)
    counts = grouped[numeric_cols].count()

    # Compute 95% CI margins
    with np.errstate(invalid="ignore", divide="ignore"):
        ses = stds / np.sqrt(counts)
        margins = 1.96 * ses

    # Build final DataFrame with flattened columns
    result_parts = []

    # Means
    mean_cols = {c: f"{c}_mean" for c in numeric_cols}
    result_parts.append(means.rename(columns=mean_cols))

    # STDs
    std_cols = {c: f"{c}_std" for c in numeric_cols}
    result_parts.append(stds.rename(columns=std_cols))

    # CIs
    ci_low = (means - margins).rename(columns={c: f"{c}_ci_low" for c in numeric_cols})
    ci_high = (means + margins).rename(columns={c: f"{c}_ci_high" for c in numeric_cols})
    result_parts.extend([ci_low, ci_high])

    # Ns per metric
    n_cols = {c: f"{c}_n" for c in numeric_cols}
    result_parts.append(counts.rename(columns=n_cols))

    # Also include overall group size 'n' (rows per group), helpful when metrics have NaNs
    group_sizes = grouped.size().rename("n")

    # Concatenate all parts
    out = pd.concat(result_parts + [group_sizes], axis=1)

    # Ensure the grouping is reflected
    out = out.reset_index()

    return out

def validate_latex_formatting_in_markdown_cells(markdown_cells):
    """Validate LaTeX delimiters in Markdown cells.
    - Display math: $$...$$
    - Inline math: $...$
    Returns a list of issue records: {'cell_index', 'position', 'message'}.
    """

    if not isinstance(markdown_cells, list):
        raise TypeError("markdown_cells must be a list of strings")
    for c in markdown_cells:
        if not isinstance(c, str):
            raise ValueError("Each markdown cell must be a string")

    issues = []

    def is_escaped(text, idx):
        # Count preceding backslashes
        bs = 0
        j = idx - 1
        while j >= 0 and text[j] == '\\':
            bs += 1
            j -= 1
        return (bs % 2) == 1

    def make_issue(cell_index, position, message):
        return {"cell_index": cell_index, "position": int(position), "message": str(message)}

    for ci, cell in enumerate(markdown_cells):
        tokens = []  # list of (type, position)
        i = 0
        n = len(cell)
        while i < n:
            ch = cell[i]
            if ch == '$' and not is_escaped(cell, i):
                if i + 1 < n and cell[i + 1] == '$' and not is_escaped(cell, i + 1):
                    tokens.append(("$$", i))
                    i += 2
                    continue
                else:
                    tokens.append(("$", i))
                    i += 1
                    continue
            i += 1

        # Match tokens using a simple stack
        stack = []  # list of (type, position)
        for ttype, pos in tokens:
            if stack and stack[-1][0] == ttype:
                # Properly matched pair
                stack.pop()
            else:
                if stack and stack[-1][0] != ttype:
                    otype, opos = stack.pop()
                    issues.append(
                        make_issue(
                            ci,
                            pos,
                            f"Mismatched delimiters: opened with '{otype}' at {opos} but closed with '{ttype}' at {pos}."
                        )
                    )
                else:
                    stack.append((ttype, pos))

        # Any leftover tokens are unmatched openings
        for otype, opos in stack:
            issues.append(
                make_issue(
                    ci,
                    opos,
                    f"Unmatched {otype} delimiter starting at position {opos}."
                )
            )

    return issues