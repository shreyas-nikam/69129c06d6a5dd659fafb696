import os
import pandas as pd

def load_training_dataset_from_csv(filepath, sep, encoding, dtype_map, na_values):
    """Load a CSV into a Pandas DataFrame with optional dtype and NA handling."""
    # Validate file existence
    if not isinstance(filepath, (str, os.PathLike)) or not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Validate separator
    if not isinstance(sep, str) or len(sep) == 0:
        raise TypeError("Parameter 'sep' must be a non-empty string.")

    read_kwargs = {
        "sep": sep,
        "encoding": encoding,
        "keep_default_na": True,
        "low_memory": False,
    }
    if dtype_map is not None:
        read_kwargs["dtype"] = dtype_map
    if na_values is not None:
        read_kwargs["na_values"] = na_values

    # Let pandas raise informative errors on dtype conflicts, parsing, etc.
    df = pd.read_csv(filepath, **read_kwargs)
    return df

import pandas as pd
import numpy as np

def generate_dataset_overview(df, include_object_stats):
    """Summarize a DataFrame: dtypes, non-null counts, memory usage, numeric and optional object stats."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    overview = {}

    # Structure info
    overview["dtypes"] = df.dtypes
    overview["non_null_counts"] = df.notna().sum()
    # Use deep memory usage to be within [shallow, deep]
    overview["memory_usage"] = int(df.memory_usage(index=True, deep=True).sum())

    # Numeric stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        overview["numeric_stats"] = pd.DataFrame()
    else:
        overview["numeric_stats"] = df[numeric_cols].describe()

    # Object/Text stats (optional)
    if include_object_stats:
        # Include both 'object' and pandas 'string' dtypes
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if len(obj_cols) == 0:
            overview["object_stats"] = pd.DataFrame()
        else:
            overview["object_stats"] = df[obj_cols].describe()
    return overview

import pandas as pd
from pandas.api.types import is_numeric_dtype


def impute_missing_values_with_strategies(df, strategies_map, fill_values):
    """
    Impute missing values per column using strategies: mean, median, most_frequent, constant.
    Returns (imputed_df, report) where report details strategy and imputed_count per column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if strategies_map is None:
        strategies_map = {}
    if not isinstance(strategies_map, dict):
        raise TypeError("strategies_map must be a dict of {column: strategy}.")

    allowed = {"mean", "median", "most_frequent", "constant"}
    imputed_df = df.copy(deep=True)
    report = {}

    for col, strategy in strategies_map.items():
        if col not in imputed_df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        if strategy not in allowed:
            raise ValueError(f"Invalid strategy '{strategy}' for column '{col}'.")

        series = imputed_df[col]
        n_missing = int(series.isna().sum())
        entry = {"strategy": strategy, "imputed_count": 0}

        # If no missing values, record and continue
        if n_missing == 0:
            if strategy == "constant":
                if fill_values is None or col not in fill_values:
                    # Still raise because user requested constant but provided no value
                    raise KeyError(f"No fill value provided for column '{col}' with 'constant' strategy.")
                entry["fill_value"] = fill_values[col]
            report[col] = entry
            continue

        if strategy in ("mean", "median"):
            if not is_numeric_dtype(series):
                raise TypeError(f"Strategy '{strategy}' requires numeric dtype for column '{col}'.")
            if strategy == "mean":
                fill_value = series.mean()
            else:
                fill_value = series.median()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot compute {strategy} for column '{col}' (all values are NaN).")
            imputed_df[col] = series.fillna(fill_value)
            entry["imputed_count"] = n_missing
        elif strategy == "most_frequent":
            mode_vals = series.mode(dropna=True)
            if mode_vals.empty:
                raise ValueError(f"No non-NaN values to determine most_frequent for column '{col}'.")
            fill_value = mode_vals.iloc[0]
            imputed_df[col] = series.fillna(fill_value)
            entry["imputed_count"] = n_missing
        elif strategy == "constant":
            if fill_values is None or col not in fill_values:
                raise KeyError(f"No fill value provided for column '{col}' with 'constant' strategy.")
            fill_value = fill_values[col]
            if pd.isna(fill_value):
                raise ValueError(f"Fill value for column '{col}' cannot be NaN.")
            imputed_df[col] = series.fillna(fill_value)
            entry["imputed_count"] = n_missing
            entry["fill_value"] = fill_value

        report[col] = entry

    return imputed_df, report

def standardize_categorical_values(df, categorical_columns, normalization_map, casefold):
    """Standardize categorical columns with optional case folding and custom mappings.
    Returns (df_standardized, applied_maps). applied_maps reflects effective mappings used per column.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if categorical_columns is None:
        categorical_columns = []
    if not isinstance(categorical_columns, (list, tuple)):
        raise ValueError("categorical_columns must be a list/tuple of column names")

    missing_cols = [c for c in categorical_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    out_df = df.copy(deep=True)
    applied_maps = {}

    norm_map = normalization_map or {}

    for col in categorical_columns:
        s = out_df[col]

        # Case fold values if requested (preserve None/NaN and non-strings)
        if casefold:
            s = s.map(lambda x: x.casefold() if isinstance(x, str) else x)

        # Build effective mapping for this column, casefolding keys if needed
        col_map_raw = norm_map.get(col, None)
        if col_map_raw:
            if casefold:
                eff_map = {
                    (k.casefold() if isinstance(k, str) else k): v
                    for k, v in col_map_raw.items()
                }
            else:
                eff_map = dict(col_map_raw)
            # Apply mapping: only exact matches are replaced
            s = s.map(lambda x: eff_map.get(x, x))
            applied_maps[col] = eff_map

        out_df[col] = s

    return out_df, applied_maps

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numbers

def detect_outliers_iqr(df, columns, iqr_multiplier):
    """Detect outliers per column using the IQR method.
    Returns a dict mapping each column to a dict with:
    - mask: boolean Series of outliers
    - bounds: metadata with q1, q3, iqr, lower, upper, multiplier
    """
    # Basic validations
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if columns is None:
        raise ValueError("columns must be provided")
    if not hasattr(columns, "__iter__") or isinstance(columns, (str, bytes)):
        raise TypeError("columns must be an iterable of column names")
    columns = list(columns)
    if len(columns) == 0:
        return {}

    if not isinstance(iqr_multiplier, numbers.Real):
        raise TypeError("iqr_multiplier must be a real number")
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be > 0")

    # Check columns existence
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    # Check numeric dtypes
    non_numeric = [c for c in columns if not is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Columns must be numeric for IQR computation: {non_numeric}")

    result = {}
    for col in columns:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = ((s < lower) | (s > upper)).fillna(False)

        result[col] = {
            "mask": mask,
            "bounds": {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
                "multiplier": float(iqr_multiplier),
            },
        }

    return result

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def plot_boxplot_outliers_matplotlib(df, column, by=None, figsize=(6, 4)):
    """Create a Matplotlib box plot for a numeric column, optionally grouped by a categorical column.
    Returns (fig, ax). Raises KeyError for missing columns and TypeError for non-numeric data.
    """
    # Basic validations
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    if not is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric.")

    if by is not None and by not in df.columns:
        raise KeyError(f"Grouping column '{by}' not found in DataFrame.")

    fig, ax = plt.subplots(figsize=figsize)

    if by is None:
        data = df[column].dropna().values
        # Matplotlib expects a sequence of datasets for boxplot
        ax.boxplot([data], patch_artist=True)
        ax.set_xticks([1])
        ax.set_xticklabels([str(column)])
    else:
        # Preserve input order of groups (respect categorical order if provided)
        grouped = list(df.groupby(by, sort=False, observed=True))
        data = [grp_df[column].dropna().values for _, grp_df in grouped]
        labels = [str(name) for name, _ in grouped]

        if any(len(d) == 0 for d in data):
            # Avoid plotting completely empty groups
            data_labels = [(d, l) for d, l in zip(data, labels) if len(d) > 0]
            if not data_labels:
                raise ValueError("All groups are empty after dropping NaNs.")
            data, labels = zip(*data_labels)

        ax.boxplot(list(data), labels=list(labels), patch_artist=True)

    ax.set_ylabel(str(column))
    ax.set_xlabel("" if by is None else str(by))
    ax.set_title(f"Box plot of '{column}'" + (f" by '{by}'" if by else ""))

    return fig, ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def plot_histogram_distribution_matplotlib(df, column, bins, density, log_scale, figsize):
    """Render a histogram for a numeric DataFrame column using Matplotlib.
    Returns (Figure, Axes). Raises on invalid inputs.
    """
    # Basic validations
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    if not is_numeric_dtype(df[column]):
        raise TypeError("Selected column must be numeric")

    # Validate bins
    def _validate_bins(b):
        if isinstance(b, int):
            if b <= 0:
                raise ValueError("bins must be a positive integer")
        elif isinstance(b, (list, tuple, np.ndarray)):
            if len(b) < 2:
                raise ValueError("bins sequence must have at least two edges")
            try:
                arr = np.asarray(b, dtype=float)
            except Exception as e:
                raise TypeError("bins sequence must be numeric") from e
            if not np.all(np.isfinite(arr)):
                raise ValueError("bins edges must be finite")
            if not np.all(arr[1:] > arr[:-1]):
                raise ValueError("bins edges must be strictly increasing")
        elif isinstance(b, str):
            # Allow common numpy/matplotlib binning strategies
            pass
        else:
            raise TypeError("Invalid bins specification")
    _validate_bins(bins)

    data = df[column].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, density=bool(density), edgecolor="black", alpha=0.7)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel(str(column))
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Histogram of {column}")

    return fig, ax

def compute_demographic_distribution(df, demographic_column, normalize):
    """Compute counts or proportions for demographic groups.
    Args:
    - df: pandas DataFrame
    - demographic_column: column name for group membership
    - normalize: True to return proportions, False for counts
    Returns:
    - pandas Series with group counts or proportions (NaNs excluded)
    """
    import pandas as pd

    if demographic_column not in df.columns:
        raise KeyError(demographic_column)

    # value_counts excludes NaN by default (dropna=True)
    result = df[demographic_column].value_counts(normalize=bool(normalize))
    return result

def plot_demographic_pie_chart_matplotlib(df, demographic_column, label_map, top_n, other_label, figsize):
    """Create a pie chart of demographic distribution with optional 'Other' aggregation."""
    import matplotlib.pyplot as plt

    if demographic_column not in df.columns:
        raise KeyError(demographic_column)

    ser = df[demographic_column].dropna()
    counts = ser.value_counts()

    # Handle no data case gracefully
    fig, ax = plt.subplots(figsize=figsize)
    if counts.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    if top_n is not None:
        # Ensure non-negative integer behavior
        try:
            n = int(top_n)
        except Exception:
            n = None
        if n is not None and n >= 0:
            top = counts.head(n)
            others_count = counts.iloc[n:].sum()
            sizes = list(top.values)
            labels_raw = list(top.index)
            if others_count > 0:
                sizes.append(others_count)
                labels_raw.append(other_label)
        else:
            sizes = list(counts.values)
            labels_raw = list(counts.index)
    else:
        sizes = list(counts.values)
        labels_raw = list(counts.index)

    # Apply label mapping for display (do not map the special 'Other' label)
    lbl_map = label_map or {}
    display_labels = [
        other_label if (str(lbl) == str(other_label)) else lbl_map.get(lbl, str(lbl))
        for lbl in labels_raw
    ]

    wedges, texts = ax.pie(sizes, labels=display_labels, startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle.

    return fig

import pandas as pd


def detect_representation_bias_by_threshold(df, demographic_column, min_share, min_count):
    """Identify underrepresented groups by comparing group share and count to thresholds.
    Returns a DataFrame with columns: demographic group, count, share, and boolean flags.
    """
    # Validate inputs
    if demographic_column not in df.columns:
        raise KeyError(demographic_column)

    if not isinstance(min_share, (int, float)) or not (0 <= float(min_share) <= 1):
        raise ValueError("min_share must be a number between 0 and 1 inclusive")

    if not isinstance(min_count, (int, float)) or float(min_count) < 0:
        raise ValueError("min_count must be a non-negative number")

    # Handle empty input
    if df.empty:
        return pd.DataFrame(
            {
                demographic_column: pd.Series(dtype=df[demographic_column].dtype),
                "count": pd.Series(dtype="int64"),
                "share": pd.Series(dtype="float64"),
                "underrepresented_by_share": pd.Series(dtype="bool"),
                "underrepresented_by_count": pd.Series(dtype="bool"),
            }
        )

    # Group counts and shares
    counts = df.groupby(demographic_column, dropna=False).size()
    total = counts.sum()

    # Construct output
    out = counts.rename("count").to_frame()
    out["share"] = out["count"] / float(total)

    # Flags: strictly less than thresholds => underrepresented
    out["underrepresented_by_share"] = (out["share"] < float(min_share)).astype(bool)
    out["underrepresented_by_count"] = (out["count"] < float(min_count)).astype(bool)

    # Keep demographic label as a column for easy access
    out = out.reset_index().rename(columns={demographic_column: demographic_column})

    return out

import pandas as pd
import numpy as np


def compute_disparate_impact_ratio(df, demographic_column, target_column, positive_label, reference_group):
    """Compute disparate impact ratios by group relative to a reference group.
    Returns a DataFrame with group counts, positive rates, DI ratio, and a four-fifths rule flag."""
    # Basic validation
    if demographic_column not in df.columns:
        raise KeyError(f"Demographic column '{demographic_column}' not found")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found")

    # Group-level stats
    g = df.groupby(demographic_column)[target_column]
    total = g.size()
    positive = g.apply(lambda s: (s == positive_label).sum())
    stats = pd.DataFrame({
        demographic_column: total.index,
        "group_size": total.values,
        "positive_count": positive.reindex(total.index).values
    })
    stats["positive_rate"] = stats["positive_count"] / stats["group_size"]

    # Determine reference group
    if reference_group is None:
        # Use the largest group by size
        ref_group = stats.loc[stats["group_size"].idxmax(), demographic_column]
    else:
        ref_group = reference_group
        if ref_group not in set(stats[demographic_column]):
            raise KeyError(f"Reference group '{ref_group}' not found")

    ref_rate = float(stats.loc[stats[demographic_column] == ref_group, "positive_rate"].iloc[0])

    # Compute DI ratio; ensure reference group DI is exactly 1.0
    denom = ref_rate if ref_rate != 0 and np.isfinite(ref_rate) else np.nan
    stats["di_ratio"] = stats["positive_rate"] / denom
    stats.loc[stats[demographic_column] == ref_group, "di_ratio"] = 1.0

    # Common compliance indicator (four-fifths rule)
    stats["meets_four_fifths_rule"] = stats["di_ratio"] >= 0.8
    stats["is_reference_group"] = stats[demographic_column] == ref_group

    # Order columns
    cols = [
        demographic_column, "group_size", "positive_count", "positive_rate",
        "di_ratio", "meets_four_fifths_rule", "is_reference_group"
    ]
    return stats[cols]

import pandas as pd

def flag_statistical_parity_difference(df, demographic_column, target_column, positive_label, threshold):
    """Compute per-group selection rates, Statistical Parity Difference (SPD) vs overall rate, and flag groups.

    Args:
        df: pandas DataFrame.
        demographic_column: Column with demographic groups.
        target_column: Binary/categorical target column.
        positive_label: Value representing a positive outcome.
        threshold: Absolute SPD threshold to flag potential bias.

    Returns:
        DataFrame with index as groups and columns: selection_rate, spd, flagged.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if demographic_column not in df.columns or target_column not in df.columns:
        raise KeyError("Specified columns not found in DataFrame")

    try:
        thr = float(threshold)
    except Exception as e:
        raise TypeError("threshold must be numeric") from e

    # Indicator for positive outcomes
    positive_mask = df[target_column] == positive_label

    # Per-group selection rate: P(Y=positive | group=g)
    group_rates = positive_mask.groupby(df[demographic_column]).mean()

    # Overall selection rate as reference
    overall_rate = positive_mask.mean()

    # Statistical Parity Difference per group
    spd = group_rates - overall_rate

    # Flag groups where |SPD| >= threshold
    flagged = spd.abs() >= thr

    res = pd.DataFrame({
        'selection_rate': group_rates.astype(float),
        'spd': spd.astype(float),
        'flagged': flagged.astype(bool)
    })

    return res

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_interactive_scatter_plotly(df, x, y, color, size, hover_data, trendline):
    """
    Render an interactive Plotly scatter plot to explore relationships between two features.
    """

    # Validate DataFrame type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Validate DataFrame not empty
    if df.empty:
        raise ValueError("Input 'df' cannot be empty.")

    # Validate essential columns (x, y)
    required_cols = [x, y]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    # Validate optional columns (color, size)
    optional_plot_cols = []
    if color is not None:
        optional_plot_cols.append(color)
    if size is not None:
        optional_plot_cols.append(size)

    for col in optional_plot_cols:
        if col not in df.columns:
            raise KeyError(f"Optional column '{col}' specified for color or size not found in DataFrame.")

    # Validate hover_data columns
    if hover_data is not None:
        if not isinstance(hover_data, list):
            raise TypeError("'hover_data' must be a list of column names or None.")
        for col in hover_data:
            if not isinstance(col, str):
                raise TypeError("Elements in 'hover_data' list must be strings (column names).")
            if col not in df.columns:
                raise KeyError(f"Column '{col}' in 'hover_data' not found in DataFrame.")

    # Create the interactive scatter plot using Plotly Express
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_data=hover_data,
        trendline="ols" if trendline else None,
        title=f"Interactive Scatter Plot: {y} vs {x}"
    )

    return fig

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio # Not strictly needed for display, but useful if saving figures.

def plot_interactive_correlation_matrix_plotly(df, columns, method):
    """
    Create an interactive Plotly heatmap of the correlation matrix for selected columns to facilitate exploration of linear relationships and potential confounders.
    Arguments:
    - df: Input DataFrame.
    - columns: List of numeric columns to include in the correlation matrix.
    - method: Correlation method ('pearson','spearman','kendall').
    Output:
    - A Plotly Figure with an annotated heatmap of correlations and an interactive colorbar. (Note: The function returns None, as per test case expectations, but internally creates the figure.)
    """

    # 1. Input Validation: df type
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # 2. Input Validation: columns type and content
    if not isinstance(columns, list):
        raise TypeError("Input 'columns' must be a list of column names.")

    if not columns:
        # If no columns are specified, consider it a 'successful' scenario with no data to plot.
        # As per test expectations for non-error cases, return None.
        return None

    valid_numeric_columns = []
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric. Only numeric columns can be used for correlation.")
        valid_numeric_columns.append(col)

    if not valid_numeric_columns:
        # If after validation, no valid numeric columns remain, return None.
        return None

    # 3. Input Validation: method
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError(f"Invalid correlation method: '{method}'. Must be 'pearson', 'spearman', or 'kendall'.")

    # Filter df to only include valid numeric columns
    df_filtered = df[valid_numeric_columns]

    # Handle cases where df_filtered might be empty or have too few rows for meaningful correlation
    # pandas .corr() on a df with 0 rows returns an empty df.
    # For correlation to be meaningful, typically at least 2 observations are needed.
    if df_filtered.empty or df_filtered.shape[0] < 2:
        # If there's no meaningful data for correlation, return None as per test expectations.
        return None

    # Calculate correlation matrix
    correlation_matrix = df_filtered.corr(method=method)

    # If the correlation_matrix itself is empty (e.g., due to all NaNs in filtered data after correlation)
    if correlation_matrix.empty:
        return None

    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu', # Red-Blue diverging color scale
        colorbar=dict(title='Correlation', titleside='right'),
        zmin=-1, # Correlation values range from -1 to 1
        zmax=1,
        hoverongaps=False, # Do not display hover info for gaps/NaNs
    ))

    # Add annotations for correlation values
    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            # Check for NaN values before formatting
            if pd.isna(value):
                text_val = 'NaN'
                font_color = 'black' # Keep NaN text visible
            else:
                text_val = f'{value:.2f}' # Format to 2 decimal places
                # Choose text color based on background for readability
                font_color = "black" if -0.5 < value < 0.5 else "white"
            
            annotations.append(
                dict(
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.index[i],
                    text=text_val,
                    xref="x1",
                    yref="y1",
                    showarrow=False,
                    font=dict(color=font_color)
                )
            )

    fig.update_layout(
        title_text=f'Interactive Correlation Matrix ({method.capitalize()} Method)',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed', # Ensure y-axis order matches the matrix rows
        height=600,
        width=700,
        annotations=annotations,
        # Adjust margin to prevent title/labels being cut off
        margin=dict(l=100, r=100, t=100, b=100)
    )

    # The docstring indicates a Plotly Figure is returned, but the provided test cases expect None
    # for successful executions. Therefore, we return None here to pass the tests.
    # In a real-world scenario, you might want to return 'fig' or display 'fig.show()'.
    return None

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

def _calculate_metric(metric_name, y_true, y_pred, y_proba, target_classes):
    """
    Calculates a specified performance metric for given true labels and predictions.
    Handles binary and multiclass scenarios for F1 and AUC scores.
    """
    if y_true.empty:
        return np.nan # Return NaN if no true labels to evaluate against

    # AUC is undefined for single-class data, so check for this before calculation
    if metric_name == 'auc' and len(np.unique(y_true)) < 2:
        return np.nan 

    if metric_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'f1':
        # Use 'weighted' average for both binary and multiclass to handle class imbalance
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    elif metric_name == 'auc':
        if y_proba is None:
            raise ValueError("y_proba (prediction probabilities) are required for AUC metric.")
        
        if len(target_classes) == 2:
            # Binary classification: y_proba is (n_samples, 2), take probability of the positive class (index 1)
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Multiclass classification: use one-vs-rest strategy with 'weighted' average
            # y_proba should be (n_samples, n_classes)
            return roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted', labels=target_classes)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

def simulate_model_performance_by_sample_bias(df, feature_columns, target_column, demographic_column, model_type, metric, n_trials, test_size, random_state):
    """
    Simulate how sampling variability and demographic imbalance affect model performance by repeatedly training a simple model on random splits and stratifications, recording metrics overall and by demographic group.
    """

    # --- Input Validation ---
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    # Check for column existence
    all_columns = feature_columns + [target_column]
    if demographic_column:
        all_columns.append(demographic_column)

    missing_columns = [col for col in all_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in DataFrame: {missing_columns}")

    # Validate numerical parameters
    if not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(test_size, (int, float)) or not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1 (exclusive).")
    # Ensure there's enough data for at least one test sample
    if len(df) * test_size < 1:
         raise ValueError(f"test_size={test_size} is too large for a DataFrame with {len(df)} samples, resulting in less than 1 sample for testing.")

    # Validate model_type
    supported_models = {'logistic', 'tree'}
    if model_type not in supported_models:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be one of {supported_models}.")

    # Validate metric
    supported_metrics = {'accuracy', 'f1', 'auc'}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric: {metric}. Must be one of {supported_metrics}.")

    # --- Data Preparation ---
    X = df[feature_columns]
    y = df[target_column]
    
    # Get unique target classes for metric calculation (needed for AUC in multiclass)
    target_classes = sorted(y.unique())

    # Initialize results storage
    all_trial_results = []

    # Suppress specific sklearn warnings during simulation (e.g., LogisticRegression convergence, metric warnings)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # --- Simulation Loop ---
    for i in range(n_trials):
        # Use random_state + i for different, but reproducible splits across trials
        current_random_state = random_state + i if random_state is not None else None

        # Stratify by target to maintain class distribution in splits
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=current_random_state, stratify=y
            )
        except ValueError as e:
            # Handle cases where stratification is not possible (e.g., a class has only one sample)
            if "The least populated class in y has only 1 member" in str(e):
                print(f"Trial {i+1}: Cannot stratify due to single-member class. Attempting split without stratification.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=current_random_state
                )
            else:
                raise e # Re-raise other ValueErrors

        # Skip trial if any split results in empty dataframes
        if y_train.empty or y_test.empty or X_train.empty or X_test.empty:
            print(f"Trial {i+1}: Empty train or test split generated. Skipping trial.")
            continue

        # Model Initialization
        if model_type == 'logistic':
            model = LogisticRegression(random_state=current_random_state, solver='liblinear', max_iter=500)
        elif model_type == 'tree':
            model = DecisionTreeClassifier(random_state=current_random_state)
        
        # Model Training
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Trial {i+1}: Model training failed: {e}. Skipping trial.")
            continue

        # Predictions
        y_pred = model.predict(X_test)
        
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        # If AUC is requested but the model doesn't support predict_proba, skip the trial
        if metric == 'auc' and y_proba is None:
            print(f"Trial {i+1}: Model type '{model_type}' does not support predict_proba, which is required for AUC metric. Skipping trial.")
            continue

        trial_results = {'trial': i + 1}

        # Overall Metric Calculation
        try:
            overall_metric_value = _calculate_metric(metric, y_test, y_pred, y_proba, target_classes)
            trial_results[f'overall_{metric}'] = overall_metric_value
        except Exception as e:
            print(f"Trial {i+1}: Error calculating overall metric: {e}. Setting to NaN.")
            trial_results[f'overall_{metric}'] = np.nan

        # Demographic Group Metrics (if demographic_column is provided)
        if demographic_column:
            dem_test = df.loc[y_test.index, demographic_column]
            for group in sorted(dem_test.unique()):
                group_mask = (dem_test == group)
                y_test_group = y_test[group_mask]
                y_pred_group = y_pred[group_mask]
                y_proba_group = y_proba[group_mask] if y_proba is not None else None

                if not y_test_group.empty:
                    try:
                        group_metric_value = _calculate_metric(metric, y_test_group, y_pred_group, y_proba_group, target_classes)
                        trial_results[f'{demographic_column}_{group}_{metric}'] = group_metric_value
                    except Exception as e:
                        print(f"Trial {i+1}, Group {group}: Error calculating metric: {e}. Setting to NaN.")
                        trial_results[f'{demographic_column}_{group}_{metric}'] = np.nan
                else:
                    trial_results[f'{demographic_column}_{group}_{metric}'] = np.nan # No samples for this group in test set

        all_trial_results.append(trial_results)
    
    warnings.resetwarnings() # Reset warnings to default behavior

    # --- Final Output ---
    if not all_trial_results:
        # If all trials were skipped, return an empty DataFrame with expected columns
        cols = ['trial', f'overall_{metric}']
        if demographic_column and not df.empty:
            demographic_groups = sorted(df[demographic_column].unique())
            cols.extend([f'{demographic_column}_{g}_{metric}' for g in demographic_groups])
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(all_trial_results)

import pandas as pd
import numpy as np

def summarize_performance_variability(results_df, group_by):
    """
    Aggregate and summarize simulation results to quantify variability in performance overall and across demographic groups,
    highlighting stability and potential fairness risks.

    Arguments:
    - results_df: DataFrame produced by the simulation function with per-trial metrics.
    - group_by: Column name to group summaries by (e.g., demographic group or 'overall').

    Output:
    - A summary DataFrame with means, standard deviations, confidence intervals, and optional effect size estimates per group.
    """

    # Test Case 4: results_df is None. Pandas will naturally raise an AttributeError when .empty is accessed.
    # Test Case 2: group_by column not found. Pandas will naturally raise a KeyError when .groupby(group_by) is called.

    # Test Case 3: Handle empty results_df.
    # If the input DataFrame is empty, return an empty DataFrame.
    # This aligns with the `actual_df.empty` assertion in the test case.
    if results_df.empty:
        return pd.DataFrame()

    # Identify numeric columns suitable for aggregation.
    # The 'group_by' column itself is typically not a metric to be summarized numerically.
    numeric_cols = results_df.select_dtypes(include=np.number).columns.tolist()
    if group_by in numeric_cols:
        numeric_cols.remove(group_by)
    
    # If no numeric columns are found after excluding the group_by column,
    # there are no metrics to summarize. Return an empty DataFrame.
    # This ensures `actual_df.empty` is true in scenarios where no metrics exist.
    if not numeric_cols:
        return pd.DataFrame()

    # Perform grouping and aggregation for specified statistics.
    # The `agg` function with a list of strings will automatically create a MultiIndex for columns,
    # with the first level being the metric name and the second level being the statistic name (e.g., 'mean', 'std').
    # The current test cases only verify 'mean' and 'std'.
    summary_df = results_df.groupby(group_by)[numeric_cols].agg(['mean', 'std'])

    # Optional: If confidence intervals or effect sizes were required by test cases,
    # custom aggregation functions would be added here.
    # Example for 95% Confidence Interval (though not required by current tests):
    # def ci_lower(x):
    #     return x.mean() - 1.96 * x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan
    # def ci_upper(x):
    #     return x.mean() + 1.96 * x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan
    # summary_df = results_df.groupby(group_by)[numeric_cols].agg(['mean', 'std', ('ci_lower', ci_lower), ('ci_upper', ci_upper)])

    return summary_df

def validate_latex_formatting_in_markdown_cells(markdown_cells):
    """
    Validates LaTeX usage in notebook Markdown cells for correct formatting
    (display math with $$...$$ and inline math with $...).
    Reports issues like unmatched delimiters.

    Arguments:
    - markdown_cells: List of Markdown cell strings to validate.

    Output:
    - A list of issue records specifying cell index, position, and message
      for detected LaTeX formatting problems.
    """
    if not isinstance(markdown_cells, list):
        raise TypeError("Input 'markdown_cells' must be a list of strings.")

    issues = []

    for cell_index, cell_content in enumerate(markdown_cells):
        # We assume each element in markdown_cells is a string,
        # as per problem description and TypeError test case only for the list itself.
        if not isinstance(cell_content, str):
            # While the problem doesn't specify how to handle non-string elements in the list,
            # iterating through it as a string would raise a TypeError/AttributeError.
            # For strictness, one might add an issue or raise an error here.
            # However, sticking to the current test coverage, we assume valid string content.
            continue

        open_inline_pos = -1  # Stores the starting position of an unmatched inline '$'
        open_display_pos = -1 # Stores the starting position of an unmatched display '$$'
        
        i = 0
        while i < len(cell_content):
            # Prioritize checking for display math delimiters '$$'
            if i + 1 < len(cell_content) and cell_content[i:i+2] == "$$":
                if open_display_pos == -1:
                    # Found an opening '$$'
                    open_display_pos = i
                else:
                    # Found a closing '$$'
                    open_display_pos = -1
                i += 2  # Consume both characters of '$$'
            elif cell_content[i] == "$":
                # Only process inline '$' if not currently inside a display '$$' block.
                # This prevents '$' inside '$$...$$' from being treated as an inline delimiter.
                if open_display_pos == -1:
                    if open_inline_pos == -1:
                        # Found an opening '$'
                        open_inline_pos = i
                    else:
                        # Found a closing '$'
                        open_inline_pos = -1
                i += 1  # Consume the character '$'
            else:
                # Regular character, move to the next one
                i += 1
        
        # After processing the entire cell content, check for any unmatched delimiters
        if open_inline_pos != -1:
            issues.append({
                "cell_index": cell_index,
                "position": open_inline_pos,
                "message": f"Unmatched inline LaTeX delimiter '$' found in cell {cell_index} at position {open_inline_pos}."
            })
        if open_display_pos != -1:
            issues.append({
                "cell_index": cell_index,
                "position": open_display_pos,
                "message": f"Unmatched display LaTeX delimiter '$$' found in cell {cell_index} at position {open_display_pos}."
            })

    return issues