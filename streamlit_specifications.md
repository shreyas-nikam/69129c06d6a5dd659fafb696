
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application will serve as an interactive tool for visualizing and analyzing datasets, with a particular focus on demographic representation and bias detection. It aims to provide users with an intuitive interface to explore data distributions, identify outliers, and understand relationships between features.

**Learning Goals:**
*   To enable users to interactively select and import datasets for analysis.
*   To facilitate the exploration of dataset characteristics through various visualizations.
*   To introduce methods for detecting potential biases within datasets, especially concerning demographic attributes.
*   To explain underlying data analysis concepts and their implications using clear markdown.
*   To demonstrate how dataset characteristics and biases can impact model performance.
*   To provide resources for further learning on data bias detection.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will feature a clear two-column layout:
*   **Sidebar:** Will host input controls for dataset selection, feature selection, and analysis options.
*   **Main Content Area:** Will display markdown explanations, interactive visualizations, and summary insights.
Sections will be presented sequentially in the main content area, guided by user interactions in the sidebar.

### Input Widgets and Controls
*   **Dataset Selection:**
    *   `st.file_uploader`: For users to upload their own CSV or Excel datasets.
    *   `st.selectbox`: (Optional, for demonstration) To select from pre-loaded sample datasets.
*   **Data Loading & Preview:**
    *   `st.button`: To trigger data loading after file upload.
    *   `st.dataframe`: To display a preview of the loaded dataset (e.g., first 5 rows and data types).
*   **Feature Selection for Visualizations:**
    *   `st.selectbox` / `st.multiselect`: For selecting features for histograms, pie charts, box plots, and scatter plots.
*   **Visualization Parameters:**
    *   `st.slider` / `st.number_input`: For adjusting bin sizes in histograms.
    *   `st.checkbox`: To toggle interactive features or specific plot overlays.

### Visualization Components (Charts, Graphs, Tables)
The main content area will dynamically display the following interactive visualizations:
*   **Dataset Overview Table:** `st.dataframe` to show basic statistics (`df.describe()`) and missing values.
*   **Histograms:** `plotly.express.histogram` or `matplotlib.pyplot.hist` for displaying data distribution of selected numerical features.
*   **Pie Charts:** `plotly.express.pie` for demographic representation of categorical features.
*   **Box Plots:** `plotly.express.box` or `seaborn.boxplot` for outlier detection in numerical features, optionally grouped by demographic features.
*   **Scatter Plots:** `plotly.express.scatter` or `matplotlib.pyplot.scatter` to visualize relationships between two selected numerical features, with optional coloring by a categorical feature.

### Interactive Elements and Feedback Mechanisms
*   **Dynamic Updates:** All visualizations will update automatically as users change input selections (e.g., selected feature, plot type, bin size).
*   **Progress Indicators:** `st.spinner` or `st.progress` will be used during computationally intensive operations like data loading or complex plot generation.
*   **Error Handling:** `st.error` messages will be displayed for invalid inputs (e.g., non-numeric data selected for a numerical plot) or failed file uploads.
*   **Informative Text:** `st.info` or `st.warning` for guiding users or highlighting potential issues.

## 3. Additional Requirements

*   **Annotation and Tooltip Specifications:**
    *   All interactive plots (Plotly) will feature built-in tooltips on hover, displaying relevant data points and values.
    *   Markdown cells (`st.markdown`) will provide contextual annotations and explanations for each visualization and analysis step.
*   **State Management:**
    *   `st.session_state` will be extensively used to preserve the state of user inputs (e.g., uploaded dataset, selected features, plot configurations) across reruns, ensuring changes are not lost when new selections are made.
    *   This includes the loaded DataFrame, selected column names, and any other user-defined parameters.

## 4. Notebook Content and Code Requirements

**Note on Code Stubs:** The provided "Jupyter Notebook Content Summary" only included a license. The OCR content was for a `ModelCardGenerator` command-line tool, which does not align with the "interactive Jupyter Notebook to visualize and analyze datasets" described in the **User Requirements**. Therefore, the code stubs below are conceptual, generated based on the **User Requirements** for data visualization and bias detection, and represent what *would* be present in a Jupyter notebook fulfilling those requirements.

### Data Loading and Initial Exploration
**Markdown Explanation:**
`st.markdown("""
### Data Loading and Initial Exploration
This section allows you to load your dataset and get a first look at its structure and basic statistics.
""")`

**Code Stub:**
```python
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) # Or pd.read_excel depending on file type
            st.success("Dataset loaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

# Example usage in Streamlit:
# uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])
# if uploaded_file:
#     df = load_data(uploaded_file)
#     if df is not None:
#         st.subheader("Dataset Preview")
#         st.dataframe(df.head())
#         st.subheader("Dataset Information")
#         st.write(df.describe())
#         st.write("Missing values per column:")
#         st.dataframe(df.isnull().sum().to_frame(name='Missing Values'))
```

### Data Distribution Analysis (Histograms)
**Markdown Explanation:**
`st.markdown("""
### Data Distribution Analysis
Histograms visualize the distribution of numerical features, helping identify patterns and concentrations in the data.
The formula for a normal distribution's Probability Density Function (PDF) is:
$$ f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2} $$
where $\\mu$ is the mean and $\\sigma$ is the standard deviation.
""")`

**Code Stub:**
```python
import plotly.express as px
import streamlit as st

def plot_histogram(df, column, bins=20):
    fig = px.histogram(df, x=column, nbins=bins,
                       title=f'Distribution of {column}',
                       labels={column: column, 'count': 'Frequency'})
    st.plotly_chart(fig, use_container_width=True)

# Example usage in Streamlit:
# if df is not None:
#     numerical_cols = df.select_dtypes(include=['number']).columns
#     if not numerical_cols.empty:
#         selected_num_col = st.sidebar.selectbox("Select numerical feature for histogram:", numerical_cols)
#         bin_size = st.sidebar.slider("Number of bins:", 5, 100, 20)
#         plot_histogram(df, selected_num_col, bin_size)
```

### Demographic Representation (Pie Charts)
**Markdown Explanation:**
`st.markdown("""
### Demographic Representation
Pie charts illustrate the proportional representation of different categories within a demographic feature.
For a category $i$, its proportion $P_i$ is given by:
$$ P_i = \\frac{\\text{Count of category } i}{\\text{Total observations}} $$
""")`

**Code Stub:**
```python
import plotly.express as px
import streamlit as st

def plot_pie_chart(df, column):
    fig = px.pie(df, names=column, title=f'Demographic Representation of {column}')
    st.plotly_chart(fig, use_container_width=True)

# Example usage in Streamlit:
# if df is not None:
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
#     if not categorical_cols.empty:
#         selected_cat_col = st.sidebar.selectbox("Select categorical feature for pie chart:", categorical_cols)
#         plot_pie_chart(df, selected_cat_col)
```

### Outlier Detection (Box Plots)
**Markdown Explanation:**
`st.markdown("""
### Outlier Detection
Box plots are effective for visualizing the distribution of numerical data and identifying potential outliers.
The Interquartile Range (IQR) is calculated as $IQR = Q_3 - Q_1$, where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.
Outliers are typically defined as data points below $Q_1 - 1.5 \\times IQR$ or above $Q_3 + 1.5 \\times IQR$.
""")`

**Code Stub:**
```python
import plotly.express as px
import streamlit as st

def plot_box_plot(df, column, group_by=None):
    if group_by and group_by in df.columns:
        fig = px.box(df, x=group_by, y=column,
                     title=f'Box Plot of {column} grouped by {group_by}')
    else:
        fig = px.box(df, y=column, title=f'Box Plot of {column}')
    st.plotly_chart(fig, use_container_width=True)

# Example usage in Streamlit:
# if df is not None:
#     numerical_cols = df.select_dtypes(include=['number']).columns
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
#     if not numerical_cols.empty:
#         selected_num_col_box = st.sidebar.selectbox("Select numerical feature for box plot:", numerical_cols, key="box_num")
#         group_by_col = st.sidebar.selectbox("Group box plot by (optional):", ['None'] + list(categorical_cols), key="box_group")
#         plot_box_plot(df, selected_num_col_box, group_by=None if group_by_col == 'None' else group_by_col)
```

### Feature Relationships (Scatter Plots)
**Markdown Explanation:**
`st.markdown("""
### Feature Relationships
Scatter plots visualize the relationship between two numerical features, helping to identify correlations or clusters.
A linear correlation coefficient $r$ between variables $X$ and $Y$ is given by:
$$ r = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2 \\sum_{i=1}^{n} (y_i - \\bar{y})^2}} $$
where $n$ is the number of observations, $x_i, y_i$ are individual data points, and $\\bar{x}, \\bar{y}$ are the means.
""")`

**Code Stub:**
```python
import plotly.express as px
import streamlit as st

def plot_scatter_plot(df, x_column, y_column, color_by=None):
    if color_by and color_by in df.columns:
        fig = px.scatter(df, x=x_column, y=y_column, color=color_by,
                         title=f'Scatter Plot of {x_column} vs. {y_column} colored by {color_by}')
    else:
        fig = px.scatter(df, x=x_column, y=y_column,
                         title=f'Scatter Plot of {x_column} vs. {y_column}')
    st.plotly_chart(fig, use_container_width=True)

# Example usage in Streamlit:
# if df is not None:
#     numerical_cols = df.select_dtypes(include=['number']).columns
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
#     if len(numerical_cols) >= 2:
#         x_col = st.sidebar.selectbox("Select X-axis feature:", numerical_cols, key="scatter_x")
#         y_col = st.sidebar.selectbox("Select Y-axis feature:", numerical_cols, key="scatter_y")
#         color_col = st.sidebar.selectbox("Color by (optional):", ['None'] + list(categorical_cols), key="scatter_color")
#         plot_scatter_plot(df, x_col, y_col, color_by=None if color_col == 'None' else color_col)
```

### Bias Detection and Summary Insights
**Markdown Explanation:**
`st.markdown("""
### Bias Detection and Summary Insights
This section focuses on identifying and understanding potential biases in the dataset, particularly concerning demographic attributes.
**Definition of Bias:** Data bias refers to systemic errors in the data collection process or intrinsic properties of the data that lead to unfair or inaccurate outcomes for certain groups or individuals.

*   **Impact on Model Performance:** Biased data can lead to models that perform poorly for underrepresented groups, perpetuate stereotypes, or make unfair predictions. For example, if a dataset is skewed towards a particular demographic, a model trained on it might exhibit lower accuracy or higher error rates for other demographics.
*   **Mitigation Strategies:** Techniques like re-sampling, re-weighting, and adversarial debiasing can be used.
*   **Further Resources:**
    *   [Awesome-Fairness-in-AI](https://github.com/EthicalML/awesome-fairness-in-ai)
    *   [Google AI's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
""")`

**Code Stub (Conceptual - will depend on specific bias metrics):**
```python
# No direct code stub for 'bias detection' without specific algorithms/metrics
# This section would primarily involve interpreting visualizations and possibly
# calculating fairness metrics if demographic data is clearly defined and sensitive attributes identified.
# Example: Calculating proportions and comparing them to population benchmarks.

# Example placeholder for a simple bias check (e.g., proportion comparison)
def check_demographic_balance(df, demographic_column):
    if demographic_column in df.columns:
        st.subheader(f"Demographic Balance for {demographic_column}")
        counts = df[demographic_column].value_counts(normalize=True)
        st.dataframe(counts.to_frame(name='Proportion'))
        st.info("Compare these proportions to real-world demographics to identify under/over-representation.")
    else:
        st.warning("Demographic column not found for bias check.")

# Example usage in Streamlit:
# if df is not None:
#     demographic_cols = [col for col in categorical_cols if col.lower() in ['gender', 'race', 'ethnicity', 'age_group']]
#     if demographic_cols:
#         selected_demographic = st.sidebar.selectbox("Select demographic for bias check:", demographic_cols)
#         check_demographic_balance(df, selected_demographic)
#     else:
#         st.info("No common demographic columns found for automated bias check. Please analyze manually using charts.")
```
