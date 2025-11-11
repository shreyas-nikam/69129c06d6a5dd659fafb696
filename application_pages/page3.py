"""application_pages/page3.py"""
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_box_plot(df, column, group_by=None):
    if group_by and group_by in df.columns:
        fig = px.box(df, x=group_by, y=column,
                     title=f'Box Plot of {column} grouped by {group_by}',
                     template="plotly_white")
    else:
        fig = px.box(df, y=column, title=f'Box Plot of {column}', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_plot(df, x_column, y_column, color_by=None):
    if x_column == y_column:
        st.warning("X-axis and Y-axis features cannot be the same for a scatter plot.")
        return

    if color_by and color_by in df.columns:
        fig = px.scatter(df, x=x_column, y=y_column, color=color_by,
                         title=f'Scatter Plot of {x_column} vs. {y_column} colored by {color_by}',
                         template="plotly_white")
    else:
        fig = px.scatter(df, x=x_column, y=y_column,
                         title=f'Scatter Plot of {x_column} vs. {y_column}',
                         template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def check_demographic_balance(df, demographic_column):
    if demographic_column in df.columns:
        st.subheader(f"Demographic Balance for '{demographic_column}'")
        counts = df[demographic_column].value_counts(normalize=True)
        st.dataframe(counts.to_frame(name='Proportion'))
        st.info("Compare these proportions to real-world demographics or expected distributions to identify under/over-representation.")
    else:
        st.warning(f"Demographic column '{demographic_column}' not found for bias check.")

def run_page3():
    st.header("Page 3: Outlier Detection, Feature Relationships & Bias Insights")

    if 'df' not in st.session_state:
        st.warning("Please load a dataset on 'Data Loading & Exploration' page first.")
        return

    df = st.session_state['df']

    # Outlier Detection (Box Plots)
    st.markdown("""
    ### Outlier Detection (Box Plots)
    Box plots are effective for visualizing the distribution of numerical data and identifying potential outliers. They display the five-number summary: minimum, first quartile ($Q_1$), median ($Q_2$), third quartile ($Q_3$), and maximum.
    The Interquartile Range (IQR) is calculated as $IQR = Q_3 - Q_1$, where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.
    Outliers are typically defined as data points below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$.
    """)

    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    if not numerical_cols.empty:
        selected_num_col_box = st.sidebar.selectbox(
            "Select numerical feature for box plot:",
            numerical_cols,
            key="box_num_col"
        )
        group_by_col_box = st.sidebar.selectbox(
            "Group box plot by (optional categorical feature):",
            ['None'] + list(categorical_cols),
            key="box_group_col"
        )
        plot_box_plot(df, selected_num_col_box, group_by=None if group_by_col_box == 'None' else group_by_col_box)
    else:
        st.info("No numerical columns found for box plot analysis.")

    st.divider()

    # Feature Relationships (Scatter Plots)
    st.markdown("""
    ### Feature Relationships (Scatter Plots)
    Scatter plots visualize the relationship between two numerical features, helping to identify correlations or clusters. They are powerful for exploring potential dependencies between variables.
    A linear correlation coefficient $r$ between variables $X$ and $Y$ is given by:
    $$ r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}} $$
    where $n$ is the number of observations, $x_i, y_i$ are individual data points, and $\bar{x}, \bar{y}$ are the means.
    """)

    if len(numerical_cols) >= 2:
        x_col_scatter = st.sidebar.selectbox(
            "Select X-axis feature for scatter plot:",
            numerical_cols,
            key="scatter_x_col"
        )
        y_col_scatter = st.sidebar.selectbox(
            "Select Y-axis feature for scatter plot:",
            [col for col in numerical_cols if col != x_col_scatter], # Exclude selected x_col
            key="scatter_y_col"
        )
        color_by_col_scatter = st.sidebar.selectbox(
            "Color scatter plot by (optional categorical feature):",
            ['None'] + list(categorical_cols),
            key="scatter_color_col"
        )
        plot_scatter_plot(df, x_col_scatter, y_col_scatter,
                          color_by=None if color_by_col_scatter == 'None' else color_by_col_scatter)
    else:
        st.info("Need at least two numerical columns for scatter plot analysis.")

    st.divider()

    # Bias Detection and Summary Insights
    st.markdown("""
    ### Bias Detection and Summary Insights
    This section focuses on identifying and understanding potential biases in the dataset, particularly concerning demographic attributes. Understanding these biases is crucial for building fair and ethical AI models.
    **Definition of Bias:** Data bias refers to systemic errors in the data collection process or intrinsic properties of the data that lead to unfair or inaccurate outcomes for certain groups or individuals.

    *   **Impact on Model Performance:** Biased data can lead to models that perform poorly for underrepresented groups, perpetuate stereotypes, or make unfair predictions. For example, if a dataset is skewed towards a particular demographic, a model trained on it might exhibit lower accuracy or higher error rates for other demographics.
    *   **Mitigation Strategies:** Techniques like re-sampling, re-weighting, and adversarial debiasing can be used to address biases. Examples include:
        *   **Re-sampling:** Adjusting the number of samples in over-represented or under-represented groups.
        *   **Re-weighting:** Assigning different weights to data points to balance their influence during model training.
        *   **Adversarial Debiasing:** Training an adversarial network to remove sensitive attribute information from feature representations.
    *   **Further Resources:**
        *   [Awesome-Fairness-in-AI](https://github.com/EthicalML/awesome-fairness-in-ai)
        *   [Google AI's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
    """)

    # Example placeholder for a simple bias check (e.g., proportion comparison)
    potential_demographic_cols = [col for col in categorical_cols if col.lower() in ['gender', 'race', 'ethnicity', 'age_group', 'sex']]
    if potential_demographic_cols:
        selected_demographic_bias = st.sidebar.selectbox(
            "Select demographic feature for bias check:",
            potential_demographic_cols,
            key="bias_demographic_col"
        )
        check_demographic_balance(df, selected_demographic_bias)
    else:
        st.info("No common demographic columns (e.g., 'gender', 'race') found for automated bias check. Please analyze manually using the charts provided.")
