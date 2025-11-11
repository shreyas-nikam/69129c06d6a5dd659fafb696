
import streamlit as st
import pandas as pd
import plotly.express as px

def plot_box_plot(df, column, group_by=None):
    if group_by and group_by in df.columns and group_by != "None":
        fig = px.box(df, x=group_by, y=column,
                     title=f'Box Plot of {column} grouped by {group_by}',
                     labels={column: column, group_by: group_by})
    else:
        fig = px.box(df, y=column, title=f'Box Plot of {column}',
                     labels={column: column})
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_plot(df, x_column, y_column, color_by=None):
    if color_by and color_by in df.columns and color_by != "None":
        fig = px.scatter(df, x=x_column, y=y_column, color=color_by,
                         title=f'Scatter Plot of {x_column} vs. {y_column} colored by {color_by}',
                         labels={x_column: x_column, y_column: y_column, color_by: color_by})
    else:
        fig = px.scatter(df, x=x_column, y=y_column,
                         title=f'Scatter Plot of {x_column} vs. {y_column}',
                         labels={x_column: x_column, y_column: y_column})
    st.plotly_chart(fig, use_container_width=True)

def check_demographic_balance(df, demographic_column):
    if demographic_column in df.columns:
        st.subheader(f"Demographic Balance for {demographic_column}")
        counts = df[demographic_column].value_counts(normalize=True)
        st.dataframe(counts.to_frame(name='Proportion'))
        st.info("Compare these proportions to real-world demographics to identify under/over-representation.")
    else:
        st.warning("Demographic column not found for bias check.")

def run_page3():
    st.title("Outliers & Relationships")
    st.markdown("""
    This page focuses on identifying outliers in your data and visualizing relationships between different features.
    It also provides insights into potential biases within the dataset.
    """)

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

        # Outlier Detection (Box Plots)
        st.markdown("""
        ### Outlier Detection
        Box plots are effective for visualizing the distribution of numerical data and identifying potential outliers.
        The Interquartile Range (IQR) is calculated as $IQR = Q_3 - Q_1$, where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.
        Outliers are typically defined as data points below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$.
        """)

        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if not numerical_cols.empty:
            selected_num_col_box = st.sidebar.selectbox("Select numerical feature for box plot:", numerical_cols, key="box_num")
            group_by_col = st.sidebar.selectbox("Group box plot by (optional):", ['None'] + list(categorical_cols), key="box_group")
            plot_box_plot(df, selected_num_col_box, group_by=None if group_by_col == 'None' else group_by_col)
        else:
            st.warning("No numerical columns found for box plot analysis.")

        st.divider()

        # Feature Relationships (Scatter Plots)
        st.markdown("""
        ### Feature Relationships
        Scatter plots visualize the relationship between two numerical features, helping to identify correlations or clusters.
        A linear correlation coefficient $r$ between variables $X$ and $Y$ is given by:
        $$ r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}} $$
        where $n$ is the number of observations, $x_i, y_i$ are individual data points, and $\bar{x}, \bar{y}$ are the means.
        """)

        if len(numerical_cols) >= 2:
            x_col = st.sidebar.selectbox("Select X-axis feature for scatter plot:", numerical_cols, key="scatter_x")
            y_col = st.sidebar.selectbox("Select Y-axis feature for scatter plot:", numerical_cols, key="scatter_y")
            color_by_col = st.sidebar.selectbox("Color scatter plot by (optional):", ['None'] + list(categorical_cols), key="scatter_color")
            plot_scatter_plot(df, x_col, y_col, color_by=None if color_by_col == 'None' else color_by_col)
        elif len(numerical_cols) == 1:
            st.info("Need at least two numerical columns to create a scatter plot.")
        else:
            st.warning("No numerical columns found for scatter plot analysis.")

        st.divider()

        # Bias Detection and Summary Insights
        st.markdown("""
        ### Bias Detection and Summary Insights
        This section focuses on identifying and understanding potential biases in the dataset, particularly concerning demographic attributes.
        **Definition of Bias:** Data bias refers to systemic errors in the data collection process or intrinsic properties of the data that lead to unfair or inaccurate outcomes for certain groups or individuals.

        *   **Impact on Model Performance:** Biased data can lead to models that perform poorly for underrepresented groups, perpetuate stereotypes, or make unfair predictions. For example, if a dataset is skewed towards a particular demographic, a model trained on it might exhibit lower accuracy or higher error rates for other demographics.
        *   **Mitigation Strategies:** Techniques like re-sampling, re-weighting, and adversarial debiasing can be used.
        *   **Further Resources:**
            *   [Awesome-Fairness-in-AI](https://github.com/EthicalML/awesome-fairness-in-ai)
            *   [Google AI's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
        """)

        if not categorical_cols.empty:
            demographic_cols_for_check = [col for col in categorical_cols if col.lower() in ['gender', 'race', 'ethnicity', 'age_group', 'country', 'region']]
            if demographic_cols_for_check:
                selected_demographic_for_bias = st.sidebar.selectbox("Select demographic for bias check:", demographic_cols_for_check, key="bias_check_col")
                check_demographic_balance(df, selected_demographic_for_bias)
            else:
                st.info("No common demographic columns (e.g., 'gender', 'race', 'ethnicity', 'age_group') found for automated bias check. Please analyze manually using charts.")
        else:
            st.warning("No categorical columns available for demographic bias checks.")

    else:
        st.info("Please load a dataset on the 'Data Loading & Exploration' page to perform outlier and relationship analysis.")
