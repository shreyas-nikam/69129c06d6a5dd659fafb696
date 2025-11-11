"""application_pages/page2.py"""
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_histogram(df, column, bins=20):
    fig = px.histogram(df, x=column, nbins=bins,
                       title=f'Distribution of {column}',
                       labels={column: column, 'count': 'Frequency'},
                       template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(df, column):
    # Ensure the column has enough distinct values for a meaningful pie chart
    if df[column].nunique() > 20: # Limit categories for better visualization
        st.warning(f"Too many unique values ({df[column].nunique()}) in '{column}' for a meaningful pie chart. Displaying top 10.")
        value_counts = df[column].value_counts().nlargest(10).index
        df_filtered = df[df[column].isin(value_counts)]
        fig = px.pie(df_filtered, names=column, title=f'Top 10 Demographic Representation of {column}', template="plotly_white")
    else:
        fig = px.pie(df, names=column, title=f'Demographic Representation of {column}', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def run_page2():
    st.header("Page 2: Data Distribution & Demographic Representation")

    if 'df' not in st.session_state:
        st.warning("Please load a dataset on 'Data Loading & Exploration' page first.")
        return

    df = st.session_state['df']

    # Data Distribution Analysis (Histograms)
    st.markdown("""
    ### Data Distribution Analysis (Histograms)
    Histograms visualize the distribution of numerical features, helping identify patterns and concentrations in the data.
    The formula for a normal distribution's Probability Density Function (PDF) is:
    $$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$
    where $\mu$ is the mean and $\sigma$ is the standard deviation.
    """)

    numerical_cols = df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        selected_num_col_hist = st.sidebar.selectbox(
            "Select numerical feature for histogram:",
            numerical_cols,
            key="hist_num_col"
        )
        bin_size = st.sidebar.slider("Number of bins for histogram:", 5, 100, 20, key="hist_bins")
        plot_histogram(df, selected_num_col_hist, bin_size)
    else:
        st.info("No numerical columns found for histogram analysis.")

    st.divider()

    # Demographic Representation (Pie Charts)
    st.markdown("""
    ### Demographic Representation (Pie Charts)
    Pie charts illustrate the proportional representation of different categories within a demographic feature. They are useful for understanding the composition of categorical data.
    For a category $i$, its proportion $P_i$ is given by:
    $$ P_i = \frac{\text{Count of category } i}{\text{Total observations}} $$
    """)

    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if not categorical_cols.empty:
        selected_cat_col_pie = st.sidebar.selectbox(
            "Select categorical feature for pie chart:",
            categorical_cols,
            key="pie_cat_col"
        )
        plot_pie_chart(df, selected_cat_col_pie)
    else:
        st.info("No categorical columns found for pie chart analysis.")
