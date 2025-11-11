
import streamlit as st
import pandas as pd
import plotly.express as px

def plot_histogram(df, column, bins=20):
    fig = px.histogram(df, x=column, nbins=bins,
                       title=f'Distribution of {column}',
                       labels={column: column, 'count': 'Frequency'})
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(df, column):
    fig = px.pie(df, names=column, title=f'Demographic Representation of {column}')
    st.plotly_chart(fig, use_container_width=True)

def run_page2():
    st.title("Distributions & Demographics")
    st.markdown("""
    This page focuses on understanding the distribution of your data and the representation of different demographic groups.
    These visualizations are crucial for identifying imbalances or unusual patterns in your dataset.
    """)

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

        # Data Distribution Analysis (Histograms)
        st.markdown("""
        ### Data Distribution Analysis
        Histograms visualize the distribution of numerical features, helping identify patterns and concentrations in the data.
        The formula for a normal distribution's Probability Density Function (PDF) is:
        $$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$
        where $\mu$ is the mean and $\sigma$ is the standard deviation.
        """)

        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            selected_num_col = st.sidebar.selectbox("Select numerical feature for histogram:", numerical_cols, key="hist_col")
            bin_size = st.sidebar.slider("Number of bins for histogram:", 5, 100, 20, key="hist_bins")
            plot_histogram(df, selected_num_col, bin_size)
        else:
            st.warning("No numerical columns found for histogram analysis.")

        st.divider()

        # Demographic Representation (Pie Charts)
        st.markdown("""
        ### Demographic Representation
        Pie charts illustrate the proportional representation of different categories within a demographic feature.
        For a category $i$, its proportion $P_i$ is given by:
        $$ P_i = \frac{\text{Count of category } i}{\text{Total observations}} $$
        """)

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            selected_cat_col = st.sidebar.selectbox("Select categorical feature for pie chart:", categorical_cols, key="pie_col")
            plot_pie_chart(df, selected_cat_col)
        else:
            st.warning("No categorical columns found for pie chart analysis.")

    else:
        st.info("Please load a dataset on the 'Data Loading & Exploration' page to view distributions and demographics.")
