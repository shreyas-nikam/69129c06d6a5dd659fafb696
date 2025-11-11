"""application_pages/page1.py"""
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None

            st.success("Dataset loaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def run_page1():
    st.header("Page 1: Data Loading & Exploration")
    st.markdown("""
    ### Data Loading and Initial Exploration
    This section allows you to load your dataset and get a first look at its structure and basic statistics.
    Upload your data to begin the analysis. The application supports CSV and Excel files.
    """)

    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df  # Store DataFrame in session state
            st.success("Dataset successfully loaded and stored in session!")

            st.subheader("Dataset Preview (First 5 Rows)")
            st.dataframe(df.head())

            st.subheader("Dataset Information")
            st.markdown("""
            The `df.describe()` method provides a summary of the central tendency, dispersion, and shape of a dataset's distribution, excluding `NaN` values.
            """)
            st.dataframe(df.describe())

            st.subheader("Missing Values per Column")
            st.markdown("""
            Identifying missing values is crucial for data cleaning and preprocessing. High counts of missing values in a column may indicate a need for imputation or feature engineering.
            """)
            missing_values = df.isnull().sum().to_frame(name='Missing Values')
            missing_values = missing_values[missing_values['Missing Values'] > 0]
            if not missing_values.empty:
                st.dataframe(missing_values)
            else:
                st.info("No missing values found in the dataset.")

            st.subheader("Data Types")
            st.markdown("""
            Understanding the data types of each column helps in selecting appropriate visualizations and analysis methods.
            """)
            st.dataframe(df.dtypes.to_frame(name='Data Type'))
    else:
        st.info("Please upload a dataset to proceed.")
