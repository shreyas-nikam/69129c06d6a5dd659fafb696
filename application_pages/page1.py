
import streamlit as st
import pandas as pd

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(tuple([".xls", ".xlsx"])):
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
    st.markdown("""
    ### Data Loading and Initial Exploration
    This section allows you to load your dataset and get a first look at its structure and basic statistics.
    """).replace("```", "\`\`\`")

    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])

    if "df" not in st.session_state:
        st.session_state.df = None

    if uploaded_file is not None:
        if st.sidebar.button("Load Data"):
            st.session_state.df = load_data(uploaded_file)

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        st.markdown(f"**Number of rows:** {df.shape[0]} | **Number of columns:** {df.shape[1]}")

        st.markdown("#### Basic Statistics (`df.describe()`)")
        st.dataframe(df.describe())

        st.markdown("#### Missing Values per Column")
        missing_values = df.isnull().sum().to_frame(name='Missing Values')
        missing_values['Percentage'] = (missing_values['Missing Values'] / len(df)) * 100
        st.dataframe(missing_values.style.format({"Percentage": "{:.2f}%"}))

        st.markdown("#### Data Types")
        st.dataframe(df.dtypes.astype(str).to_frame(name='Data Type'))

    else:
        st.info("Please upload a dataset using the sidebar to begin.")
