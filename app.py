"""app.py"""
import streamlit as st
st.set_page_config(page_title="QuLab - Model Card Generator", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we will explore the **Model Card Generator**, a tool designed to help users understand, analyze, and document their machine learning models comprehensively. A model card provides a structured way to report on a model's characteristics, performance, and ethical considerations.

### Why Model Cards?
Model cards enhance transparency and accountability in AI systems. They serve as essential documentation for model developers, deployers, and end-users, ensuring that the models are used responsibly and effectively. Key aspects include:

*   **Transparency**: Clearly documenting how a model was built, what data it was trained on, and its intended use cases.
*   **Accountability**: Providing a record of potential biases, limitations, and performance metrics across different demographic groups.
*   **Risk Mitigation**: Helping identify and address potential fairness, privacy, and security concerns before deployment.

### Lab Structure
This interactive Streamlit application is divided into several pages, each focusing on a different aspect of data analysis and model understanding crucial for generating a comprehensive model card:

*   **Page 1: Data Loading & Exploration**: Load your datasets, preview their structure, and understand basic statistics.
*   **Page 2: Data Distribution & Demographic Representation**: Visualize data distributions using histograms and analyze demographic representation with pie charts.
*   **Page 3: Outlier Detection, Feature Relationships & Bias Insights**: Identify outliers with box plots, explore feature relationships using scatter plots, and delve into conceptual bias detection.

Throughout the lab, we will emphasize interactive visualizations using **Plotly** and clear explanations to foster a deeper understanding of your data and its implications for model development.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Loading & Exploration", "Data Distribution & Demographic Representation", "Outlier Detection, Feature Relationships & Bias Insights"])
if page == "Data Loading & Exploration":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Data Distribution & Demographic Representation":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Outlier Detection, Feature Relationships & Bias Insights":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
