id: 69129c06d6a5dd659fafb696_documentation
summary: Model Card Generator Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building a Model Card Generator: Data Understanding with Streamlit

## 1. Introduction to Model Cards and Application Overview
Duration: 0:10:00

In the rapidly evolving landscape of Artificial Intelligence, the need for transparency, accountability, and ethical considerations has become paramount. This codelab explores the **Model Card Generator**, a Streamlit application designed to facilitate comprehensive data understanding, a critical first step towards creating robust and responsible machine learning model cards.

### Why Model Cards are Essential
Model cards serve as structured documentation for machine learning models, much like nutritional labels for food products. They provide vital information for model developers, deployers, and end-users, ensuring that models are used responsibly and effectively. Key benefits include:

*   **Transparency**: Clearly documenting how a model was built, the data it was trained on, its intended use cases, and known limitations.
*   **Accountability**: Providing a record of potential biases, performance metrics across different demographic groups, and ethical considerations.
*   **Risk Mitigation**: Helping identify and address potential fairness, privacy, and security concerns before model deployment, thereby reducing risks associated with unintended consequences.

This application focuses specifically on the **data understanding** aspect of model card generation. Before a model can be built and documented, it is crucial to thoroughly explore and analyze the underlying dataset. This involves:

*   **Initial Data Exploration**: Gaining a first look at the dataset's structure, summary statistics, and identifying missing values.
*   **Data Distribution Analysis**: Visualizing how numerical features are distributed, revealing skewness, outliers, or multimodal patterns.
*   **Demographic Representation**: Examining the proportions of different categories within demographic features to identify potential imbalances that could lead to biased model outcomes.
*   **Outlier Detection**: Using statistical plots to identify unusual data points that might impact model training.
*   **Feature Relationships**: Understanding how different features interact with each other, crucial for feature engineering and model interpretation.
*   **Bias Detection and Summary Insights**: A dedicated section to interpret findings and discuss potential biases, their impact on model performance, and mitigation strategies.

All visualizations in this application are interactive and generated using **Plotly**, allowing for dynamic exploration of the data. The application also extensively uses Streamlit's `st.session_state` to maintain data persistence, ensuring a seamless user experience as you navigate through different analysis steps.

### Application Architecture
The application is structured modularly, enhancing readability and maintainability.

```mermaid
graph TD
    A[User Browser] -->|Interacts with| B(Streamlit Application)
    B -->|Sidebar Navigation| C{app.py}
    C -->|Calls function based on selection| D[Page 1: Data Loading & Exploration]
    C -->|Calls function based on selection| E[Page 2: Distributions & Demographics]
    C -->|Calls function based on selection| F[Page 3: Outliers & Relationships]
    D --|>|G[st.session_state['df']]
    E --|>|G
    F --|>|G
    G --|>|D
    G --|>|E
    G --|>|F
    D -->|Data Previews, Statistics| H[Pandas DataFrames]
    E -->|Histograms, Pie Charts| I[Plotly Visualizations]
    F -->|Box Plots, Scatter Plots, Bias Insights| I
```
<br>

### Setting up and Running the Application

To run this application, you need to have Python and `pip` installed.

1.  **Clone the Repository (or create files):**
    First, ensure you have the application code structured as follows:

    ```
    .
    ├── app.py
    └── application_pages/
        ├── __init__.py
        ├── page1.py
        ├── page2.py
        └── page3.py
    ```

    You can create these files manually and copy the provided code into them.

2.  **Install Dependencies:**
    Open your terminal or command prompt and navigate to the root directory of your application (where `app.py` is located). Then install the required Python libraries:

    ```bash
    pip install streamlit pandas plotly openpyxl
    ```
    *   `streamlit`: For building the interactive web application.
    *   `pandas`: For data manipulation and analysis.
    *   `plotly`: For creating interactive visualizations.
    *   `openpyxl`: Required by pandas to read `.xlsx` files.

3.  **Run the Streamlit Application:**
    In the same terminal, run the application using the Streamlit command:

    ```bash
    streamlit run app.py
    ```

    This command will open a new tab in your web browser, displaying the Streamlit application.

<aside class="positive">
Always create a virtual environment (`python -m venv venv` then `source venv/bin/activate` on Linux/macOS or `.\venv\Scripts\activate` on Windows) to manage project dependencies. This prevents conflicts with other Python projects.
</aside>

## 2. Data Loading & Initial Exploration
Duration: 0:15:00

This step covers the functionalities of `application_pages/page1.py`, which is responsible for allowing users to upload their datasets and perform initial exploratory data analysis.

### `application_pages/page1.py` - Code Overview

```python
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
    """)

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
```

### Key Functionalities

1.  **File Upload (`st.file_uploader`)**:
    The application provides a file uploader in the sidebar, allowing users to upload their datasets. It supports `CSV` and `Excel` (`.xlsx`, `.xls`) file formats.

    ```python
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV, Excel)", type=["csv", "xlsx"])
    ```

2.  **Data Loading (`load_data`)**:
    The `load_data` function, decorated with `@st.cache_data`, efficiently reads the uploaded file into a Pandas DataFrame. The `@st.cache_data` decorator ensures that the function's output is cached, preventing re-execution on subsequent runs with the same input, which is crucial for performance.

    ```python
    @st.cache_data
    def load_data(uploaded_file):
        # ... file type detection and reading ...
    ```

3.  **Session State for Data Persistence (`st.session_state`)**:
    Once loaded, the DataFrame is stored in `st.session_state['df']`. This is a powerful feature in Streamlit that allows data to persist across different pages and user interactions without reloading, providing a smooth user experience.

    ```python
    if uploaded_file is not None:
        if st.sidebar.button("Load Data"):
            st.session_state.df = load_data(uploaded_file)
    ```

4.  **Dataset Preview (`df.head()`)**:
    Displays the first few rows of the loaded dataset, offering a quick glimpse into its structure and content.

    ```python
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    ```

5.  **Dataset Information (`df.shape`)**:
    Provides the number of rows and columns, giving a basic understanding of the dataset's size.

    ```python
    st.markdown(f"**Number of rows:** {df.shape[0]} | **Number of columns:** {df.shape[1]}")
    ```

6.  **Basic Statistics (`df.describe()`)**:
    Generates descriptive statistics for numerical columns, including count, mean, standard deviation, minimum, quartiles, and maximum values. This is essential for understanding the central tendency, dispersion, and shape of the data.

    ```python
    st.markdown("#### Basic Statistics (`df.describe()`)")
    st.dataframe(df.describe())
    ```

7.  **Missing Values Analysis (`df.isnull().sum()`)**:
    Calculates the number and percentage of missing values per column. Identifying missing data is a critical step in data quality assessment, often leading to imputation or feature engineering decisions.

    ```python
    st.markdown("#### Missing Values per Column")
    missing_values = df.isnull().sum().to_frame(name='Missing Values')
    missing_values['Percentage'] = (missing_values['Missing Values'] / len(df)) * 100
    st.dataframe(missing_values.style.format({"Percentage": "{:.2f}%"}))
    ```

8.  **Data Types (`df.dtypes`)**:
    Displays the data type of each column. Understanding data types is crucial for selecting appropriate analysis methods and visualizations.

    ```python
    st.markdown("#### Data Types")
    st.dataframe(df.dtypes.astype(str).to_frame(name='Data Type'))
    ```

<aside class="positive">
Initial data exploration, including checking for missing values and understanding data types, is a foundational step. It directly influences subsequent data preprocessing, feature engineering, and model selection. Thorough exploration here can prevent downstream issues and enhance model reliability.
</aside>

## 3. Data Distribution & Demographic Representation
Duration: 0:20:00

This step delves into `application_pages/page2.py`, focusing on visualizing the distribution of numerical features and the representation of different categories within demographic features. These insights are vital for understanding the data's composition and identifying potential imbalances.

### `application_pages/page2.py` - Code Overview

```python
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
```

### Key Functionalities

1.  **Data Persistence Check**:
    The page first checks if a DataFrame (`df`) is available in `st.session_state`. If not, it prompts the user to load data on the first page, ensuring a consistent workflow.

    ```python
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        # ... proceed with analysis ...
    else:
        st.info("Please load a dataset on the 'Data Loading & Exploration' page to view distributions and demographics.")
    ```

2.  **Data Distribution Analysis (Histograms)**:
    *   **Purpose**: Histograms visually represent the frequency distribution of numerical data. They help identify the shape of the distribution (e.g., normal, skewed, multimodal), central tendency, and spread.
    *   **Implementation (`plot_histogram`)**: Uses `plotly.express.histogram` to create interactive histograms. The user can select a numerical column and adjust the number of bins via `st.sidebar.selectbox` and `st.sidebar.slider`.
    *   **Mathematical Concept**: The Probability Density Function (PDF) for a normal (Gaussian) distribution is presented as a theoretical context:
        $$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$
        where $\mu$ is the mean and $\sigma$ is the standard deviation.

    ```python
    def plot_histogram(df, column, bins=20):
        fig = px.histogram(df, x=column, nbins=bins,
                           title=f'Distribution of {column}',
                           labels={column: column, 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)

    # ... in run_page2 ...
    numerical_cols = df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        selected_num_col = st.sidebar.selectbox("Select numerical feature for histogram:", numerical_cols, key="hist_col")
        bin_size = st.sidebar.slider("Number of bins for histogram:", 5, 100, 20, key="hist_bins")
        plot_histogram(df, selected_num_col, bin_size)
    ```

3.  **Demographic Representation (Pie Charts)**:
    *   **Purpose**: Pie charts are used to visualize the proportional composition of categorical features, especially useful for understanding demographic representation. They quickly show whether certain categories are over- or under-represented in the dataset.
    *   **Implementation (`plot_pie_chart`)**: Uses `plotly.express.pie` to generate interactive pie charts. Users can select a categorical column from `st.sidebar.selectbox`.
    *   **Mathematical Concept**: The proportion $P_i$ for a given category $i$ is calculated as:
        $$ P_i = \frac{\text{Count of category } i}{\text{Total observations}} $$

    ```python
    def plot_pie_chart(df, column):
        fig = px.pie(df, names=column, title=f'Demographic Representation of {column}')
        st.plotly_chart(fig, use_container_width=True)

    # ... in run_page2 ...
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        selected_cat_col = st.sidebar.selectbox("Select categorical feature for pie chart:", categorical_cols, key="pie_col")
        plot_pie_chart(df, selected_cat_col)
    ```

<aside class="negative">
Be cautious when interpreting pie charts with too many categories, as they can become cluttered and hard to read. For features with a high cardinality, consider displaying only the top N categories or using a bar chart instead.
</aside>

## 4. Outlier Detection, Feature Relationships & Bias Insights
Duration: 0:25:00

This step details the functionalities within `application_pages/page3.py`, covering advanced data visualization techniques for outlier detection, exploring relationships between features, and a crucial section on understanding and detecting conceptual biases in the dataset.

### `application_pages/page3.py` - Code Overview

```python
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
            *   [Download Awesome-Fairness-in-AI](https://github.com/EthicalML/awesome-fairness-in-ai)
            *   [Download Google AI's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
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
```

### Key Functionalities

1.  **Outlier Detection (Box Plots)**:
    *   **Purpose**: Box plots effectively visualize the distribution of numerical data and are excellent for identifying potential outliers. They display the five-number summary: minimum, first quartile ($Q_1$), median ($Q_2$), third quartile ($Q_3$), and maximum.
    *   **Implementation (`plot_box_plot`)**: Uses `plotly.express.box`. Users can select a numerical column and optionally group the box plot by a categorical feature using `st.sidebar.selectbox`.
    *   **Mathematical Concept**: Outliers are statistically defined using the Interquartile Range (IQR):
        *   $IQR = Q_3 - Q_1$
        *   Outliers are typically defined as data points below $Q_1 - 1.5 \times IQR$ or above $Q_3 + 1.5 \times IQR$.

    ```python
    def plot_box_plot(df, column, group_by=None):
        # ... plotly box plot creation ...
        st.plotly_chart(fig, use_container_width=True)

    # ... in run_page3 ...
    selected_num_col_box = st.sidebar.selectbox("Select numerical feature for box plot:", numerical_cols, key="box_num")
    group_by_col = st.sidebar.selectbox("Group box plot by (optional):", ['None'] + list(categorical_cols), key="box_group")
    plot_box_plot(df, selected_num_col_box, group_by=None if group_by_col == 'None' else group_by_col)
    ```

2.  **Feature Relationships (Scatter Plots)**:
    *   **Purpose**: Scatter plots visualize the relationship between two numerical features, helping to identify correlations, trends, or clusters. They are powerful for exploring potential dependencies between variables.
    *   **Implementation (`plot_scatter_plot`)**: Uses `plotly.express.scatter`. Users select two numerical columns for the X and Y axes and can optionally color the points by a categorical feature.
    *   **Mathematical Concept**: The linear correlation coefficient $r$ between variables $X$ and $Y$ is a measure of the strength and direction of a linear relationship:
        $$ r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}} $$
        where $n$ is the number of observations, $x_i, y_i$ are individual data points, and $\bar{x}, \bar{y}$ are the means.

    ```python
    def plot_scatter_plot(df, x_column, y_column, color_by=None):
        # ... plotly scatter plot creation ...
        st.plotly_chart(fig, use_container_width=True)

    # ... in run_page3 ...
    x_col = st.sidebar.selectbox("Select X-axis feature for scatter plot:", numerical_cols, key="scatter_x")
    y_col = st.sidebar.selectbox("Select Y-axis feature for scatter plot:", numerical_cols, key="scatter_y")
    color_by_col = st.sidebar.selectbox("Color scatter plot by (optional):", ['None'] + list(categorical_cols), key="scatter_color")
    plot_scatter_plot(df, x_col, y_col, color_by=None if color_by_col == 'None' else color_by_col)
    ```

3.  **Bias Detection and Summary Insights**:
    *   **Purpose**: This crucial section highlights the importance of identifying and understanding potential biases in the dataset, particularly concerning demographic attributes. It emphasizes that data bias can lead to unfair or inaccurate model outcomes.
    *   **Definition of Bias**: Explained as systemic errors in data collection or intrinsic properties of data leading to unfair outcomes.
    *   **Impact on Model Performance**: Discusses how biased data can lead to models that perform poorly for underrepresented groups or perpetuate stereotypes.
    *   **Mitigation Strategies**: Briefly outlines techniques like re-sampling, re-weighting, and adversarial debiasing.
    *   **Demographic Balance Check (`check_demographic_balance`)**: A simple function to display the proportional distribution of selected demographic columns. This helps users quickly see if certain groups are under- or over-represented.

    ```python
    def check_demographic_balance(df, demographic_column):
        # ... calculates and displays proportions ...
        st.info("Compare these proportions to real-world demographics to identify under/over-representation.")

    # ... in run_page3 ...
    demographic_cols_for_check = [col for col in categorical_cols if col.lower() in ['gender', 'race', 'ethnicity', 'age_group', 'country', 'region']]
    if demographic_cols_for_check:
        selected_demographic_for_bias = st.sidebar.selectbox("Select demographic for bias check:", demographic_cols_for_check, key="bias_check_col")
        check_demographic_balance(df, selected_demographic_for_bias)
    ```
    *   **Further Resources**: Provides links to valuable external resources for fairness in AI and responsible AI practices using download buttons.

<button>
  [Download Awesome-Fairness-in-AI](https://github.com/EthicalML/awesome-fairness-in-ai)
</button>
<button>
  [Download Google AI's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
</button>

<aside class="positive">
Understanding and addressing biases at the data analysis stage is critical for building ethical AI. This proactive approach helps in designing fairer data collection strategies or applying debiasing techniques before model training.
</aside>

## 5. Understanding the Streamlit Application Structure
Duration: 0:10:00

This step provides a deeper look into the overall structure of the Streamlit application, specifically how `app.py` orchestrates the different analytical pages.

### `app.py` - Main Application Entry Point

```python
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we present a "Model Card Generator" application, an interactive tool designed to help users understand, visualize, and analyze datasets for potential biases and demographic representation. Model cards provide a structured way to document the intended uses, performance characteristics, and ethical considerations of machine learning models.

This application focuses on the **data understanding** aspect of model card generation. Before a model can be built and documented, it is crucial to thoroughly understand the underlying dataset. This involves:

*   **Initial Data Exploration**: Getting a first look at the dataset's structure, summary statistics, and identifying missing values.
*   **Data Distribution Analysis**: Visualizing how numerical features are distributed, which can reveal skewness, outliers, or multimodal patterns.
*   **Demographic Representation**: Examining the proportions of different categories within demographic features to identify potential imbalances that could lead to biased model outcomes.
*   **Outlier Detection**: Using statistical plots to identify unusual data points that might impact model training.
*   **Feature Relationships**: Understanding how different features interact with each other, which is crucial for feature engineering and model interpretation.
*   **Bias Detection and Summary Insights**: A dedicated section to interpret findings and discuss potential biases, their impact on model performance, and mitigation strategies.

Understanding these aspects of your data is a foundational step in building responsible and fair AI systems. By providing an interactive platform, we aim to make this process intuitive and insightful for users.

All visualizations in this application are interactive and generated using **Plotly**, allowing for dynamic exploration of the data. We also extensively use `st.session_state` to maintain the application's state, ensuring a seamless user experience as you navigate through different analysis steps.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Loading & Exploration", "Distributions & Demographics", "Outliers & Relationships"])
if page == "Data Loading & Exploration":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Distributions & Demographics":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Outliers & Relationships":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
```

### Key Aspects of `app.py`

1.  **Page Configuration (`st.set_page_config`)**:
    Sets the overall configuration for the Streamlit page, including the title displayed in the browser tab and the layout (e.g., `wide` for more screen real estate).

    ```python
    st.set_page_config(page_title="QuLab", layout="wide")
    ```

2.  **Sidebar Elements**:
    The sidebar is used for branding (logo, title) and, most importantly, for navigation.
    *   `st.sidebar.image()`: Displays a logo.
    *   `st.sidebar.divider()`: Adds a visual separator.
    *   `st.title()`: Sets the main title of the application.

    ```python
    st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
    st.sidebar.divider()
    st.title("QuLab")
    ```

3.  **Navigation (`st.sidebar.selectbox`)**:
    This is the core of the multi-page structure. A `selectbox` in the sidebar allows users to select which analysis page they want to view. Based on the selection, `app.py` dynamically imports and calls the corresponding `run_page` function from the `application_pages` module.

    ```python
    page = st.sidebar.selectbox(label="Navigation", options=["Data Loading & Exploration", "Distributions & Demographics", "Outliers & Relationships"])
    if page == "Data Loading & Exploration":
        from application_pages.page1 import run_page1
        run_page1()
    elif page == "Distributions & Demographics":
        from application_pages.page2 import run_page2
        run_page2()
    elif page == "Outliers & Relationships":
        from application_pages.page3 import run_page3
        run_page3()
    ```
    This conditional rendering ensures that only the selected page's content is displayed, while `st.session_state` preserves the loaded data across these page switches.

4.  **Modular Design**:
    By separating the functionalities into `page1.py`, `page2.py`, and `page3.py` files within the `application_pages` directory, the application achieves a modular and organized codebase. Each page focuses on a specific set of analyses, making the code easier to understand, maintain, and extend. The `__init__.py` file (even if empty) makes `application_pages` a Python package, allowing for imports like `from application_pages.page1 import run_page1`.

<aside class="positive">
Using `st.session_state` for data persistence is crucial in multi-page Streamlit apps. It avoids reloading data every time the page changes, significantly improving user experience and application responsiveness.
</aside>

## 6. Conclusion and Next Steps
Duration: 0:05:00

### Summary of Learning
Through this codelab, you have gained a comprehensive understanding of how to build an interactive Streamlit application for **data understanding**, specifically tailored as a precursor to generating **model cards**. You've learned to:

*   Structure a multi-page Streamlit application using `st.sidebar.selectbox` and modular Python files.
*   Leverage `st.cache_data` for efficient data loading and `st.session_state` for seamless data persistence across pages.
*   Implement various exploratory data analysis (EDA) techniques, including:
    *   Displaying dataset previews, basic statistics, missing values, and data types.
    *   Visualizing data distributions with interactive Plotly histograms.
    *   Analyzing demographic representation using interactive Plotly pie charts.
    *   Detecting outliers with interactive Plotly box plots.
    *   Exploring feature relationships with interactive Plotly scatter plots.
*   Discuss the critical importance of **bias detection** in datasets and its implications for responsible AI.

Understanding your data in such depth is a foundational pillar for developing fair, transparent, and robust machine learning models. The insights gained from this application are directly transferable to populating sections of a comprehensive model card, ensuring that model characteristics and ethical considerations are well-documented.

### Next Steps

This application lays the groundwork for data understanding. Here are some potential next steps to expand its capabilities and move towards a full model card generator:

1.  **Automated Feature Engineering**: Integrate tools for feature scaling, encoding categorical variables, or creating new features based on the exploratory analysis.
2.  **Model Training and Evaluation**: Add a new page for users to select a target variable, train simple machine learning models (e.g., Logistic Regression, Decision Tree), and evaluate their performance.
3.  **Advanced Bias Mitigation**: Implement and demonstrate specific debiasing techniques (e.g., re-sampling, re-weighting, disparate impact remover) and show their effect on demographic fairness.
4.  **Full Model Card Generation**: Create a dedicated section to compile all the gathered information (data characteristics, model performance, fairness metrics) into a structured model card document, perhaps allowing export in formats like Markdown or PDF.
5.  **Explainability (XAI)**: Incorporate tools like SHAP or LIME to explain model predictions, adding another layer of transparency.
6.  **Data Quality Checks**: Extend missing value analysis to suggest imputation strategies or anomaly detection algorithms beyond simple box plots.

By continuously enhancing applications like this, we move closer to a future where AI development is synonymous with responsibility and transparency.
