import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# ---------------------------
# Set Page Configuration
# ---------------------------
st.set_page_config(layout = "wide",page_title="Parkinson's Disease Detection App")

# ---------------------------
# Caching Functions for Data and Models
# ---------------------------
@st.cache_data
def load_data():
    """Load the preprocessed dataset from processed_data.csv."""
    return pd.read_csv("processed_data.csv")

@st.cache_resource
def load_models():
    """Load the trained models and scaler."""
    log_reg = joblib.load("logistic_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return log_reg, rf, scaler

# Load the complete dataset and models
df = load_data()
log_reg, rf, scaler = load_models()

# ---------------------------
# Navigation Sidebar
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", options=["Project Summary", "Model Performance Overview", "EDA"])

# ---------------------------
# Perform a Train-Test Split (on the fly using processed_data.csv)
# ---------------------------
X = df.drop(columns=["status"])
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Scale the test set for models that require scaling (e.g., Logistic Regression)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Page: Project Summary
# ---------------------------
if page == "Project Summary":
    st.title("Project Summary")
    st.write("### Parkinson’s Disease Detection using Machine Learning")
    
    st.header("🔍 Motivation")
    st.write("""
    Parkinson’s disease (PD) is a neurodegenerative disorder affecting millions worldwide.
    Early detection is crucial for improving patient outcomes, and machine learning offers
    promising tools for more accurate diagnosis.
    """)
    
    st.header("🎯 Goal")
    st.write("""
    Build an **end-to-end machine learning pipeline** for detecting Parkinson’s disease using biomedical voice data.
    The project involves data preprocessing, EDA, feature engineering, and model training with Logistic Regression
    and Random Forest.
    """)
    
    st.header("⚙️ Methodology")
    st.write("""
    - **Data Preprocessing**: Cleaned and normalized voice measurements.
    - **EDA**: Explored correlations and patterns in the dataset.
    - **Feature Engineering**: Selected key features to enhance model performance.
    - **Model Training**: Implemented Logistic Regression and Random Forest.
    - **Evaluation**: Compared models using metrics like Accuracy, Precision, Recall, F1 Score, and AUC.
    """)
    
    st.header("🚧 Challenges")
    st.write("""
    - **Data Imbalance**: Handling the skewed distribution of PD vs. healthy cases.
    - **Feature Selection**: Identifying the most predictive features.
    - **Generalization**: Ensuring models perform well on unseen data.
    """)
    
    st.header("🔮 Future Work")
    st.write("""
    - Augment the dataset by incorporating more features from the Parkinson’s Progression Markers Initiative (PPMI)
      once access is granted.
    - Explore advanced models, including deep learning approaches.
    - Deploy the solution as a scalable, web-based diagnostic tool.
    """)
    
    st.header("📌 Conclusion")
    st.write("""
    This project showcases the potential of machine learning in medical diagnostics.
    While the current models perform well, further enhancements and larger datasets will help drive
    even better accuracy and reliability in real-world scenarios.
    """)
    
    st.subheader("✨ Thanks for Exploring this Project! 🚀")

# ---------------------------
# Page: Model Performance Overview
# ---------------------------
elif page == "Model Performance Overview":
    st.title("Model Performance Overview")
    st.write("""
    This page presents a summary of the models built for Parkinson's Disease Detection.
    It includes:
    - Confusion Matrices for each model.
    - Feature Importances from both Logistic Regression and Random Forest.
    - A bar chart comparing performance metrics (Accuracy, Precision, Recall, F1 Score, and AUC).
    """)
    
    st.subheader("Confusion Matrices")
    col_cm1, col_cm2 = st.columns(2)
    st.image("/Users/owenfisher/Desktop/Parkinsons Disease Detection/Screenshot 2025-02-05 at 6.49.46 PM.png", caption="Logistic Regression Confusion Matrix")
    st.image("/Users/owenfisher/Desktop/Parkinsons Disease Detection/Screenshot 2025-02-05 at 6.50.02 PM.png", caption="Logistic Regression Feature Importances")

    st.subheader("Performance Metrics Comparison")
    # Replace with the actual file path for the performance metrics chart image
    st.image("/Users/owenfisher/Desktop/Parkinsons Disease Detection/Screenshot 2025-02-05 at 6.50.51 PM.png", caption="Performance Metrics Comparison")
    
    st.subheader("Detailed Classification Reports")
    st.write("Please refer to the saved report files for detailed classification reports.")

# ---------------------------
# Page: EDA
# ---------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("""
    Use this page to explore the dataset.  
    Select two features from the sidebar and choose a plot type to visualize their relationship.
    """)
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Correlation Heatmap")
    st.image("correlation_heatmap.png", caption="Correlation Heatmap", use_container_width=True)
    
    # EDA Sidebar Options (using additional keys to avoid conflicts)
    st.sidebar.subheader("EDA Settings")
    eda_feature_options = list(df.columns)
    if "status" in eda_feature_options:
        eda_feature_options.remove("status")
    
    eda_feature1 = st.sidebar.selectbox("Select Feature 1 for EDA", eda_feature_options, index=0, key="eda1")
    eda_feature2 = st.sidebar.selectbox("Select Feature 2 for EDA", eda_feature_options, index=1, key="eda2")
    eda_plot_type = st.sidebar.radio("Choose EDA Plot Type", ["Scatter", "Box", "Violin"], key="eda_plot")
    
    st.subheader(f"{eda_plot_type} Plot")
    fig_eda, ax_eda = plt.subplots(figsize=(10, 6))
    if eda_plot_type == "Scatter":
        sns.scatterplot(data=df, x=eda_feature1, y=eda_feature2, hue=df['status'], palette="coolwarm", ax=ax_eda)
        ax_eda.set_title(f"Scatter Plot: {eda_feature1} vs {eda_feature2}")
    elif eda_plot_type == "Box":
        sns.boxplot(data=df, x="status", y=eda_feature1, palette="coolwarm", ax=ax_eda)
        ax_eda.set_title(f"Box Plot: {eda_feature1} by Status")
    elif eda_plot_type == "Violin":
        sns.violinplot(data=df, x="status", y=eda_feature1, palette="coolwarm", ax=ax_eda)
        ax_eda.set_title(f"Violin Plot: {eda_feature1} by Status")
    st.pyplot(fig_eda)
