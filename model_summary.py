# model_summary.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

# Set page configuration for full width
st.set_page_config(layout="wide", page_title="Model Summary")

# ---------------------------
# 1. Load Test Data and Models with Caching
# ---------------------------
@st.cache_data
def load_test_data():
    # Update these paths if needed
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()  # Convert to a Series
    return X_test, y_test

@st.cache_resource
def load_models():
    # Load models and scaler; adjust paths if needed
    log_reg = joblib.load("logistic_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return log_reg, rf, scaler

X_test, y_test = load_test_data()
log_reg, rf, scaler = load_models()

# If needed, scale the test set for models that require scaling (like Logistic Regression)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 2. Make Predictions and Compute Metrics
# ---------------------------
# Predictions for Logistic Regression and Random Forest
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

# Define a function to compute metrics
def compute_metrics(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred)
    }
    return metrics

metrics_log = compute_metrics(y_test, y_pred_log)
metrics_rf = compute_metrics(y_test, y_pred_rf)

# Create a DataFrame for comparison
metrics_df = pd.DataFrame([metrics_log, metrics_rf], index=["Logistic Regression", "Random Forest"])

# ---------------------------
# 3. Streamlit App Layout: Title and Descriptions
# ---------------------------
st.title("Model Summary")
st.write("""
This page presents a summary of the models built for Parkinson's Disease Detection.
It includes:
- Confusion Matrices for each model.
- Feature Importances from both Logistic Regression and Random Forest.
- A bar chart comparing performance metrics (Accuracy, Precision, Recall, F1 Score, and AUC).
""")

# ---------------------------
# 4. Confusion Matrices Visualization
# ---------------------------
st.subheader("Confusion Matrices")

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax,
                xticklabels=["Healthy", "Parkinson's"], yticklabels=["Healthy", "Parkinson's"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    return fig

col_cm1, col_cm2 = st.columns(2)

with col_cm1:
    st.pyplot(plot_confusion_matrix(y_test, y_pred_log, "Logistic Regression"))
with col_cm2:
    st.pyplot(plot_confusion_matrix(y_test, y_pred_rf, "Random Forest"))

# ---------------------------
# 5. Feature Importance Visualization
# ---------------------------
st.subheader("Feature Importances")

# For Logistic Regression, we use the absolute value of coefficients (averaged over classes if multi-class)
logistic_importances = np.abs(log_reg.coef_).flatten()
# For Random Forest, use the model's feature_importances_ attribute
rf_importances = rf.feature_importances_

# Create two side-by-side plots for feature importances
col_fi1, col_fi2 = st.columns(2)

with col_fi1:
    fig_lr, ax_lr = plt.subplots(figsize=(8, 5))
    # Sort indices for a better display
    indices_lr = np.argsort(logistic_importances)
    ax_lr.barh(np.array(X_test.columns)[indices_lr], logistic_importances[indices_lr], color='skyblue')
    ax_lr.set_title("Logistic Regression Feature Importances")
    ax_lr.set_xlabel("Coefficient Magnitude")
    st.pyplot(fig_lr)

with col_fi2:
    fig_rf, ax_rf = plt.subplots(figsize=(8, 5))
    indices_rf = np.argsort(rf_importances)
    ax_rf.barh(np.array(X_test.columns)[indices_rf], rf_importances[indices_rf], color='lightgreen')
    ax_rf.set_title("Random Forest Feature Importances")
    ax_rf.set_xlabel("Importance")
    st.pyplot(fig_rf)

# ---------------------------
# 6. Bar Chart of Model Performance Metrics
# ---------------------------
st.subheader("Model Performance Comparison")

# Plot the metrics DataFrame as a bar chart
fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
metrics_df.plot(kind="bar", ax=ax_metrics, colormap="viridis")
ax_metrics.set_ylabel("Score")
ax_metrics.set_ylim(0, 1)  # Scores are between 0 and 1
ax_metrics.set_title("Performance Metrics: Accuracy, Precision, Recall, F1 Score, and AUC")
ax_metrics.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_metrics)

# ---------------------------
# 7. Display Detailed Classification Reports (Optional)
# ---------------------------
with st.expander("View Detailed Classification Reports"):
    st.subheader("Logistic Regression Report")
    st.text(classification_report(y_test, y_pred_log))
    st.subheader("Random Forest Report")
    st.text(classification_report(y_test, y_pred_rf))
