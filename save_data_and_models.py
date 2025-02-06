# save_data_and_models.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
# Update the path to your original dataset if needed.
df = pd.read_csv('/Users/owenfisher/Desktop/Parkinsons Disease Detection/data/parkinsons.data')

# Drop the 'name' column as it's not a predictive feature
if 'name' in df.columns:
    df.drop(columns='name', inplace=True)

# Save the preprocessed dataset for quick access later
df.to_csv("processed_data.csv", index=False)
print("Processed dataset saved as processed_data.csv")

# ---------------------------
# 2. Visualizations and Correlation Heatmap
# ---------------------------
# Plot histograms for all numerical features
df.hist(figsize=(25, 20))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()
print("Histograms saved as histograms.png")

# Compute and plot the correlation matrix (including all features)
plt.figure(figsize=(20, 20))
correl = df.drop(columns='status').corr()
sns.heatmap(correl, annot=True, cmap='OrRd')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("Correlation heatmap saved as correlation_heatmap.png")

# ---------------------------
# 3. Build Models
# ---------------------------
# Define features and target
X = df.drop(columns='status')
y = df['status']

# ---------------------------
# 4. Data Splitting and Scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    stratify=y)
print("Train and test sets created.")

# Scale data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")

# ---------------------------
# 5. Train Models
# ---------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate (optional)
print("Logistic Regression Test Accuracy:", log_reg.score(X_test_scaled, y_test))
print("Random Forest Test Accuracy:", rf.score(X_test, y_test))

# ---------------------------
# 6. Save Trained Models
# ---------------------------
joblib.dump(log_reg, "logistic_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
print("Models saved as logistic_model.pkl and random_forest_model.pkl")
