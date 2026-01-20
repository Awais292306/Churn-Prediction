import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('churn.csv')

# --- Data Analysis ---
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# Distribution of Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=df)
plt.title('Churn Distribution (0: No, 1: Yes)')
plt.savefig('churn_distribution.png')
plt.close()

# Correlation Heatmap (only numeric)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# --- Preprocessing ---

# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# One-hot encoding for Geography and Gender
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Define Features and Target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---

# 1. Logistic Regression (Standard for classification)
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# 2. Linear Regression (Linear Probability Model)
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

# --- Evaluation ---
print("\n--- Logistic Regression ---")
y_pred_log = log_model.predict(X_test_scaled)
print("Accuracy Score:", accuracy_score(y_test, y_pred_log))

print("\n--- Linear Regression (LPM) ---")
# For Linear Regression, we round the output to 0 or 1 for classification
y_pred_lin = lin_model.predict(X_test_scaled).clip(0, 1).round()
print("Accuracy Score:", accuracy_score(y_test, y_pred_lin))

# Save the models, scaler, and feature columns
with open('model_logistic.pkl', 'wb') as f:
    pickle.dump(log_model, f)

with open('model_linear.pkl', 'wb') as f:
    pickle.dump(lin_model, f)

# Also save the original model.pkl as logistic for compatibility with old app version if needed
with open('model.pkl', 'wb') as f:
    pickle.dump(log_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nModel, scaler, and columns saved successfully.")
