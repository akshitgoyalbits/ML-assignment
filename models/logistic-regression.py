"""
Logistic Regression Model for Bankruptcy Prediction
Trains on financial ratios data and saves the model as a pickle file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load data
data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop('Bankrupt?', axis=1)
y = df['Bankrupt?']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
print("Training Logistic Regression model...")
# Use class_weight='balanced' to handle imbalanced data
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Logistic Regression Model Results")
print(f"{'='*50}")
print(f"Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'logistic_regression.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_lr.pkl')

joblib.dump(lr_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
