"""
Gaussian Naive Bayes Classifier for Bankruptcy Prediction
Trains on financial ratios data and saves the model as a pickle file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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

# Train Gaussian NB model
print("Training Gaussian Naive Bayes model...")
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = gnb_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*50)
print("Gaussian Naive Bayes Model Results")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'gaussian_nb.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_gnb.pkl')

joblib.dump(gnb_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
