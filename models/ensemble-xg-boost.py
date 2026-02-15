"""
XGBoost Classifier for Bankruptcy Prediction
Trains on financial ratios data and saves the model as a pickle file.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
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

# Calculate class weights for imbalanced data
from collections import Counter
counter = Counter(y_train)
scale_pos_weight = counter[0] / counter[1] if counter[1] > 0 else 1

# Train XGBoost model
print("Training XGBoost model...")
print(f"Class distribution - Class 0: {counter[0]}, Class 1: {counter[1]}")
print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight  # Handle class imbalance
)
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*50)
print("XGBoost Model Results")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'xg_boost.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_xgb.pkl')

joblib.dump(xgb_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
