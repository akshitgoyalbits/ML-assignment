"""
Train all models at once
Run this script to generate all .pkl files for the bankruptcy prediction system
"""

import os
import subprocess
import sys

# List of model scripts to run
model_scripts = [
    'models/logistic-regression.py',
    'models/decision-tree-classifier.py',
    'models/knn-classifier.py',
    'models/gaussian-nb.py',
    'models/random-forest.py',
    'models/ensemble-xg-boost.py'
]

print("="*60)
print("Training All Models for Bankruptcy Prediction")
print("="*60)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Train each model
for i, script in enumerate(model_scripts, 1):
    script_path = os.path.join(script_dir, script)
    print(f"\n[{i}/{len(model_scripts)}] Running: {script}")
    print("-" * 50)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.join(script_dir, 'models'),
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error in {script}:")
            print(result.stderr)
    except Exception as e:
        print(f"Failed to run {script}: {str(e)}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Check which models were created
print("\nGenerated Model Files:")
print("-" * 40)

models_dir = os.path.join(script_dir, 'models')
expected_files = [
    'logistic_regression.pkl',
    'decision_tree.pkl',
    'knn_classifier.pkl',
    'gaussian_nb.pkl',
    'random_forest.pkl',
    'xg_boost.pkl'
]

for file in expected_files:
    file_path = os.path.join(models_dir, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} (not found)")

print("\nGenerated Scaler Files:")
print("-" * 40)

scaler_files = [
    'scaler_lr.pkl',
    'scaler_dt.pkl',
    'scaler_knn.pkl',
    'scaler_gnb.pkl',
    'scaler_rf.pkl',
    'scaler_xgb.pkl'
]

for file in scaler_files:
    file_path = os.path.join(models_dir, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} (not found)")

print("\n" + "="*60)
print("All done! You can now run the Streamlit app:")
print("  streamlit run streamlit_app.py")
print("="*60)
