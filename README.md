# Bankruptcy Prediction System

A machine learning project for predicting company bankruptcy using financial ratios. The system includes 6 different classification models and a Streamlit web interface for making predictions.

## 📁 Project Structure

```
Assignment-2/
├── models/
│   ├── data.csv                      # Training dataset
│   ├── logistic-regression.py        # Logistic Regression model training
│   ├── decision-tree-classifier.py   # Decision Tree model training
│   ├── knn-classifier.py             # KNN model training
│   ├── gaussian-nb.py                # Gaussian Naive Bayes model training
│   ├── random-forest.py              # Random Forest model training
│   ├── ensemble-xg-boost.py          # XGBoost model training
│   ├── *.pkl                         # Trained models (generated after training)
│   └── scaler_*.pkl                  # Scaler objects (generated after training)
├── streamlit_app.py                  # Streamlit web application
└── requirements.txt                  # Python dependencies
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

Run each model training script to generate the `.pkl` files:

```bash
# From the Assignment-2 directory
python models/logistic-regression.py
python models/decision-tree-classifier.py
python models/knn-classifier.py
python models/gaussian-nb.py
python models/random-forest.py
python models/ensemble-xg-boost.py
```

After running these scripts, you will have:
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `knn_classifier.pkl`
- `gaussian_nb.pkl`
- `random_forest.pkl`
- `xg_boost.pkl`
- `scaler_*.pkl` (scaler for each model)

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Dataset

The dataset contains 95 financial ratio features:
- ROA (Return on Assets) variations
- Operating margins and profit rates
- Net Value Per Share metrics
- EPS (Earnings Per Share)
- Growth rates
- Asset turnover ratios
- Liability ratios
- And many more financial indicators

**Target Variable:** `Bankrupt?` (0 = Not Bankrupt, 1 = Bankrupt)

## 🎯 How to Use the Streamlit App

1. **Select Models**: Choose one or more models from the sidebar
2. **Upload CSV**: Upload a CSV file with financial features (95 columns)
3. **Make Predictions**: Click the prediction button
4. **View Results**: See predictions for each model and ensemble results

### Input File Format

Your CSV file should have:
- **95 columns** of financial ratios (same as training data)
- **No** 'Bankrupt?' column (this will be predicted)
- Same column names as the training data

### Output

The app provides:
- Individual predictions from each selected model
- Ensemble prediction (majority voting)
- Summary statistics
- Visualizations
- Downloadable CSV with results

## 🤖 Models Included

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear model for binary classification |
| Decision Tree | Tree-based model with max_depth=10 |
| KNN | K-Nearest Neighbors with k=5 |
| Gaussian Naive Bayes | Probabilistic classifier |
| Random Forest | Ensemble of 100 decision trees |
| XGBoost | Gradient boosting classifier |

## 📈 Model Performance

After training, each model will display:
- Accuracy score
- Classification report (precision, recall, f1-score)
- Confusion matrix

## Comparison table with evaluation metrics calculated

| Model                | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|----------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression  | 0.8792   | 0.9489 | 0.1968    | 0.8909 | 0.3224   | 0.3851 |
| Random Forest        | 0.9940   | 0.9914 | 0.9786    | 0.8318 | 0.8993   | 0.8993 |
| XGBoost              | 0.9919   | 0.9897 | 0.8541    | 0.9045 | 0.8786   | 0.8748 |
| KNN Classifier       | 0.9743   | 0.9309 | 0.7922    | 0.2773 | 0.4108   | 0.4596 |
| Gaussian Naive Bayes | 0.2713   | 0.8279 | 0.0413    | 0.9727 | 0.0793   | 0.0912 |
| Decision Tree        | 0.9487   | 0.9408 | 0.3792    | 0.9273 | 0.5383   | 0.5746 |

## Observations for each model

| ML Model Name            | Observation about model performance                                                                     |
|--------------------------|---------------------------------------------------------------------------------------------------------|
| Logistic Regression      | High recall (0.8909) - catches most positives<br>  - Very low precision (0.1968) - many false positives |
| Decision Tree            | Good recall (0.9273) but low precision (0.3792)<br>  - Overfits to positive class                       |
| kNN                      | High accuracy (97.43%) but misleading due to class imbalance                                            |
| Naive Bayes              | Very low accuracy (27.13%) - below random guessing<br>  - Extremely low precision (0.0413)              |
| Random Forest (Ensemble) | Excellent precision (0.9786) - very few false positives Well-balanced with F1 score of 0.8993          |
| XGBoost<br>(Ensemble)    | Best recall (0.9045) among the high-precision models                                                    |
