"""
Streamlit App for Bankruptcy Prediction
Allows users to upload a CSV file and get predictions from multiple ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏢 Bankruptcy Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("⚙️ Settings")

# Available models
models_info = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN Classifier": "knn_classifier.pkl",
    "Gaussian Naive Bayes": "gaussian_nb.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xg_boost.pkl"
}

scalers_info = {
    "Logistic Regression": "scaler_lr.pkl",
    "Decision Tree": "scaler_dt.pkl",
    "KNN Classifier": "scaler_knn.pkl",
    "Gaussian Naive Bayes": "scaler_gnb.pkl",
    "Random Forest": "scaler_rf.pkl",
    "XGBoost": "scaler_xgb.pkl"
}

# Model selection
selected_models = st.sidebar.multiselect(
    "Select Models to Use",
    options=list(models_info.keys()),
    default=["Logistic Regression", "Random Forest", "XGBoost"]
)

# File upload section
st.header("📁 Upload Your Data")
st.markdown('<div class="info-box">Upload a CSV file with financial ratios data. The file should have the same features as the training data (95 financial ratio columns).</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with financial features (without Bankrupt? column)"
)

# Information about expected features
with st.expander("ℹ️ Expected Features"):
    st.write("""
    The model expects 95 financial ratio features including:
    - ROA(C) before interest and depreciation before interest
    - Operating Gross Margin
    - Research and development expense rate
    - Net Value Per Share (A/B/C)
    - Persistent EPS in the Last Four Seasons
    - ... and 90 more financial ratios

    **Note:** Do not include the 'Bankrupt?' column in your test file.
    """)

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

        # Check if Bankrupt column exists
        if 'Bankrupt?' in df.columns:
            # Clear old session state
            if 'actual_values' in st.session_state:
                del st.session_state['actual_values']

            actual_vals = df['Bankrupt?'].values
            st.session_state['actual_values'] = actual_vals

            # Show class distribution
            unique, counts = np.unique(actual_vals, return_counts=True)
            st.info(f"📊 Actual values distribution: Class 0 (Not Bankrupt): {counts[0] if len(counts) > 0 and 0 in unique else 0}, Class 1 (Bankrupt): {counts[1] if len(counts) > 1 and 1 in unique else 0}")

            st.warning("⚠️ Your file contains 'Bankrupt?' column. It will be removed for prediction but used for evaluation metrics.")
            df = df.drop('Bankrupt?', axis=1)

        # Validate number of features
        expected_features = 95
        if df.shape[1] != expected_features:
            st.error(f"❌ Expected {expected_features} features, but found {df.shape[1]}. Please check your data.")
        else:
            st.success(f"✅ Data looks good! {df.shape[1]} features detected.")

            # Predict button
            if st.button("🔮 Make Predictions", type="primary"):
                if not selected_models:
                    st.error("Please select at least one model.")
                else:
                    # Create a results dataframe
                    results_df = pd.DataFrame()
                    results_df['Row_Index'] = range(len(df))

                    # Create columns for predictions
                    predictions = {}
                    prediction_probas = {}  # Store probabilities for AUC
                    progress_bar = st.progress(0)

                    for i, model_name in enumerate(selected_models):
                        # Load model and scaler
                        models_dir = "models"
                        model_path = os.path.join(models_dir, models_info[model_name])
                        scaler_path = os.path.join(models_dir, scalers_info[model_name])

                        try:
                            model = joblib.load(model_path)
                            scaler = joblib.load(scaler_path)

                            # Scale the data
                            X_scaled = scaler.transform(df)

                            # Make predictions
                            pred = model.predict(X_scaled)
                            predictions[model_name] = pred

                            # Get prediction probabilities if available (for AUC)
                            try:
                                proba = model.predict_proba(X_scaled)
                                # For binary classification, get probability of class 1
                                if proba.shape[1] == 2:
                                    prediction_probas[model_name] = proba[:, 1]
                                else:
                                    prediction_probas[model_name] = None
                            except:
                                prediction_probas[model_name] = None

                            progress_bar.progress((i + 1) / len(selected_models))

                        except Exception as e:
                            st.error(f"Error loading {model_name}: {str(e)}")

                    progress_bar.empty()

                    # Add predictions to results dataframe
                    for model_name, pred in predictions.items():
                        results_df[model_name] = pred

                    # Calculate ensemble prediction (majority voting)
                    if len(predictions) > 1:
                        pred_matrix = np.column_stack(list(predictions.values()))
                        ensemble_pred = []
                        for row in pred_matrix:
                            ensemble_pred.append(int(np.bincount(row).argmax()))
                        results_df['Ensemble (Majority Vote)'] = ensemble_pred

                    # Display results
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(results_df)

                    # Summary statistics
                    st.subheader("📈 Summary Statistics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Records", len(df))

                    with col2:
                        # Count bankrupt predictions across all models
                        total_bankrupt = sum([sum(pred) for pred in predictions.values()])
                        avg_bankrupt = total_bankrupt / (len(predictions) * len(df))
                        st.metric("Avg Bankruptcy Rate", f"{avg_bankrupt:.1%}")

                    with col3:
                        st.metric("Models Used", len(selected_models))

                    # Individual model statistics
                    st.subheader("📊 Model-wise Statistics")

                    model_stats = []
                    for model_name, pred in predictions.items():
                        bankrupt_count = sum(pred)
                        non_bankrupt = len(pred) - bankrupt_count
                        model_stats.append({
                            'Model': model_name,
                            'Bankrupt (1)': bankrupt_count,
                            'Non-Bankrupt (0)': non_bankrupt,
                            'Bankruptcy %': f"{(bankrupt_count/len(pred))*100:.2f}%"
                        })

                    # Add ensemble stats if available
                    if 'Ensemble (Majority Vote)' in results_df.columns:
                        ensemble_pred = results_df['Ensemble (Majority Vote)'].values
                        bankrupt_count = sum(ensemble_pred)
                        model_stats.append({
                            'Model': 'Ensemble (Majority Vote)',
                            'Bankrupt (1)': bankrupt_count,
                            'Non-Bankrupt (0)': len(ensemble_pred) - bankrupt_count,
                            'Bankruptcy %': f"{(bankrupt_count/len(ensemble_pred))*100:.2f}%"
                        })

                    stats_df = pd.DataFrame(model_stats)
                    st.dataframe(stats_df, use_container_width=True)

                    # Visualization
                    st.subheader("📉 Visualization")

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    # Plot 1: Bankruptcy counts by model
                    ax1 = axes[0]
                    plot_data = stats_df.melt(
                        id_vars=['Model'],
                        value_vars=['Bankrupt (1)', 'Non-Bankrupt (0)'],
                        var_name='Category',
                        value_name='Count'
                    )
                    sns.barplot(data=plot_data, x='Model', y='Count', hue='Category', ax=ax1)
                    ax1.set_title('Bankruptcy Predictions by Model')
                    ax1.set_xlabel('Model')
                    ax1.set_ylabel('Count')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.legend(title='Prediction')

                    # Plot 2: Bankruptcy percentage
                    ax2 = axes[1]
                    bankruptcy_pct = [float(s['Bankruptcy %'].rstrip('%')) for s in model_stats]
                    model_names = [s['Model'] for s in model_stats]
                    colors = ['#ff6b6b' if pct > 5 else '#51cf66' for pct in bankruptcy_pct]
                    ax2.barh(model_names, bankruptcy_pct, color=colors)
                    ax2.set_title('Bankruptcy Percentage by Model')
                    ax2.set_xlabel('Percentage (%)')
                    ax2.set_xlim(0, max(bankruptcy_pct) * 1.2)

                    # Add value labels
                    for i, v in enumerate(bankruptcy_pct):
                        ax2.text(v + 0.1, i, f'{v:.1f}%', va='center')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Download results
                    st.subheader("💾 Download Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="bankruptcy_predictions.csv",
                        mime="text/csv"
                    )

                    # Compare with actual values if available - Show all evaluation metrics
                    if 'actual_values' in st.session_state:
                        st.subheader("📊 Comprehensive Evaluation Metrics")

                        actual = st.session_state['actual_values']

                        # Show class imbalance warning
                        class_0_count = sum(actual == 0)
                        class_1_count = sum(actual == 1)
                        imbalance_ratio = class_0_count / max(class_1_count, 1)

                        if imbalance_ratio > 10:
                            st.warning(f"⚠️ **Highly Imbalanced Data Detected**: Class ratio is {imbalance_ratio:.1f}:1 ({class_0_count} vs {class_1_count}). "
                                     "This can lead to low MCC even with good accuracy. Models should be retrained with class_weight='balanced'.")

                        # Show debugging info about predictions
                        st.info("🔍 Debugging Info - Check predictions vs actual values")
                        debug_cols = st.columns(len(predictions) + 1)
                        debug_cols[0].metric("Actual - Class 0", class_0_count)
                        debug_cols[0].metric("Actual - Class 1", class_1_count)

                        for idx, (model_name, pred) in enumerate(predictions.items(), 1):
                            pred_0 = sum(pred == 0)
                            pred_1 = sum(pred == 1)
                            debug_cols[idx].metric(f"{model_name}\nPredicted 0", pred_0)
                            debug_cols[idx].metric(f"{model_name}\nPredicted 1", pred_1)

                            # Warn if model predicts only one class
                            if pred_1 == 0:
                                st.warning(f"⚠️ {model_name} predicted ALL samples as Class 0. This indicates the model needs retraining with better class balancing.")

                        # Calculate all metrics for each model
                        metrics_data = []

                        for model_name, pred in predictions.items():
                            # Calculate metrics
                            acc = accuracy_score(actual, pred)
                            precision = precision_score(actual, pred, zero_division=0)
                            recall = recall_score(actual, pred, zero_division=0)
                            f1 = f1_score(actual, pred, zero_division=0)
                            mcc = matthews_corrcoef(actual, pred)

                            # Calculate AUC if probabilities are available and both classes exist
                            auc = None
                            auc_note = ""
                            if model_name in prediction_probas and prediction_probas[model_name] is not None:
                                try:
                                    # Check if both classes exist in actual values
                                    unique_classes = np.unique(actual)
                                    if len(unique_classes) < 2:
                                        auc_note = "(Single class)"
                                    elif np.all(pred == actual[0]):  # All predictions are the same
                                        auc_note = "(Const. pred)"
                                    else:
                                        auc = roc_auc_score(actual, prediction_probas[model_name])
                                except ValueError as e:
                                    auc_note = f"(Error: {str(e)[:20]})"
                                except Exception as e:
                                    auc_note = "(N/A)"

                            metrics_data.append({
                                'Model': model_name,
                                'Accuracy': f"{acc:.4f}",
                                'AUC': f"{auc:.4f} {auc_note}" if auc is not None else f"N/A {auc_note}",
                                'Precision': f"{precision:.4f}",
                                'Recall': f"{recall:.4f}",
                                'F1 Score': f"{f1:.4f}",
                                'MCC': f"{mcc:.4f}"
                            })

                        # Add ensemble metrics if available
                        if 'Ensemble (Majority Vote)' in results_df.columns:
                            ensemble_pred = results_df['Ensemble (Majority Vote)'].values
                            acc = accuracy_score(actual, ensemble_pred)
                            precision = precision_score(actual, ensemble_pred, zero_division=0)
                            recall = recall_score(actual, ensemble_pred, zero_division=0)
                            f1 = f1_score(actual, ensemble_pred, zero_division=0)
                            mcc = matthews_corrcoef(actual, ensemble_pred)

                            metrics_data.append({
                                'Model': 'Ensemble (Majority Vote)',
                                'Accuracy': f"{acc:.4f}",
                                'AUC': "N/A",
                                'Precision': f"{precision:.4f}",
                                'Recall': f"{recall:.4f}",
                                'F1 Score': f"{f1:.4f}",
                                'MCC': f"{mcc:.4f}"
                            })

                        # Create metrics DataFrame
                        metrics_df = pd.DataFrame(metrics_data)

                        # Display metrics table with styling
                        st.dataframe(metrics_df, use_container_width=True)

                        # Create a comparison chart for metrics
                        st.subheader("📈 Metrics Comparison Chart")

                        # Prepare data for plotting (exclude ensemble and N/A values)
                        plot_metrics = []
                        for data in metrics_data:
                            if data['Model'] != 'Ensemble (Majority Vote)':
                                plot_metrics.append({
                                    'Model': data['Model'],
                                    'Accuracy': float(data['Accuracy']),
                                    'Precision': float(data['Precision']),
                                    'Recall': float(data['Recall']),
                                    'F1 Score': float(data['F1 Score']),
                                    'MCC': float(data['MCC'])
                                })

                        plot_df = pd.DataFrame(plot_metrics)
                        plot_df_melted = plot_df.melt(
                            id_vars=['Model'],
                            var_name='Metric',
                            value_name='Score'
                        )

                        # Create grouped bar chart
                        fig, ax = plt.subplots(figsize=(14, 6))

                        models_list = plot_df['Model'].tolist()
                        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']
                        x = np.arange(len(models_list))
                        width = 0.15

                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                        for i, metric in enumerate(metrics_list):
                            values = [float(m[metric]) for _, m in plot_df.iterrows()]
                            ax.bar(x + i * width, values, width, label=metric, color=colors[i])

                        ax.set_xlabel('Models', fontweight='bold')
                        ax.set_ylabel('Score', fontweight='bold')
                        ax.set_title('Evaluation Metrics Comparison by Model', fontweight='bold', fontsize=14)
                        ax.set_xticks(x + width * 2)
                        ax.set_xticklabels(models_list, rotation=45, ha='right')
                        ax.legend(title='Metrics', loc='lower right')
                        ax.set_ylim(0, 1.1)
                        ax.grid(axis='y', alpha=0.3)

                        # Add value labels on bars
                        for i, metric in enumerate(metrics_list):
                            values = [float(m[metric]) for _, m in plot_df.iterrows()]
                            for j, v in enumerate(values):
                                ax.text(j + i * width, v + 0.02, f'{v:.3f}',
                                       ha='center', va='bottom', fontsize=7)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Create a heatmap for easier comparison
                        st.subheader("🔥 Metrics Heatmap")

                        # Prepare heatmap data
                        heatmap_data = plot_df.set_index('Model')

                        fig2, ax2 = plt.subplots(figsize=(10, 6))

                        sns.heatmap(
                            heatmap_data,
                            annot=True,
                            fmt='.3f',
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=1,
                            cbar_kws={'label': 'Score'},
                            linewidths=0.5,
                            ax=ax2
                        )

                        ax2.set_title('Evaluation Metrics Heatmap', fontweight='bold', fontsize=14)
                        ax2.set_xlabel('Metrics', fontweight='bold')
                        ax2.set_ylabel('Models', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig2)

                        # Best model for each metric
                        st.subheader("🏆 Best Model by Metric")

                        best_models = {}
                        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']:
                            best_idx = plot_df[metric].idxmax()
                            best_models[metric] = plot_df.loc[best_idx, 'Model']

                        best_df = pd.DataFrame({
                            'Metric': list(best_models.keys()),
                            'Best Model': list(best_models.values())
                        })
                        st.dataframe(best_df, use_container_width=True)

                        # Confusion Matrices
                        st.subheader("🎯 Confusion Matrices")

                        n_models = len(predictions)
                        n_cols = 3
                        n_rows = (n_models + n_cols - 1) // n_cols

                        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                        if n_models == 1:
                            axes3 = np.array([axes3])
                        axes3 = axes3.flatten()

                        for idx, (model_name, pred) in enumerate(predictions.items()):
                            cm = confusion_matrix(actual, pred)
                            ax = axes3[idx]

                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt='d',
                                cmap='Blues',
                                cbar=False,
                                ax=ax,
                                xticklabels=['Not Bankrupt', 'Bankrupt'],
                                yticklabels=['Not Bankrupt', 'Bankrupt']
                            )
                            ax.set_title(f'{model_name}', fontweight='bold')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')

                        # Hide empty subplots
                        for idx in range(len(predictions), len(axes3)):
                            axes3[idx].axis('off')

                        plt.tight_layout()
                        st.pyplot(fig3)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Tip:</strong> For best results, ensure your input data has the same format as the training data.</p>
    <p>Made with ❤️ using Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
