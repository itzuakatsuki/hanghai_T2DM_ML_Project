import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# =========================
# 1. 設置輸出目錄
output_dir = "output"
viz_dir = os.path.join(output_dir, "visualizations")
results_dir = os.path.join(output_dir, "model_results")

os.makedirs(viz_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2. 設置風格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# =========================
# 3. 讀取所有結果文件
try:
    lr_results = pd.read_csv(os.path.join(results_dir, "Logistic_Regression_Results.csv"))
    print("✓ 已讀取 Logistic Regression 結果")
except FileNotFoundError:
    print("✗ 找不到 Logistic Regression 結果")
    lr_results = None

try:
    xgb_clf_results = pd.read_csv(os.path.join(results_dir, "XGBoost_Results.csv"))
    print("✓ 已讀取 XGBoost 分類結果")
except FileNotFoundError:
    print("✗ 找不到 XGBoost 分類結果")
    xgb_clf_results = None

try:
    reg_results = pd.read_csv(os.path.join(results_dir, "Regression_Model_Results.csv"))
    print("✓ 已讀取 迴歸模型結果")
except FileNotFoundError:
    print("✗ 找不到 迴歸模型結果")
    reg_results = None

# =========================
# 4. 分類模型性能對比圖
if lr_results is not None and xgb_clf_results is not None:
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Classification Models Performance Comparison', fontsize=16, fontweight='bold')
    
    # 合併 LR 和 XGBoost 結果
    lr_results['Model'] = 'Logistic Regression'
    xgb_clf_results['Model'] = 'XGBoost'
    
    comparison_clf = pd.concat([lr_results, xgb_clf_results], ignore_index=True)
    
    # 1. Accuracy 對比
    metrics_data = []
    for _, row in comparison_clf.iterrows():
        metrics_data.append({
            'Target': row['Target'],
            'Model': row['Model'],
            'Accuracy': row['Accuracy_mean'],
            'Std': row['Accuracy_sd']
        })
    
    acc_df = pd.DataFrame(metrics_data)
    sns.barplot(data=acc_df, x='Target', y='Accuracy', hue='Model', ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].set_ylabel('Accuracy')
    
    # 2. Precision 對比
    prec_df = []
    for _, row in comparison_clf.iterrows():
        prec_df.append({
            'Target': row['Target'],
            'Model': row['Model'],
            'Precision': row['Precision_mean']
        })
    
    prec_data = pd.DataFrame(prec_df)
    sns.barplot(data=prec_data, x='Target', y='Precision', hue='Model', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Precision Comparison', fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_ylabel('Precision')
    
    # 3. Recall 對比
    rec_df = []
    for _, row in comparison_clf.iterrows():
        rec_df.append({
            'Target': row['Target'],
            'Model': row['Model'],
            'Recall': row['Recall_mean']
        })
    
    rec_data = pd.DataFrame(rec_df)
    sns.barplot(data=rec_data, x='Target', y='Recall', hue='Model', ax=axes[0, 2], palette='Set2')
    axes[0, 2].set_title('Recall Comparison', fontweight='bold')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].set_ylabel('Recall')
    
    # 4. F1-Score 對比
    f1_df = []
    for _, row in comparison_clf.iterrows():
        f1_df.append({
            'Target': row['Target'],
            'Model': row['Model'],
            'F1': row['F1_mean']
        })
    
    f1_data = pd.DataFrame(f1_df)
    sns.barplot(data=f1_data, x='Target', y='F1', hue='Model', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].set_ylabel('F1-Score')
    
    # 5. AUC-ROC 對比
    auc_df = []
    for _, row in comparison_clf.iterrows():
        auc_df.append({
            'Target': row['Target'],
            'Model': row['Model'],
            'AUC': row['AUC_mean']
        })
    
    auc_data = pd.DataFrame(auc_df)
    sns.barplot(data=auc_data, x='Target', y='AUC', hue='Model', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('AUC-ROC Comparison', fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_ylabel('AUC-ROC')
    
    # 6. 類別分佈
    class_dist = []
    for _, row in comparison_clf.iterrows():
        total = row['Positive_Count'] + row['Negative_Count']
        pos_pct = (row['Positive_Count'] / total) * 100
        class_dist.append({
            'Target': row['Target'],
            'Positive %': pos_pct,
            'Negative %': 100 - pos_pct
        })
    
    class_data = pd.DataFrame(class_dist).drop_duplicates(subset=['Target'])
    class_data.set_index('Target')[['Positive %', 'Negative %']].plot(
        kind='bar', stacked=True, ax=axes[1, 2], color=['#ff9999', '#66b3ff']
    )
    axes[1, 2].set_title('Class Distribution', fontweight='bold')
    axes[1, 2].set_ylabel('Percentage (%)')
    axes[1, 2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'classification_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 已保存 classification_comparison.png")
    plt.close()

# =========================
# 5. 迴歸模型性能對比圖
if reg_results is not None:
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regression Models Performance Comparison (HbA1c Prediction)', 
                 fontsize=16, fontweight='bold')
    
    # 1. MAE 對比
    sns.barplot(data=reg_results, x='Model', y='MAE', ax=axes[0, 0], palette='Set1')
    axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0, 0].set_ylabel('MAE')
    
    # 2. RMSE 對比
    sns.barplot(data=reg_results, x='Model', y='RMSE', ax=axes[0, 1], palette='Set1')
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
    axes[0, 1].set_ylabel('RMSE')
    
    # 3. MAPE 對比
    sns.barplot(data=reg_results, x='Model', y='MAPE', ax=axes[1, 0], palette='Set1')
    axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE %)', fontweight='bold')
    axes[1, 0].set_ylabel('MAPE (%)')
    
    # 4. R² 對比
    sns.barplot(data=reg_results, x='Model', y='R2', ax=axes[1, 1], palette='Set1')
    axes[1, 1].set_title('R² Score (Coefficient of Determination)', fontweight='bold')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'regression_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 已保存 regression_comparison.png")
    plt.close()

# =========================
# 6. 建立模型性能總結表格 (HTML)
if lr_results is not None or xgb_clf_results is not None or reg_results is not None:
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Model Results Summary</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f0f0f0;
            }
            .section {
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .timestamp {
                color: #666;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <h1>🏥 Shanghai T2DM ML Project - Model Results Summary</h1>
        <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """
    
    # 分類模型結果
    if lr_results is not None:
        html_content += """
        <div class="section">
            <h2>📊 Logistic Regression - Classification Results</h2>
            <table>
                <tr>
                    <th>Target Variable</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>AUC-ROC</th>
                    <th>Positive Count</th>
                    <th>Negative Count</th>
                </tr>
        """
        for _, row in lr_results.iterrows():
            html_content += f"""
                <tr>
                    <td><strong>{row['Target']}</strong></td>
                    <td>{row['Accuracy_mean']:.4f} ± {row['Accuracy_sd']:.4f}</td>
                    <td>{row['Precision_mean']:.4f} ± {row['Precision_sd']:.4f}</td>
                    <td>{row['Recall_mean']:.4f} ± {row['Recall_sd']:.4f}</td>
                    <td>{row['F1_mean']:.4f} ± {row['F1_sd']:.4f}</td>
                    <td>{row['AUC_mean']:.4f} ± {row['AUC_sd']:.4f}</td>
                    <td>{int(row['Positive_Count'])}</td>
                    <td>{int(row['Negative_Count'])}</td>
                </tr>
            """
        html_content += """
            </table>
        </div>
        """
    
    if xgb_clf_results is not None:
        html_content += """
        <div class="section">
            <h2>📊 XGBoost - Classification Results</h2>
            <table>
                <tr>
                    <th>Target Variable</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>AUC-ROC</th>
                    <th>Positive Count</th>
                    <th>Negative Count</th>
                </tr>
        """
        for _, row in xgb_clf_results.iterrows():
            html_content += f"""
                <tr>
                    <td><strong>{row['Target']}</strong></td>
                    <td>{row['Accuracy_mean']:.4f} ± {row['Accuracy_sd']:.4f}</td>
                    <td>{row['Precision_mean']:.4f} ± {row['Precision_sd']:.4f}</td>
                    <td>{row['Recall_mean']:.4f} ± {row['Recall_sd']:.4f}</td>
                    <td>{row['F1_mean']:.4f} ± {row['F1_sd']:.4f}</td>
                    <td>{row['AUC_mean']:.4f} ± {row['AUC_sd']:.4f}</td>
                    <td>{int(row['Positive_Count'])}</td>
                    <td>{int(row['Negative_Count'])}</td>
                </tr>
            """
        html_content += """
            </table>
        </div>
        """
    
    # 迴歸模型結果
    if reg_results is not None:
        html_content += """
        <div class="section">
            <h2>📈 Regression Models - HbA1c Prediction Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>MAPE (%)</th>
                    <th>R² Score</th>
                </tr>
        """
        for _, row in reg_results.iterrows():
            html_content += f"""
                <tr>
                    <td><strong>{row['Model']}</strong></td>
                    <td>{row['MAE']:.4f}</td>
                    <td>{row['RMSE']:.4f}</td>
                    <td>{row['MAPE']:.2f}%</td>
                    <td>{row['R2']:.4f}</td>
                </tr>
            """
        html_content += """
            </table>
        </div>
        """
    
    html_content += """
        <div class="section">
            <h2>📝 Key Metrics Explanation</h2>
            <ul>
                <li><strong>Accuracy:</strong> Proportion of correct predictions</li>
                <li><strong>Precision:</strong> Proportion of positive predictions that are correct</li>
                <li><strong>Recall:</strong> Proportion of actual positives correctly identified</li>
                <li><strong>F1-Score:</strong> Harmonic mean of Precision and Recall</li>
                <li><strong>AUC-ROC:</strong> Area Under the Receiver Operating Characteristic Curve</li>
                <li><strong>MAE:</strong> Mean Absolute Error for regression</li>
                <li><strong>RMSE:</strong> Root Mean Squared Error for regression</li>
                <li><strong>MAPE:</strong> Mean Absolute Percentage Error for regression</li>
                <li><strong>R²:</strong> Coefficient of Determination (0-1, higher is better)</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # 保存 HTML 文件
    html_path = os.path.join(results_dir, 'model_results_summary.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ 已保存 model_results_summary.html")

# =========================
# 7. 生成 Markdown 總結報告
if lr_results is not None or xgb_clf_results is not None or reg_results is not None:
    
    md_content = """# 🏥 Shanghai T2DM ML Project - Results Summary

**Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

## 📊 Overview

This report summarizes the machine learning model performance for predicting Type 2 Diabetes Mellitus complications.

### Targets:
- **Macrovascular Complications** (Coronary, Cerebral, Peripheral Events)
- **Microvascular Complications** (Retinopathy, Nephropathy, Neuropathy)
- **Hypoglycemia** (Binary: Yes/No)
- **HbA1c Levels** (Continuous: Regression)

---

## 🏆 Classification Models Performance

### Logistic Regression
"""
    
    if lr_results is not None:
        md_content += "| Target | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
        md_content += "|--------|----------|-----------|--------|----------|----------|\n"
        for _, row in lr_results.iterrows():
            md_content += f"| {row['Target']} | {row['Accuracy_mean']:.4f}±{row['Accuracy_sd']:.4f} | {row['Precision_mean']:.4f}±{row['Precision_sd']:.4f} | {row['Recall_mean']:.4f}±{row['Recall_sd']:.4f} | {row['F1_mean']:.4f}±{row['F1_sd']:.4f} | {row['AUC_mean']:.4f}±{row['AUC_sd']:.4f} |\n"
    
    md_content += "\n### XGBoost Classifier\n"
    
    if xgb_clf_results is not None:
        md_content += "| Target | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n"
        md_content += "|--------|----------|-----------|--------|----------|----------|\n"
        for _, row in xgb_clf_results.iterrows():
            md_content += f"| {row['Target']} | {row['Accuracy_mean']:.4f}±{row['Accuracy_sd']:.4f} | {row['Precision_mean']:.4f}±{row['Precision_sd']:.4f} | {row['Recall_mean']:.4f}±{row['Recall_sd']:.4f} | {row['F1_mean']:.4f}±{row['F1_sd']:.4f} | {row['AUC_mean']:.4f}±{row['AUC_sd']:.4f} |\n"
    
    md_content += "\n---\n\n## 📈 Regression Models Performance (HbA1c Prediction)\n\n"
    
    if reg_results is not None:
        md_content += "| Model | MAE | RMSE | MAPE (%) | R² Score |\n"
        md_content += "|-------|-----|------|----------|----------|\n"
        for _, row in reg_results.iterrows():
            md_content += f"| {row['Model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['MAPE']:.2f}% | {row['R2']:.4f} |\n"
    
    md_content += """
---

## 🎯 Key Findings

### Classification Results
- **Best Model:** Compare AUC-ROC scores across models
- **Balanced Performance:** Check F1-scores for class imbalance handling
- **Class Distribution:** See positive vs. negative samples in results

### Regression Results
- **Best Predictor:** Lower MAE/RMSE indicates better predictions
- **R² Score:** Higher R² (closer to 1) indicates better model fit
- **MAPE:** Percentage error metric for practical interpretation

---

## 📁 Output Files

### CSV Files
- `Logistic_Regression_Results.csv` - Detailed LR results
- `XGBoost_Results.csv` - Detailed XGBoost results
- `Regression_Model_Results.csv` - Regression model results

### Visualizations
- `classification_comparison.png` - Classification metrics comparison
- `regression_comparison.png` - Regression metrics comparison

### HTML Report
- `model_results_summary.html` - Interactive HTML summary

---

## 📝 Next Steps

1. Review the visualizations to identify best performing models
2. Analyze feature importance for key predictors
3. Consider ensemble methods or hyperparameter tuning
4. Validate results on hold-out test set
5. Deploy model to production environment

---

**Project Repository:** https://github.com/itzuakatsuki/hanghai_T2DM_ML_Project
"""
    
    # 保存 Markdown 文件
    md_path = os.path.join(results_dir, 'results_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✓ 已保存 results_summary.md")

print("\n" + "="*50)
print("✓ 所有可視化和報告已生成完成！")
print("="*50)
print(f"\n📁 輸出位置:")
print(f"   - 圖表: {viz_dir}/")
print(f"   - 報告: {results_dir}/")
