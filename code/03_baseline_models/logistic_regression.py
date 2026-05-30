# Logistic Regression
'''
目標變數
Macrovascular_Complication
Microvascular_Complication
Hypoglycemia_binary
'''
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# =========================
# 1. 讀取資料
df = pd.read_csv("Shanghai_T2DM_Standardized.csv")

# =========================
# 2. 設定目標變數
target_cols = [
    "Macrovascular_Complication",
    "Microvascular_Complication",
    "Hypoglycemia_binary"
]

# =========================
# 3. 設定自變數 X
# 使用標準化後 z_ 欄位 + 二元變數
z_cols = [col for col in df.columns if col.startswith("z_")]

binary_cols = [
    "Gender_binary",
    "Smoking_binary",
    "Alcohol_binary",
    "Comorbidities_binary",
    "Hypoglycemic_Agents_binary",
    "Other_Agents_binary"
]

feature_cols = z_cols + binary_cols

X = df[feature_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# =========================
# 4. Cross-validation 設定
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc"
}

# =========================
# 5. Logistic Regression
results = []

for target in target_cols:
    y = df[target].astype(int)
    
    # 如果某目標只有 0 或只有 1，不能建模
    if y.nunique() < 2:
        print(f"{target} 只有單一類別，無法建模")
        continue
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )
    
    # Cross-validation
    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score=np.nan
    )
    
    results.append({
        "Target": target,
        "Accuracy_mean": np.nanmean(cv_scores["test_accuracy"]),
        "Precision_mean": np.nanmean(cv_scores["test_precision"]),
        "Recall_mean": np.nanmean(cv_scores["test_recall"]),
        "F1_mean": np.nanmean(cv_scores["test_f1"]),
        "AUC_mean": np.nanmean(cv_scores["test_auc"]),
        "Accuracy_sd": np.nanstd(cv_scores["test_accuracy"]),
        "Precision_sd": np.nanstd(cv_scores["test_precision"]),
        "Recall_sd": np.nanstd(cv_scores["test_recall"]),
        "F1_sd": np.nanstd(cv_scores["test_f1"]),
        "AUC_sd": np.nanstd(cv_scores["test_auc"]),
        "Positive_Count": int(y.sum()),
        "Negative_Count": int((y == 0).sum())
    })

# =========================
# 6. 輸出結果
results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    "Logistic_Regression_Results.csv",
    index=False,
    encoding="utf-8-sig"
)