#XGBoost
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

# =========================
# 1. 讀取資料
df = pd.read_csv(
    "Shanghai_T2DM_Standardized.csv"
)

# =========================
# 2. 設定目標變數
target_cols = [
    "Macrovascular_Complication",
    "Microvascular_Complication",
    "Hypoglycemia_binary"
]

# =========================
# 3. 建立自變數 X
# 使用 z-score 欄位
z_cols = [
    col for col in df.columns
    if col.startswith("z_")
]

# 保留 binary variables
binary_cols = [
    "Gender_binary",
    "Smoking_binary",
    "Alcohol_binary",
    "Comorbidities_binary",
    "Hypoglycemic_Agents_binary",
    "Other_Agents_binary"
]

feature_cols = (
    z_cols
    +
    binary_cols
)
X = df[feature_cols].copy()
# 保險轉 numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)
# 缺失值補中位數
X = X.fillna(
    X.median()
)

# =========================
# 4. Cross-validation 設定
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
scoring = {
    "accuracy":"accuracy",
    "precision":"precision",
    "recall":"recall",
    "f1":"f1",
    "auc":"roc_auc"
}

# =========================
# 5. XGBoost 建模
results = []

for target in target_cols:
    print("="*60)
    print("Target:", target)
    y = df[target].astype(int)

    # 檢查是否可建模
    if y.nunique() < 2:
        print("只有單一類別，跳過")
        continue

    # 權重
    neg = sum(y == 0)
    pos = sum(y == 1)
    scale_pos_weight = (
        neg / pos
        if pos > 0
        else 1
    )

    # 建立模型
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
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

    result = {
        "Target":target,
        "Accuracy_mean":
            np.nanmean(
                cv_scores["test_accuracy"]
            ),
        "Precision_mean":
            np.nanmean(
                cv_scores["test_precision"]
            ),
        "Recall_mean":
            np.nanmean(
                cv_scores["test_recall"]
            ),
        "F1_mean":
            np.nanmean(
                cv_scores["test_f1"]
            ),
        "AUC_mean":
            np.nanmean(
                cv_scores["test_auc"]
            ),
        "Accuracy_sd":
            np.nanstd(
                cv_scores["test_accuracy"]
            ),
        "Precision_sd":
            np.nanstd(
                cv_scores["test_precision"]
            ),
        "Recall_sd":
            np.nanstd(
                cv_scores["test_recall"]
            ),
        "F1_sd":
            np.nanstd(
                cv_scores["test_f1"]
            ),
        "AUC_sd":
            np.nanstd(
                cv_scores["test_auc"]
            ),
        "Positive_Count":
            int(pos),
        "Negative_Count":
            int(neg)
    }
    results.append(result)

# =========================
# 6. 輸出結果
results_df = pd.DataFrame(results)

print("\nXGBoost Results")
print(results_df)

results_df.to_csv(
    "XGBoost_Results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n完成")
print("輸出檔案：XGBoost_Results.csv")
