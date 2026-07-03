"""
CGM 馬克夫鏈 + 邏輯斯迴歸預測模型
目標：預測下一個 15 分鐘是高/中/低血糖的哪一類

血糖狀態定義 (mg/dL)：
  S0 = Low: < 70
  S1 = Target: 70-180
  S2 = High: > 180

輸入：combined_cgm_data.csv (需先執行 data_preparation.py)
輸出：
  - transition_count.csv (轉移計數)
  - transition_probability.csv (轉移機率)
  - logistic_coefficients.csv (模型係數)
  - 模型評估指標 (混淆矩陣、分類報告)
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold,
    cross_validate
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================
# 1. 讀資料
# =====================================

df = pd.read_csv(
    "combined_cgm_data.csv"
)

# =====================================
# 2. 定義血糖狀態
# =====================================
# S0 = Low
# S1 = Target
# S2 = High

def glucose_state(x):
    """
    根據 CGM 值定義血糖狀態
    
    Args:
        x: CGM 血糖值 (mg/dL)
    
    Returns:
        0 (Low), 1 (Target), 2 (High)
    """
    if x < 70:
        return 0
    elif x <= 180:
        return 1
    else:
        return 2


df["state"] = df["CGM"].apply(
    glucose_state
)

# =====================================
# 3. 排序
# =====================================

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = df.sort_values(
    ["patient_id", "datetime"]
)

df["time_diff_min"] = (
    df.groupby("patient_id")["datetime"]
      .diff()
      .dt.total_seconds() / 60
)

df = df[
    df["time_diff_min"].between(14, 16)
].copy()

# =====================================
# 4. 建立 t-1 特徵 (15分鐘前的狀態和血糖值)
# =====================================

df["prev_state"] = (
    df.groupby("patient_id")["state"]
      .shift(1)
)

df["prev_cgm"] = (
    df.groupby("patient_id")["CGM"]
      .shift(1)
)

df["prev2_cgm"] = (
    df.groupby("patient_id")["CGM"]
      .shift(2)
)

# 血糖變化量 (t-1 - t-2)
df["prev_delta"] = (
    df["prev_cgm"]
    -
    df["prev2_cgm"]
)

# =====================================
# 5. 移除缺失
# =====================================

df = df.dropna(
    subset=[
        "prev_state",
        "prev_cgm",
        "prev_delta"
    ]
)

# =====================================
# 6. 建立馬克夫轉移矩陣
# =====================================

transition_count = pd.crosstab(
    df["prev_state"],
    df["state"]
)

transition_prob = transition_count.div(
    transition_count.sum(axis=1),
    axis=0
)

print("\n========================")
print("Transition Count (轉移計數)")
print("========================")
print(transition_count)

print("\n========================")
print("Transition Probability (轉移機率)")
print("========================")
print(
    transition_prob.round(4)
)

# =====================================
# 7. 準備邏輯斯迴歸特徵
# =====================================

X = pd.get_dummies(
    df[["prev_state"]],
    drop_first=True
)

X["prev_cgm"] = df["prev_cgm"]

X["prev_delta"] = df["prev_delta"]

y = df["state"]

groups = df["patient_id"]

# =====================================
# 8. 交叉驗證 (Leave-One-Patient-Out)
# =====================================

cv = GroupKFold(
    n_splits=5
)

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    groups=groups,
    scoring={
      "accuracy": "accuracy",
      "balanced_accuracy": "balanced_accuracy",
      "f1_macro": "f1_macro"
    }
)

print("\n========================")
print("Cross Validation (交叉驗證)")
print("========================")

print(
    "Accuracy Mean =",
    scores["test_accuracy"].mean()
)

print(
    "Accuracy SD =",
    scores["test_accuracy"].std()
)

# =====================================
# 9. 用全部資料訓練模型
# =====================================

model.fit(X, y)

pred = model.predict(X)

# =====================================
# 10. 混淆矩陣
# =====================================

cm = confusion_matrix(
    y,
    pred
)

print("\n========================")
print("Confusion Matrix (混淆矩陣)")
print("========================")
print(cm)

# =====================================
# 11. 分類報告
# =====================================

print("\n========================")
print("Classification Report (分類報告)")
print("========================")

print(
    classification_report(
        y,
        pred,
        target_names=[
            "Low",
            "Target",
            "High"
        ]
    )
)

# =====================================
# 12. 模型係數
# =====================================

coef_df = pd.DataFrame(
    model.coef_.T,
    index=X.columns,
    columns=[
        "Low",
        "Target",
        "High"
    ]
)

print("\n========================")
print("Coefficients (模型係數)")
print("========================")

print(
    coef_df
)

# =====================================
# 13. 匯出結果
# =====================================

transition_count.to_csv(
    "transition_count.csv",
    encoding="utf-8-sig"
)

transition_prob.to_csv(
    "transition_probability.csv",
    encoding="utf-8-sig"
)

coef_df.to_csv(
    "logistic_coefficients.csv",
    encoding="utf-8-sig"
)

print("\n完成輸出")
