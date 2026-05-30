import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from xgboost import XGBRegressor

# =========================
# 1. 讀取資料
# =========================
df = pd.read_csv("Shanghai_T2DM_Standardized.csv")

# =========================
# 2. 設定迴歸目標變數
# =========================
target = "HbA1c (mmol/mol)"
y = df[target]

# =========================
# 3. 建立特徵變數 X
# =========================

# 使用 z-score 欄位
z_cols = [
    col for col in df.columns
    if col.startswith("z_")
]

binary_cols = [
    "Gender_binary",
    "Smoking_binary",
    "Alcohol_binary",
    "Comorbidities_binary",
    "Hypoglycemic_Agents_binary",
    "Other_Agents_binary"
]

feature_cols = z_cols + binary_cols

# =========================
# 4. 移除目標相關欄位，避免 data leakage
# =========================
leakage_cols = [
    "HbA1c (mmol/mol)",
    "z_HbA1c (mmol/mol)",
    "log_HbA1c (mmol/mol)",
    "z_log_HbA1c (mmol/mol)",
    "Interaction_HbA1c_Duration",
    "z_Interaction_HbA1c_Duration"
]

feature_cols = [
    col for col in feature_cols
    if col not in leakage_cols
]

X = df[feature_cols].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# =========================
# 5. Cross-validation
# =========================
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================
# 6. 評估函數
# =========================
def evaluate_regression(model, X, y, model_name):
    
    pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )
    
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mape = mean_absolute_percentage_error(y, pred) * 100
    r2 = r2_score(y, pred)
    
    print("=" * 50)
    print(model_name)
    print("MAE  =", mae)
    print("RMSE =", rmse)
    print("MAPE =", mape)
    print("R²   =", r2)
    
    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

# =========================
# 7. Linear Regression
# =========================
lr_model = LinearRegression()

lr_result = evaluate_regression(
    lr_model,
    X,
    y,
    "Linear Regression"
)

# =========================
# 8. XGBoost Regressor
# =========================
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_result = evaluate_regression(
    xgb_model,
    X,
    y,
    "XGBoost Regressor"
)

# =========================
# 9. 輸出結果
# =========================
results_df = pd.DataFrame([
    lr_result,
    xgb_result
])

print(results_df)

results_df.to_csv(
    "Regression_Model_Results.csv",
    index=False,
    encoding="utf-8-sig"
)