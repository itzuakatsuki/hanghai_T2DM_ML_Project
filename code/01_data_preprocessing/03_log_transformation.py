# 資料Log轉換
import pandas as pd
import numpy as np

# =========================
# 1. 讀取資料
df = pd.read_csv(
    "Shanghai_T2DM_Median_Imputed.csv"
)

# =========================
# 2. 不做 log 的 binary variables
binary_cols = [
    "Gender_binary",
    "Smoking_binary",
    "Alcohol_binary",
    "Macrovascular_Complication",
    "Microvascular_Complication",
    "Comorbidities_binary",
    "Hypoglycemic_Agents_binary",
    "Other_Agents_binary",
    "Hypoglycemia_binary"
]

# =========================
# 3. 不做 log 的 body-size variables
non_log_continuous = [
    "Age (years)",
    "Height (cm)",
    "Weight (kg)",
    "BMI (kg/m2)"
]

# =========================
# 4. 合併排除欄位
exclude_cols = (
    binary_cols
    +
    non_log_continuous
)

# =========================
# 5. 找出 numeric variables
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

# =========================
# 6. 自動做 log transform
log_transformed_cols = []

for col in numeric_cols:
    if col not in exclude_cols:
        if (df[col] >= 0).all():
            df[f"log_{col}"] = np.log1p(
                df[col]
            )
            log_transformed_cols.append(col)
            print(f"完成 log transform: {col}")

# =========================
# 7. 查看完成欄位
print("\n完成 log transform 的欄位：")
print(log_transformed_cols)

# =========================
# 8. 輸出資料
df.to_csv(
    "Shanghai_T2DM_Log_Transformed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nLog transform 完成")
print("輸出檔案：Shanghai_T2DM_Log_Transformed.csv")