# 用中位數補缺失值
import pandas as pd
import numpy as np

# =========================
# 1. 讀取已處理類別變數資料
df = pd.read_csv(
    "Shanghai_T2DM_Categorical_Processed.csv"
)

# =========================
# 2. 將 "/" 轉成 NaN
df = df.replace("/", np.nan)

# =========================
# 3. 轉成數值型態
for col in df.columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="ignore"
    )

# =========================
# 4. 找出數值欄位
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

# =========================
# 5. Median Imputation
for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(
        median_value
    )

# =========================
# 6. 檢查是否還有缺失值
missing_summary = (
    df[numeric_cols]
    .isnull()
    .sum()
)
print("剩餘缺失值：")
print(
    missing_summary[
        missing_summary > 0
    ]
)

# =========================
# 7. 輸出新資料
df.to_csv(
    "Shanghai_T2DM_Median_Imputed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Median imputation 完成")
print("輸出檔案：Shanghai_T2DM_Median_Imputed.csv")