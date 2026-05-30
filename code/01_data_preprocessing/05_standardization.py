# 標準化
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# =========================
# 1. 讀取資料
df = pd.read_csv(
    "Shanghai_T2DM_Log_Interaction.csv"
)

# =========================
# 2. 不做 standardization 的 binary variables
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
# 3. 找 numeric variables
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# =========================
# 4. 要 standardize 的欄位
scale_cols = [
    col for col in numeric_cols
    if col not in binary_cols
]
print("需要標準化的欄位：")
print(scale_cols)

# =========================
# 5. 建立 scaler
scaler = StandardScaler()

# =========================
# 6. 做 Z-score standardization
scaled_values = scaler.fit_transform(
    df[scale_cols]
)

scaled_df = pd.DataFrame(
    scaled_values,
    columns=[f"z_{col}" for col in scale_cols]
)

# =========================
# 7. 合併回原資料
df_standardized = pd.concat(
    [df, scaled_df],
    axis=1
)

# =========================
# 8. 查看結果
print(df_standardized.head())

# =========================
# 9. 輸出資料
df_standardized.to_csv(
    "Shanghai_T2DM_Standardized.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nStandardization 完成")
print("輸出檔案：Shanghai_T2DM_Standardized.csv")