import pandas as pd
import numpy as np

df = pd.read_csv("Shanghai_T2DM_Standardized.csv")

# 數值變數
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

eda_summary = pd.DataFrame({
    "Mean": df[numeric_cols].mean(),
    "SD": df[numeric_cols].std(),
    "Missing_Count": df[numeric_cols].isnull().sum(),
    "Missing_%": (
        df[numeric_cols].isnull().mean() * 100
    ),
    "Skewness": df[numeric_cols].skew()
})

eda_summary = eda_summary.round(3)

print(eda_summary)

eda_summary.to_csv(
    "EDA_Summary.csv",
    encoding="utf-8-sig"
)

eda_summary["Skew_Type"] = np.where(
    abs(eda_summary["Skewness"]) < 0.5,
    "Approximately Normal",
    np.where(
        abs(eda_summary["Skewness"]) < 1,
        "Moderately Skewed",
        "Highly Skewed"
    )
)

print(
    eda_summary[
        ["Skewness","Skew_Type"]
    ]
)