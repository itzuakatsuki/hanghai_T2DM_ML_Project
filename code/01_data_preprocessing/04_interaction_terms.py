# 交互作用
import pandas as pd

# =========================
# 1. 讀取 LOG 轉換後資料
df = pd.read_csv("Shanghai_T2DM_Log_Transformed.csv")

# =========================
# 2. 建立交互作用項
# 年齡 × BMI
df["Interaction_Age_BMI"] = (
    df["Age (years)"] * df["BMI (kg/m2)"]
)

# 糖化血色素 × 罹病時間
df["Interaction_HbA1c_Duration"] = (
    df["log_HbA1c (mmol/mol)"] *
    df["log_Duration of diabetes (years)"]
)

# 空腹血糖 × 三酸甘油酯
df["Interaction_FPG_TG"] = (
    df["log_Fasting Plasma Glucose (mg/dl)"] *
    df["log_Triglyceride (mmol/L)"]
)

# BMI × 吸菸狀態
df["Interaction_BMI_Smoking"] = (
    df["BMI (kg/m2)"] *
    df["Smoking_binary"]
)

# =========================
# 3. 輸出新資料
df.to_csv(
    "Shanghai_T2DM_Log_Interaction.csv",
    index=False,
    encoding="utf-8-sig"
)

print("交互作用項建立完成")
print("新增欄位：")
print([
    "Interaction_Age_BMI",
    "Interaction_HbA1c_Duration",
    "Interaction_FPG_TG",
    "Interaction_BMI_Smoking"
])