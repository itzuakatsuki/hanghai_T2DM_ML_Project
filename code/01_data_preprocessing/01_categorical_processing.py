# 類別資料處理
'''
Patient Number 刪除 
Gender (Female=1, Male=2) 改 0, 1 
Smoking History (pack year) 改 0(沒抽過菸), 1(有抽過菸) 
Alcohol Drinking History 改0(沒喝過酒), 1(有喝過酒)
Type of Diabetes 刪除欄位 
Acute Diabetic Complications 刪除欄位 
Diabetic Macrovascular Complications 改 0(無), 1(有) 
Diabetic Microvascular Complications 改 0(無), 1(有) 
Comorbidities 改 0(無), 1(有)
Hypoglycemic Agents 改 0(無), 1(有) 
Other Agents 改 0(無), 1(有) 
Hypoglycemia (yes/no) 改 0(無), 1(有)
'''
import pandas as pd
import numpy as np

# =========================
# 1. 讀取資料
df = pd.read_csv("Shanghai_T2DM_Summary.csv")
df_processed = df.copy()

# =========================
# 2. 通用函式：判斷有無事件
def has_event(x):
    if pd.isna(x):
        return 0    
    x = str(x).strip().lower()    
    if x in ["none", "no", "nan", ""]:
        return 0  
    return 1

# =========================
# 3. 刪除 Patient Number
if "Patient Number" in df_processed.columns:
    df_processed = df_processed.drop(columns=["Patient Number"])

# =========================
# 4. Gender
df_processed["Gender_binary"] = df_processed["Gender (Female=1, Male=2)"].map({1: 0, 2: 1})
df_processed = df_processed.drop(columns=["Gender (Female=1, Male=2)"])

# =========================
# 5. Smoking History
df_processed["Smoking_binary"] = (
    pd.to_numeric(df_processed["Smoking History (pack year)"], errors="coerce")
    .fillna(0) > 0
).astype(int)
df_processed = df_processed.drop(columns=["Smoking History (pack year)"])

# =========================
# 6. Alcohol Drinking History
df_processed["Alcohol_binary"] = (
    df_processed["Alcohol Drinking History (drinker/non-drinker)"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"non-drinker": 0, "drinker": 1})
    .fillna(0)
    .astype(int)
)
df_processed = df_processed.drop(columns=["Alcohol Drinking History (drinker/non-drinker)"])

# =========================
# 7. 刪除 Type of Diabetes
if "Type of Diabetes" in df_processed.columns:
    df_processed = df_processed.drop(columns=["Type of Diabetes"])

# =========================
# 8. 刪除 Acute Diabetic Complications
if "Acute Diabetic Complications" in df_processed.columns:
    df_processed = df_processed.drop(columns=["Acute Diabetic Complications"])

# =========================
# 9. Diabetic Macrovascular Complications
df_processed["Macrovascular_Complication"] = (
    df_processed["Diabetic Macrovascular  Complications"]
    .apply(has_event)
)
df_processed = df_processed.drop(columns=["Diabetic Macrovascular  Complications"])

# =========================
# 10. Diabetic Microvascular Complications
df_processed["Microvascular_Complication"] = (
    df_processed["Diabetic Microvascular Complications"]
    .apply(has_event)
)
df_processed = df_processed.drop(columns=["Diabetic Microvascular Complications"])

# =========================
# 11. Comorbidities
df_processed["Comorbidities_binary"] = (
    df_processed["Comorbidities"]
    .apply(has_event)
)
df_processed = df_processed.drop(columns=["Comorbidities"])

# =========================
# 12. Hypoglycemic Agents
df_processed["Hypoglycemic_Agents_binary"] = (
    df_processed["Hypoglycemic Agents"]
    .apply(has_event)
)
df_processed = df_processed.drop(columns=["Hypoglycemic Agents"])

# =========================
# 13. Other Agents
df_processed["Other_Agents_binary"] = (
    df_processed["Other Agents"]
    .apply(has_event)
)
df_processed = df_processed.drop(columns=["Other Agents"])

# =========================
# 14. Hypoglycemia
df_processed["Hypoglycemia_binary"] = (
    df_processed["Hypoglycemia (yes/no)"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"no": 0, "yes": 1})
    .fillna(0)
    .astype(int)
)
df_processed = df_processed.drop(columns=["Hypoglycemia (yes/no)"])

# =========================
# 15. 輸出處理後資料
df_processed.to_csv(
    "Shanghai_T2DM_Categorical_Processed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("類別變數處理完成")
print("處理後資料維度：", df_processed.shape)
print(df_processed.head())