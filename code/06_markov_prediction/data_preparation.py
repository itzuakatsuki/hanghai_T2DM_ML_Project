"""
CGM 數據準備模塊
讀取 Shanghai_T2DM 資料夾中的所有 Excel 檔案
合併、清理、格式化 CGM 數據
輸出：combined_cgm_data.csv
"""

import pandas as pd
import glob
import os

# 1. 放你所有 CGM Excel 檔案的資料夾
data_path = "Shanghai_T2DM"   # 改成你的資料夾名稱

files = glob.glob(os.path.join(data_path, "*.xlsx")) + \
        glob.glob(os.path.join(data_path, "*.xls"))

all_data = []

for file in files:
    print("讀取：", file)

    df = pd.read_excel(file)

    # 只取前兩欄：時間、CGM血糖值
    temp = df.iloc[:, 0:2].copy()
    temp.columns = ["datetime", "CGM"]

    # 用檔名當病人ID
    temp["patient_id"] = os.path.basename(file).split(".")[0]

    all_data.append(temp)

# 2. 合併全部檔案
combined_cgm_data = pd.concat(all_data, ignore_index=True)

# 3. 時間與CGM轉格式
combined_cgm_data["datetime"] = pd.to_datetime(
    combined_cgm_data["datetime"],
    errors="coerce"
)

combined_cgm_data["CGM"] = pd.to_numeric(
    combined_cgm_data["CGM"],
    errors="coerce"
)

# 4. 刪除缺失值
combined_cgm_data = combined_cgm_data.dropna(
    subset=["datetime", "CGM"]
)

# 5. 排序
combined_cgm_data = combined_cgm_data.sort_values(
    ["patient_id", "datetime"]
)

# 6. 輸出
combined_cgm_data.to_csv(
    "combined_cgm_data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("完成：combined_cgm_data.csv")
print(combined_cgm_data.head())
print(combined_cgm_data.shape)
