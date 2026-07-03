# CGM 馬克夫鏈預測模型

## 概述
使用連續血糖監測 (Continuous Glucose Monitoring, CGM) 數據，建立馬克夫鏈模型預測患者下一個 15 分鐘血糖狀態。

## 血糖狀態定義
| 狀態 | 代號 | 血糖範圍 (mg/dL) |
|------|------|-----------------|
| 低血糖 | S0 | < 70 |
| 目標範圍 | S1 | 70 - 180 |
| 高血糖 | S2 | > 180 |

## 使用步驟

### 1. 數據準備
執行 `data_preparation.py` 準備 CGM 數據：
```bash
python code/data_preparation.py
```

**要求：**
- 在項目根目錄創建 `Shanghai_T2DM` 資料夾
- 放入患者的 Excel 檔案 (`.xlsx` 或 `.xls`)
- 每個檔案的前兩欄必須是：時間 (datetime) 和 CGM 值

**輸出：**
- `combined_cgm_data.csv` - 合併後的清理數據

### 2. 模型訓練與評估
執行 `markov_logistic_prediction.py` 建立預測模型：
```bash
python code/markov_logistic_prediction.py
```

**特徵��**
- `prev_state`: t-1 時刻的血糖狀態 (馬克夫特性)
- `prev_cgm`: t-1 時刻的 CGM 值
- `prev_delta`: 血糖變化量 (t-1 - t-2)

**目標：**
- 預測 t 時刻的血糖狀態

**輸出：**
- `transition_count.csv` - 馬克夫轉移計數
- `transition_probability.csv` - 馬克夫轉移機率
- `logistic_coefficients.csv` - 邏輯斯迴歸係數

## 模型架構

### 馬克夫鏈分析
計算狀態間的轉移機率：
- 從低血糖轉移到目標範圍的概率
- 從目標範圍轉移到高血糖的概率
- 各狀態的穩定性

### 邏輯斯迴歸預測
- **模型類型**：多分類邏輯斯迴歸
- **驗證方式**：5-Fold 交叉驗證 (按患者分組)
- **評估指標**：準確率、混淆矩陣、精準率/召回率

## 環境需求
```
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.22.0
openpyxl  # 讀取 Excel
```

## 檔案結構
```
hanghai_T2DM_ML_Project/
├── code/
│   ├── data_preparation.py           # 數據準備
│   ├── markov_logistic_prediction.py # 模型訓練
│   └── README.md                      # 本檔案
├── output/                            # 輸出結果
└── Shanghai_T2DM/                     # 輸入 Excel 檔案 (自建)
```

## 注意事項
- 確保 Excel 檔案格式正確，前兩欄為時間和 CGM 值
- 時間欄會自動轉換為 pandas datetime 格式
- CGM 值會自動轉換為數值，非數值會被移除
- 患者 ID 由 Excel 檔案名稱決定 (不含副檔名)
