# CGM 時間序列預測 - 馬克夫鏈模型

## 📋 模塊概述

本模塊使用連續血糖監測 (Continuous Glucose Monitoring, CGM) 數據，建立**一階馬克夫鏈**模型與**邏輯斯迴歸**，預測患者下一個 **15 分鐘**內血糖狀態的變化。

### 核心特性
- ✅ **時間序列分析**：基於患者的歷史血糖數據
- ✅ **狀態轉移模型**：馬克夫鏈刻畫血糖狀態之間的轉移規律
- ✅ **多分類預測**：預測下一時刻是低血糖、正常還是高血糖
- ✅ **患者級交叉驗證**：防止數據洩漏，確保模型泛化性能

---

## 🩺 血糖狀態定義

| 狀態 | 代號 | 血糖範圍 (mg/dL) | 臨床含義 |
|------|------|-----------------|----------|
| 低血糖 | S0 | < 70 | 高風險，需要即時干預 |
| 目標範圍 | S1 | 70 - 180 | 正常/穩定 |
| 高血糖 | S2 | > 180 | 需要控制和監測 |

---

## 🔄 時間序列預測工作流

```
CGM 連續血糖監測數據
        ↓
   數據準備
  (清理、格式化)
        ↓
   一階馬克夫鏈
 (計算狀態轉移機率)
        ↓
   轉移矩陣分析
 (評估患者間差異)
        ↓
   邏輯斯迴歸
 (融合多個特徵)
        ↓
預測下一個 15 分鐘
   血糖狀態
```

---

## 🚀 快速開始

### 前置要求
```bash
pip install pandas numpy scikit-learn openpyxl
```

### 執行步驟

#### 步驟 1：準備 CGM 數據
```bash
cd code/06_markov_prediction/
python data_preparation.py
```

**輸入：** `Shanghai_T2DM/` 資料夾中的 Excel 檔案  
**輸出：** `combined_cgm_data.csv`

**數據要求：**
- 在項目根目錄建立 `Shanghai_T2DM/` 資料夾
- 放入患者的 Excel 檔案 (`.xlsx` 或 `.xls` 格式)
- **每個檔案的前兩欄**必須是：
  - 第 1 欄：時間戳 (datetime format)
  - 第 2 欄：CGM 血糖值 (numeric)
- 檔案名稱（不含副檔名）會自動作為患者 ID

#### 步驟 2：訓練模型
```bash
python markov_logistic_prediction.py
```

**輸入：** `combined_cgm_data.csv`  
**輸出：**
- `transition_count.csv` - 馬克夫轉移計數
- `transition_probability.csv` - 轉移機率矩陣
- `logistic_coefficients.csv` - 模型係數
- 終端輸出：混淆矩陣、分類報告、準確率

---

## 📊 模型架構

### 特徵工程

| 特徵名 | 說明 | 來源 |
|--------|------|------|
| `prev_state` | t-1 時刻的血糖狀態 (S0/S1/S2) | 馬克夫性質 |
| `prev_cgm` | t-1 時刻的 CGM 值 (mg/dL) | 一階滯後 |
| `prev_delta` | 血糖變化量 (t-1 時 CGM - t-2 時 CGM) | 趨勢信息 |

### 馬克夫鏈分析

構建**狀態轉移機率矩陣 (Transition Probability Matrix, TPM)**：

```
        S0      S1      S2
S0   [p00    p01    p02]
S1   [p10    p11    p12]
S2   [p20    p21    p22]
```

其中 pij = P(下一時刻狀態為 Sj | 當前狀態為 Si)

**解釋：**
- **對角線** (p00, p11, p22)：狀態穩定性
  - 高值表示患者傾向停留在該狀態
- **非對角元素**：狀態轉移傾向
  - p01, p12 > 0.5：患者易從低血糖升至高血糖
  - p10, p21 > 0.5：患者易發生血糖波動

### 邏輯斯迴歸預測

**模型配置：**
- **類型**：多分類邏輯斯迴歸 (Multinomial)
- **求解器**：自動選擇
- **最大迭代**：3000
- **目標**：預測 t 時刻的血糖狀態

**驗證策略：**
- **5-Fold 患者級交叉驗證** (GroupKFold)
- 按患者分組，防止同一患者的數據分散在訓練/測試集
- 確保模型能推廣到新患者

---

## 📈 輸出解釋

### 轉移機率矩陣 (`transition_probability.csv`)
```
     0      1      2
0  0.6200 0.3200 0.0600
1  0.1000 0.7500 0.1500
2  0.0300 0.4200 0.5500
```

**解讀示例：**
- 患者在低血糖 (S0) 時，下一時刻：
  - 62% 機率停留在低血糖
  - 32% 機率進入目標範圍
  - 6% 機率升高至高血糖

### 混淆矩陣 (Confusion Matrix)
```
           Predicted
           S0   S1   S2
Actual S0 [200  50   10]
       S1 [ 30 800   20]
       S2 [  5  100  300]
```

### 分類報告 (Classification Report)
```
         Precision Recall F1-Score Support
Low          0.83   0.78    0.80     260
Target       0.89   0.93    0.91     850
High         0.89   0.74    0.81     405

Accuracy                        0.86   1515
```

### 模型係數 (`logistic_coefficients.csv`)
```
              Low    Target    High
prev_state_1  -0.52   1.23    -0.71
prev_cgm      -0.08   0.02     0.06
prev_delta     0.01   0.003   -0.01
```

**解釋：**
- 正係數：增加該狀態的預測概率
- 負係數：降低該狀態的預測概率
- 絕對值：特徵的相對重要性

---

## 🔍 核心算法說明

### 一階馬克夫性質 (First-Order Markov Property)

$$P(S_t | S_{t-1}, S_{t-2}, ..., S_0) = P(S_t | S_{t-1})$$

下一時刻的狀態只取決於當前狀態，與更遠的歷史無關。

**優點：**
- 計算高效
- 參數少，易於解釋
- 對短期血糖波動捕捉良好

**限制：**
- 可能遺漏長期趨勢
- 患者個體差異大時可能欠擬合

### 患者級交叉驗證

防止**數據洩漏 (Data Leakage)**：

```
訓練集中不包含測試患者的任何數據
    ↓
模型學習從該患者群體的模式
    ↓
但預測完全陌生患者的表現
```

---

## 📋 項目結構

```
hanghai_T2DM_ML_Project/
│
├── code/
│   ├── 01_data_preprocessing/      # 臨床數據預處理 (已完成)
│   ├── 02_eda/                     # 探索性數據分析 (已完成)
│   ├── 03_baseline_models/         # 基礎模型 (已完成)
│   ├── 04_advanced_models/         # 進階模型 (已完成)
│   ├── 05_model_evaluation/        # 模型評估 (已完成)
│   │
│   └── 06_markov_prediction/       # ✨ CGM 時間序列預測 (NEW)
│       ├── data_preparation.py     # 數據準備
│       ├── markov_logistic_prediction.py  # 模型訓練
│       └── README.md               # 本檔案
│
├── data/
│   ├── raw_data/
│   │   └── Shanghai_T2DM_Summary.csv
│   └── processed_data/
│
├── output/
│   ├── model_results/
│   └── visualizations/
│
├── Shanghai_T2DM/                  # ← CGM Excel 數據 (自建)
│   ├── patient_001.xlsx
│   ├── patient_002.xlsx
│   └── ...
│
└── README.md                        # 主項目文檔
```

---

## ⚙️ 高級配置

### 調整血糖狀態閾值

編輯 `markov_logistic_prediction.py` 中的 `glucose_state()` 函數：

```python
def glucose_state(x):
    if x < 80:          # 改為 80 (更嚴格的低血糖定義)
        return 0
    elif x <= 160:      # 改為 160
        return 1
    else:
        return 2
```

### 調整交叉驗證折數

```python
cv = GroupKFold(n_splits=10)  # 改為 10-fold
```

### 更改邏輯斯迴歸求解器

```python
model = LogisticRegression(
    multi_class="multinomial",
    solver="newton-cg",      # 或 "lbfgs", "sag"
    max_iter=5000
)
```

---

## 🐛 常見問題

**Q1: 為什麼交叉驗證準確率比全數據準確率低？**

A: 這是正常現象。交叉驗證評估的是模型對**新患者**的泛化能力，因此通常低於在訓練數據上的性能。

**Q2: 能用更高階馬克夫鏈嗎？**

A: 可以。修改特徵工程部分：
```python
df["prev3_cgm"] = df.groupby("patient_id")["CGM"].shift(3)
X["prev3_cgm"] = df["prev3_cgm"]
```

**Q3: 輸出的 CSV 文件位置？**

A: 與 Python 腳本同一目錄。可改為：
```python
output_dir = "../../output/"
transition_count.to_csv(
    os.path.join(output_dir, "transition_count.csv")
)
```

---

## 📚 參考資料

- Scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Markov Chains Theory: https://en.wikipedia.org/wiki/Markov_chain
- CGM Data Analysis: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5329148/
- Pandas Time Series: https://pandas.pydata.org/docs/user_guide/timeseries.html

---

## 📝 最後更新

**日期：** 2026-07-03  
**版本：** 1.0  
**狀態：** ✅ 就緒執行

