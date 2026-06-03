# 修復目錄結構 - 步驟指南

## 問題
```
code/
├── 04_advanced_models/      ← 空目錄 (只有 .gitkeep)
└── 04_main_models/          ← 舊目錄 (包含實際文件)
    ├── knnvsgmm.py
    └── xgboost_classifier.py
```

## 解決方案 A - 本地操作步驟

### 步驟 1: 進入本地倉庫
```bash
cd hanghai_T2DM_ML_Project
```

### 步驟 2: 備份並移動文件
```bash
# 備份舊目錄中的文件
cp code/04_main_models/knnvsgmm.py code/04_advanced_models/
cp code/04_main_models/xgboost_classifier.py code/04_advanced_models/

# 或者整個目錄移動
rm -rf code/04_advanced_models/.gitkeep
mv code/04_main_models/* code/04_advanced_models/
rmdir code/04_main_models
```

### 步驟 3: 重新命名 knnvsgmm.py (可選)
```bash
# 如果想要更清晰的命名
mv code/04_advanced_models/knnvsgmm.py code/04_advanced_models/clustering_analysis.py
```

### 步驟 4: 提交到 Git
```bash
git add .
git commit -m "Fix: Rename 04_main_models to 04_advanced_models for consistent naming"
git push origin main
```

## 最終結構
```
code/
├── 01_data_preprocessing/
│   ├── 01_categorical_processing.py
│   ├── 02_missing_value_imputation.py
│   ├── 03_log_transformation.py
│   ├── 04_interaction_terms.py
│   └── 05_standardization.py
├── 02_eda/
│   └── eda_analysis.py
├── 03_baseline_models/
│   ├── logistic_regression.py
│   └── linear_regression.py
├── 04_advanced_models/
│   ├── xgboost_classifier.py
│   ├── xgboost_regressor.py
│   ├── clustering_analysis.py (原 knnvsgmm.py)
│   └── knnvsgmm.py (如果保留舊文件)
└── 05_model_evaluation/
    └── visualize_results.py
```

---

**完成後，請告訴我，我會更新 README！**
