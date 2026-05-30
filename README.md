# Shanghai Type 2 Diabetes Mellitus (T2DM) Machine Learning Analysis

## 📋 Project Overview

This project implements comprehensive machine learning analysis on Shanghai Type 2 Diabetes Mellitus patient data. The analysis includes data preprocessing, exploratory data analysis (EDA), baseline models, and advanced machine learning models with rigorous evaluation.

**Objectives:**
- Predict diabetic complications (macrovascular and microvascular)
- Predict hypoglycemia occurrence
- Identify key risk factors through feature importance analysis
- Compare model performance between traditional and advanced ML approaches

---

## 📁 Project Structure

```
Shanghai_T2DM_ML_Project/
│
├── data/
│   ├── raw_data/
│   │   └── Shanghai_T2DM_Summary.csv          # Original patient data
│   │
│   └── processed_data/
│       ├── Shanghai_T2DM_Categorical_Processed.csv
│       ├── Shanghai_T2DM_Median_Imputed.csv
│       ├── Shanghai_T2DM_Log_Transformed.csv
│       ├── Shanghai_T2DM_Log_Interaction.csv
│       └── Shanghai_T2DM_Standardized.csv     # Final input for models
│
├── code/
│   ├── 01_data_preprocessing/
│   │   ├── 01_categorical_processing.py       # Binary encoding
│   │   ├── 02_missing_value_imputation.py    # Median imputation
│   │   ├── 03_log_transformation.py           # Log transformation for skewed features
│   │   ├── 04_interaction_terms.py            # Feature interaction creation
│   │   └── 05_standardization.py              # Z-score standardization
│   │
│   ├── 02_eda/
│   │   └── eda_analysis.py                    # Exploratory Data Analysis
│   │
│   ├── 03_baseline_models/
│   │   ├── logistic_regression.py             # Logistic Regression baseline
│   │   └── linear_regression.py               # Linear Regression for continuous targets
│   │
│   ├── 04_advanced_models/
│   │   ├── xgboost_classifier.py              # XGBoost for classification
│   │   ├── xgboost_regressor.py               # XGBoost for regression
│   │   └── clustering_analysis.py             # KMeans & GMM clustering
│   │
│   └── 05_model_evaluation/
│       └── model_comparison.py                # Consolidated evaluation
│
├── output/
│   ├── eda_results/
│   │   ├── EDA_Summary.csv
│   │   └── eda_visualizations/
│   │
│   ├── model_results/
│   │   ├── Logistic_Regression_Results.csv
│   │   ├── XGBoost_Results.csv
│   │   ├── Regression_Model_Results.csv
│   │   └── clustering_results/
│   │
│   └── feature_importance/
│       └── feature_analysis.csv
│
└── README.md                                   # This file
```

---

## 🔄 Data Processing Pipeline

### Step 1: Categorical Data Processing (`01_categorical_processing.py`)
**Input:** `Shanghai_T2DM_Summary.csv`
**Output:** `Shanghai_T2DM_Categorical_Processed.csv`

**Transformations:**
- Gender: 1/2 → 0/1 (Female/Male binary)
- Smoking History: Pack-year → Binary (0/1)
- Alcohol Drinking: Categorical → Binary (0/1)
- Macrovascular Complications: Event detection → Binary (0/1)
- Microvascular Complications: Event detection → Binary (0/1)
- Comorbidities: Event detection → Binary (0/1)
- Hypoglycemic Agents: Availability → Binary (0/1)
- Other Agents: Availability → Binary (0/1)
- Hypoglycemia: Yes/No → Binary (0/1)

**Removed columns:** Patient Number, Type of Diabetes, Acute Diabetic Complications

---

### Step 2: Missing Value Imputation (`02_missing_value_imputation.py`)
**Input:** `Shanghai_T2DM_Categorical_Processed.csv`
**Output:** `Shanghai_T2DM_Median_Imputed.csv`

**Method:** Median imputation for numeric columns
- Handles missing values while preserving data distribution
- Suitable for continuous variables

---

### Step 3: Log Transformation (`03_log_transformation.py`)
**Input:** `Shanghai_T2DM_Median_Imputed.csv`
**Output:** `Shanghai_T2DM_Log_Transformed.csv`

**Rationale:** Reduce skewness in right-skewed distributions

**Log-transformed variables:**
- HbA1c (mmol/mol)
- Fasting Plasma Glucose (mg/dl)
- Triglyceride (mmol/L)
- HDL Cholesterol (mmol/L)
- LDL Cholesterol (mmol/L)
- Creatinine (umol/L)
- eGFR (ml/min/1.73m²)

**Excluded from log transformation:**
- Binary variables
- Body size variables (Age, Height, Weight, BMI)

---

### Step 4: Feature Interaction Terms (`04_interaction_terms.py`)
**Input:** `Shanghai_T2DM_Log_Transformed.csv`
**Output:** `Shanghai_T2DM_Log_Interaction.csv`

**Interaction terms created:**
1. `Interaction_Age_BMI`: Age × BMI
2. `Interaction_HbA1c_Duration`: log(HbA1c) × log(Duration)
3. `Interaction_FPG_TG`: log(FPG) × log(Triglyceride)
4. `Interaction_BMI_Smoking`: BMI × Smoking status

**Rationale:** Capture non-additive effects and synergistic relationships

---

### Step 5: Standardization (`05_standardization.py`)
**Input:** `Shanghai_T2DM_Log_Interaction.csv`
**Output:** `Shanghai_T2DM_Standardized.csv`

**Method:** Z-score standardization (μ=0, σ=1)

**Formula:** z = (x - μ) / σ

**Features standardized:**
- All numeric continuous variables
- Interaction terms
- Log-transformed features

**Features NOT standardized:**
- Binary variables (already 0/1 scale)
- Target variables

**Output:** Original features + z-score versions (prefixed with `z_`)

---

## 📊 Exploratory Data Analysis (EDA)

**Script:** `02_eda/eda_analysis.py`
**Output:** `output/eda_results/EDA_Summary.csv`

**Analysis includes:**
- Descriptive statistics (Mean, SD)
- Missing value assessment
- Skewness analysis
- Data distribution classification
  - Approximately Normal: |Skewness| < 0.5
  - Moderately Skewed: 0.5 ≤ |Skewness| < 1.0
  - Highly Skewed: |Skewness| ≥ 1.0

---

## 🧠 Machine Learning Models

### Classification Models (Predict Complications & Hypoglycemia)

#### Baseline: Logistic Regression
**Script:** `03_baseline_models/logistic_regression.py`
**Output:** `output/model_results/Logistic_Regression_Results.csv`

**Target Variables:**
1. Macrovascular_Complication
2. Microvascular_Complication
3. Hypoglycemia_binary

**Configuration:**
- Solver: liblinear
- Class Weight: balanced (handles class imbalance)
- Max Iterations: 1000
- Cross-Validation: 5-fold Stratified

**Evaluation Metrics:**
- Accuracy (mean ± SD)
- Precision (mean ± SD)
- Recall (mean ± SD)
- F1-Score (mean ± SD)
- AUC-ROC (mean ± SD)
- Class distribution (Positive/Negative counts)

---

#### Advanced: XGBoost Classifier
**Script:** `04_advanced_models/xgboost_classifier.py`
**Output:** `output/model_results/XGBoost_Results.csv`

**Target Variables:** Same as Logistic Regression

**Hyperparameters:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=auto_calculated,  # Adjusts for class imbalance
    eval_metric="logloss",
    random_state=42
)
```

**Class Imbalance Handling:**
- `scale_pos_weight` = (negative_count) / (positive_count)
- Automatically adjusted per target variable

**Evaluation Metrics:** Same as Logistic Regression

---

### Regression Models (Predict HbA1c levels)

#### Baseline: Linear Regression
**Script:** `03_baseline_models/linear_regression.py`
**Output:** `output/model_results/Regression_Model_Results.csv` (rows 1-2)

**Target Variable:** HbA1c (mmol/mol)

**Configuration:**
- Cross-Validation: 5-fold KFold
- Data Leakage Prevention: All HbA1c-related features excluded

**Evaluation Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error) %
- R² Score

---

#### Advanced: XGBoost Regressor
**Script:** `04_advanced_models/xgboost_regressor.py`
**Output:** `output/model_results/Regression_Model_Results.csv` (rows 3-4)

**Target Variable:** HbA1c (mmol/mol)

**Hyperparameters:**
```python
XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Evaluation Metrics:** Same as Linear Regression

---

### Clustering Analysis

**Script:** `04_advanced_models/clustering_analysis.py`
**Output:** `output/model_results/clustering_results/`

#### K-Means Clustering
**Configuration:**
- n_clusters: 2
- Evaluation: Silhouette Score, Chi-square test

#### Gaussian Mixture Model (GMM)
**Configuration:**
- n_components: 3
- Evaluation: Silhouette Score, Chi-square test

**Analysis:**
- Cluster characteristics (Age, BMI, HbA1c, Duration means)
- Complication distribution by cluster
- Statistical significance testing
- PCA visualization (2D projection)

---

## 📈 Model Evaluation & Comparison

### Cross-Validation Strategy
- **Classification:** 5-fold Stratified KFold (preserves class distribution)
- **Regression:** 5-fold KFold
- **Random State:** 42 (reproducibility)

### Output Files

**Classification Results:**
```csv
Target,Accuracy_mean,Accuracy_sd,Precision_mean,Precision_sd,...,Positive_Count,Negative_Count
Macrovascular_Complication,0.XXX,0.XXX,...
Microvascular_Complication,0.XXX,0.XXX,...
Hypoglycemia_binary,0.XXX,0.XXX,...
```

**Regression Results:**
```csv
Model,MAE,RMSE,MAPE,R2
Linear Regression,X.XXX,X.XXX,X.XXX,X.XXX
XGBoost Regressor,X.XXX,X.XXX,X.XXX,X.XXX
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib
```

### Execution Order

**1. Data Preprocessing (One-time setup)**
```bash
cd code/01_data_preprocessing/
python 01_categorical_processing.py
python 02_missing_value_imputation.py
python 03_log_transformation.py
python 04_interaction_terms.py
python 05_standardization.py
```

**2. EDA**
```bash
cd code/02_eda/
python eda_analysis.py
```

**3. Baseline Models**
```bash
cd code/03_baseline_models/
python logistic_regression.py
python linear_regression.py
```

**4. Advanced Models**
```bash
cd code/04_advanced_models/
python xgboost_classifier.py
python xgboost_regressor.py
python clustering_analysis.py
```

**5. Model Comparison**
```bash
cd code/05_model_evaluation/
python model_comparison.py
```

---

## 📋 Input/Output Data Dictionary

### Raw Features (Original Data)
- **Age (years)**: Patient age
- **Height (cm)**: Patient height
- **Weight (kg)**: Patient weight
- **BMI (kg/m²)**: Body Mass Index
- **HbA1c (mmol/mol)**: Glycated hemoglobin
- **Fasting Plasma Glucose (mg/dl)**: FPG level
- **Duration of diabetes (years)**: Disease duration
- **Triglyceride (mmol/L)**: Triglyceride level
- **HDL Cholesterol (mmol/L)**: High-density lipoprotein
- **LDL Cholesterol (mmol/L)**: Low-density lipoprotein
- **Creatinine (umol/L)**: Kidney function marker
- **eGFR (ml/min/1.73m²)**: Estimated glomerular filtration rate

### Binary Features
- **Gender_binary**: 0=Female, 1=Male
- **Smoking_binary**: 0=Never, 1=Ever
- **Alcohol_binary**: 0=Non-drinker, 1=Drinker
- **Comorbidities_binary**: 0=No, 1=Yes
- **Hypoglycemic_Agents_binary**: 0=No, 1=Yes
- **Other_Agents_binary**: 0=No, 1=Yes

### Target Variables
- **Macrovascular_Complication**: 0=No, 1=Yes (Coronary, cerebral, peripheral events)
- **Microvascular_Complication**: 0=No, 1=Yes (Retinopathy, nephropathy, neuropathy)
- **Hypoglycemia_binary**: 0=No, 1=Yes
- **HbA1c (mmol/mol)**: Continuous outcome (regression target)

### Processed Features
- **log_[feature]**: Log-transformed continuous variables
- **z_[feature]**: Standardized (z-score normalized) features
- **Interaction_[A]_[B]**: Feature interaction terms

---

## 📊 Key Findings

(To be updated after model runs)

- Model performance comparison (Classification vs Regression)
- Most important features for each target
- Optimal hyperparameter configurations
- Clustering insights

---

## 👥 Team Information

**Project Lead:** itzuakatsuki  
**Analysis Date:** 2026-05-30

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Shanghai T2DM Study Data

---

## 📝 Notes

- All results are reproducible (random_state=42)
- Missing values handled via median imputation
- Class imbalance addressed through `scale_pos_weight` in XGBoost
- Feature standardization applied except for binary variables
- Cross-validation ensures robust model evaluation

---

**Last Updated:** 2026-05-30
