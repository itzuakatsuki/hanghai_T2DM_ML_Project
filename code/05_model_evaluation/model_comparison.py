# 05_model_evaluation/model_evaluation.py

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from xgboost import XGBClassifier, XGBRegressor


# =========================
# 1. Load Data
# =========================

df = pd.read_csv("../../data/processed_data.csv")


# =========================
# 2. Classification Evaluation
# =========================

target_cols = [
    "Macrovascular_Complication",
    "Microvascular_Complication",
    "Hypoglycemia_binary"
]

z_cols = [col for col in df.columns if col.startswith("z_")]

binary_cols = [
    "Gender_binary",
    "Smoking_binary",
    "Alcohol_binary",
    "Comorbidities_binary",
    "Hypoglycemic_Agents_binary",
    "Other_Agents_binary"
]

feature_cols = z_cols + binary_cols

X_class = df[feature_cols].copy()
X_class = X_class.apply(pd.to_numeric, errors="coerce")
X_class = X_class.fillna(X_class.median())

cv_class = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring_class = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc"
}

classification_results = []

for target in target_cols:

    y = df[target].astype(int)

    if y.nunique() < 2:
        continue

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    for model_name, model in models.items():

        scores = cross_validate(
            model,
            X_class,
            y,
            cv=cv_class,
            scoring=scoring_class,
            return_train_score=False,
            error_score=np.nan
        )

        classification_results.append({
            "Model": model_name,
            "Target": target,
            "Accuracy_mean": np.nanmean(scores["test_accuracy"]),
            "Accuracy_sd": np.nanstd(scores["test_accuracy"]),
            "Precision_mean": np.nanmean(scores["test_precision"]),
            "Precision_sd": np.nanstd(scores["test_precision"]),
            "Recall_mean": np.nanmean(scores["test_recall"]),
            "Recall_sd": np.nanstd(scores["test_recall"]),
            "F1_mean": np.nanmean(scores["test_f1"]),
            "F1_sd": np.nanstd(scores["test_f1"]),
            "AUC_mean": np.nanmean(scores["test_auc"]),
            "AUC_sd": np.nanstd(scores["test_auc"]),
            "Positive_Count": int(y.sum()),
            "Negative_Count": int((y == 0).sum())
        })

classification_df = pd.DataFrame(classification_results)

classification_df.to_csv(
    "../../output/classification_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nClassification Results")
print(classification_df)


# =========================
# 3. Regression Evaluation
# =========================

target_reg = "HbA1c (mmol/mol)"
y_reg = df[target_reg]

leakage_cols = [
    "HbA1c (mmol/mol)",
    "z_HbA1c (mmol/mol)",
    "log_HbA1c (mmol/mol)",
    "z_log_HbA1c (mmol/mol)",
    "Interaction_HbA1c_Duration",
    "z_Interaction_HbA1c_Duration"
]

reg_feature_cols = [
    col for col in feature_cols
    if col not in leakage_cols
]

X_reg = df[reg_feature_cols].copy()
X_reg = X_reg.apply(pd.to_numeric, errors="coerce")
X_reg = X_reg.fillna(X_reg.median())

cv_reg = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

regression_results = []

for model_name, model in regression_models.items():

    pred = cross_val_predict(
        model,
        X_reg,
        y_reg,
        cv=cv_reg
    )

    mae = mean_absolute_error(y_reg, pred)
    rmse = np.sqrt(mean_squared_error(y_reg, pred))
    mape = mean_absolute_percentage_error(y_reg, pred) * 100
    r2 = r2_score(y_reg, pred)

    regression_results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    })

regression_df = pd.DataFrame(regression_results)

regression_df.to_csv(
    "../../output/regression_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRegression Results")
print(regression_df)


# =========================
# 4. Combined Summary
# =========================

with open("../../output/model_evaluation_summary.txt", "w", encoding="utf-8") as f:

    f.write("Model Evaluation & Comparison\n")
    f.write("=" * 40 + "\n\n")

    f.write("Cross-Validation Strategy\n")
    f.write("- Classification: 5-fold Stratified KFold\n")
    f.write("- Regression: 5-fold KFold\n")
    f.write("- Random State: 42\n\n")

    f.write("Classification Results\n")
    f.write(classification_df.to_string(index=False))
    f.write("\n\n")

    f.write("Regression Results\n")
    f.write(regression_df.to_string(index=False))
    f.write("\n\n")

print("\nOutput files created:")
print("../../output/classification_results.csv")
print("../../output/regression_results.csv")
print("../../output/model_evaluation_summary.txt")
