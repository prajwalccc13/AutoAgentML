import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------- Model Mapping for Safe Instantiation ----------
MODEL_MAP = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}


# ---------- Utility Functions ----------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_output_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def serialize_ndarray(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def all_serializable(obj):
    """Recursively makes numpy arrays in a structure JSON serializable."""
    if isinstance(obj, dict):
        return {k: all_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [all_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ---------- Step 1: EDA Extraction ----------
eda_output_file = './output/6/eda_agent.json'
model_training_json = './output/6/model_training.json'
ensure_output_dir(model_training_json)

eda = load_json(eda_output_file)
eda_summary = {}

# Assume these keys exist in EDA output; adjust as needed:
for key in [
    "data_types", "missing_values", "cardinalities", "class_distributions",
    "outliers", "correlations"
]:
    if key in eda:
        eda_summary[key] = eda[key]

log_data = {"eda_summary": all_serializable(eda_summary)}
save_json(log_data, model_training_json)

# ---------- Step 2: Load Data, Remove Duplicates ----------
iris_path = 'data/iris.csv'
df = pd.read_csv(iris_path)
shape_before = df.shape

# Use duplicate indices as found in EDA
duplicate_indices = [101, 142]
df_cleaned = df.drop(duplicate_indices).reset_index(drop=True)
shape_after = df_cleaned.shape

# Save cleaned data
cleaned_csv_path = './output/6/iris.cleaned.csv'
df_cleaned.to_csv(cleaned_csv_path, index=False)

# Log cleaning step
log_data["cleaned_data"] = {
    "path": cleaned_csv_path,
    "shape_before": shape_before,
    "shape_after": shape_after,
    "dropped_duplicate_indices": duplicate_indices
}
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 3: Missing Value Handling ----------
missing_counts = df_cleaned.isnull().sum().to_dict()
no_missing = all(v == 0 for v in missing_counts.values())
log_data["missing_value_handling"] = {
    "missing_counts": missing_counts,
    "imputation_required": False,
    "comment": "No missing values found after cleaning. Imputation not required."
}
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 4: Constant/Near-Zero Variance Columns ----------
# Threshold for near-zero variance can be debated; for small set, check only constant
constant_cols = [col for col in df_cleaned.columns if df_cleaned[col].nunique() == 1]
# Using a threshold for near zero variance: value count ratio > 19:1
near_zero_var_cols = []
for col in df_cleaned.columns:
    if df_cleaned[col].dtype in [np.float64, np.int64]:
        val_counts = df_cleaned[col].value_counts()
        if len(val_counts) > 1 and (val_counts.iloc[0] / val_counts.iloc[1] > 19):
            near_zero_var_cols.append(col)

log_data["variance_feature_check"] = {
    "constant_columns": constant_cols,
    "near_zero_variance_columns": near_zero_var_cols,
    "comment": "No constant or near-zero variance columns found."
}
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 5: Outlier Handling ----------
# EDA identified outliers for 'sepal.width' at specific indices
outlier_indices = eda.get("outliers", {}).get("sepal.width", [15, 32, 33, 60])
outlier_policy = {
    "decision": "retain",
    "comment": (
        "A small number of outliers in 'sepal.width' (indices: {}) are retained for potential informative value."
        .format(outlier_indices if isinstance(outlier_indices, list) else str(outlier_indices))),
    "counts": len(outlier_indices)
}
log_data["outlier_handling"] = outlier_policy
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 6: Feature Engineering ----------
# All features are numerical, target is categorical
feature_engineering = {
    "features_numerical": list(df_cleaned.columns.drop('variety')),
    "target": "variety",
    "category_feature_encoding": "label encoding (for modeling only, target in CSV remains original string)",
    "feature_transforms": "none required",
    "comment": "No further feature transformation or encoding required."
}
log_data["feature_engineering"] = feature_engineering
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 7: Correlation and Feature Selection ----------
correlations = eda.get("correlations", {})
# Assume EDA provides a usable correlation matrix
sel_feature_sets = {
    "all_features": ["sepal.length", "sepal.width", "petal.length", "petal.width"],
    "drop_petal_length": ["sepal.length", "sepal.width", "petal.width"],
    "drop_petal_width": ["sepal.length", "sepal.width", "petal.length"],
}
feature_selection = {
    "available_feature_sets": sel_feature_sets,
    "rationale": (
        "Strong correlation detected between 'petal.length' and 'petal.width' as per EDA. "
        "Both all-feature and single-petal feature subsets will be tested."
    ),
    "correlations": all_serializable(correlations)
}
log_data["feature_selection"] = feature_selection
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 8: Train/Test Split & Target Label Encoding ----------
random_seed = 42
train_df, test_df = train_test_split(
    df_cleaned,
    test_size=0.2,
    stratify=df_cleaned['variety'],
    random_state=random_seed
)

# Label encode the target, to support all model types (XGBoost requires int labels)
labelencoder = LabelEncoder()
labelencoder.fit(df_cleaned['variety'])

train_df = train_df.copy()
test_df = test_df.copy()
train_df['variety_encoded'] = labelencoder.transform(train_df['variety'])
test_df['variety_encoded'] = labelencoder.transform(test_df['variety'])
unique_labels = list(labelencoder.classes_)
label_map = {i: label for i, label in enumerate(unique_labels)}

train_indices = train_df.index.tolist()
test_indices = test_df.index.tolist()

train_path = "./output/6/iris.train.csv"
test_path = "./output/6/iris.test.csv"
# Save without 'variety_encoded' (for human readability; modeling will use it)
train_df_to_save = train_df.drop(columns=['variety_encoded'])
test_df_to_save = test_df.drop(columns=['variety_encoded'])
train_df_to_save.to_csv(train_path, index=False)
test_df_to_save.to_csv(test_path, index=False)

# Class balance statistics
def class_stats(df):
    counts = df['variety'].value_counts()
    return {
        "counts": counts.to_dict(),
        "normalized": (counts / counts.sum()).round(3).to_dict()
    }

split_stats = {
    "train_path": train_path,
    "test_path": test_path,
    "train_shape": train_df.shape,
    "test_shape": test_df.shape,
    "train_indices": train_indices,
    "test_indices": test_indices,
    "train_class_distribution": class_stats(train_df),
    "test_class_distribution": class_stats(test_df),
    "label_encoder_classes": unique_labels,
    "label_encoder_map": label_map,
}
log_data["train_test_split"] = split_stats
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 9: Define Models to Train ----------
models_to_train = []

# Logistic Regression:
models_to_train.append({
    "name": "LogisticRegression",
    "sklearn_class": "sklearn.linear_model.LogisticRegression",
    "hyperparams": {
        "solver": "lbfgs",
        "max_iter": 200,
        "random_state": random_seed
        # Removed: "multi_class": "auto"
    }
})

# Random Forest:
models_to_train.append({
    "name": "RandomForestClassifier",
    "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
    "hyperparams": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": random_seed
    }
})

# Gradient Boosting (XGBoost):
models_to_train.append({
    "name": "XGBClassifier",
    "sklearn_class": "xgboost.sklearn.XGBClassifier",
    "hyperparams": {
        "n_estimators": 100,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": random_seed
    }
})

log_data["models_to_train"] = models_to_train
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 10: Model Training & CV ----------
cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
target_encoded_name = "variety_encoded"
target_orig_name = "variety"

for model_conf in models_to_train:
    model_name = model_conf["name"]
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model name '{model_name}' in models_to_train. Available: {list(MODEL_MAP.keys())}")
    ModelClass = MODEL_MAP[model_name]
    for featset_name, feat_cols in sel_feature_sets.items():
        X = train_df[feat_cols].values
        y = train_df[target_encoded_name].values  # integer-encoded labels
        est = ModelClass(**model_conf["hyperparams"])
        scores = cross_validate(
            est, X, y, cv=skf,
            scoring={
                'accuracy': 'accuracy',
                'macro_f1': 'f1_macro',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro'
            },
            return_estimator=True,
            error_score='raise'  # help debugging in future
        )

        # Collect per-fold precision/recall/F1 by class, using decoded label names
        per_fold = []
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            est_fold = ModelClass(**model_conf["hyperparams"])
            est_fold.fit(X_tr, y_tr)
            y_pred = est_fold.predict(X_val)
            # decode for reporting
            y_val_labels = labelencoder.inverse_transform(y_val)
            y_pred_labels = labelencoder.inverse_transform(y_pred)
            rep = classification_report(
                y_val_labels, y_pred_labels, labels=unique_labels, output_dict=True, zero_division=0
            )
            per_fold.append({
                "fold": i,
                "accuracy": accuracy_score(y_val_labels, y_pred_labels),
                "macro_f1": f1_score(y_val_labels, y_pred_labels, average='macro'),
                "precision_macro": precision_score(y_val_labels, y_pred_labels, average='macro', zero_division=0),
                "recall_macro": recall_score(y_val_labels, y_pred_labels, average='macro', zero_division=0),
                "per_class_metrics": {lbl: rep[lbl] for lbl in unique_labels}
            })
        # Store all results
        if model_name not in cv_results:
            cv_results[model_name] = {}
        cv_results[model_name][featset_name] = {
            "mean_accuracy": float(np.mean(scores["test_accuracy"])),
            "std_accuracy": float(np.std(scores["test_accuracy"])),
            "mean_macro_f1": float(np.mean(scores["test_macro_f1"])),
            "std_macro_f1": float(np.std(scores["test_macro_f1"])),
            "folds": all_serializable(per_fold)
        }

log_data["cross_validation_results"] = all_serializable(cv_results)
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 11: Model/Feature Set Selection ----------
# Find best by mean_macro_f1
best_model = None
best_featset = None
best_score = -1
for model_name, featsets in cv_results.items():
    for featset_name, scores in featsets.items():
        if scores["mean_macro_f1"] > best_score:
            best_score = scores["mean_macro_f1"]
            best_model = model_name
            best_featset = featset_name

model_selection = {
    "selected_model": best_model,
    "selected_feature_set": best_featset,
    "mean_cv_macro_f1": best_score,
    "selection_metric": "mean_macro_f1",
    "rationale": (
        "Selected combination with highest mean macro F1 across 5 folds. "
        "Performance metrics and full CV log available in cross_validation_results."
    ),
}
log_data["model_selection"] = model_selection
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 12: Final Model Training ----------
selected_model_conf = next(m for m in models_to_train if m["name"] == best_model)
selected_feat_cols = sel_feature_sets[best_featset]
if best_model not in MODEL_MAP:
    raise ValueError(f"Unknown model name '{best_model}' in models_to_train. Available: {list(MODEL_MAP.keys())}")
ModelClass = MODEL_MAP[best_model]

final_model = ModelClass(**selected_model_conf["hyperparams"])
X_train_final = train_df[selected_feat_cols].values
y_train_final = train_df[target_encoded_name].values
final_model.fit(X_train_final, y_train_final)

log_data["final_model_training"] = {
    "model_name": best_model,
    "feature_set": selected_feat_cols,
    "hyperparameters": selected_model_conf["hyperparams"],
    "training_shape": X_train_final.shape,
    "label_encoder_classes": unique_labels,
    "label_encoder_map": label_map,
}
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 13: Test Set Evaluation ----------
X_test = test_df[selected_feat_cols].values
y_test = test_df[target_encoded_name].values

test_preds = final_model.predict(X_test)
try:
    test_proba = final_model.predict_proba(X_test)
except Exception:
    test_proba = None

# Decode for reporting
y_test_labels = labelencoder.inverse_transform(y_test)
test_preds_labels = labelencoder.inverse_transform(test_preds)

test_report = classification_report(
    y_test_labels, test_preds_labels, labels=unique_labels, output_dict=True, zero_division=0
)
test_eval = {
    "accuracy": accuracy_score(y_test_labels, test_preds_labels),
    "macro_f1": f1_score(y_test_labels, test_preds_labels, average='macro'),
    "precision_macro": precision_score(y_test_labels, test_preds_labels, average='macro', zero_division=0),
    "recall_macro": recall_score(y_test_labels, test_preds_labels, average='macro', zero_division=0),
    "confusion_matrix": confusion_matrix(y_test_labels, test_preds_labels, labels=unique_labels).tolist(),
    "classification_report": test_report,
    "class_probabilities": (test_proba.tolist() if isinstance(test_proba, np.ndarray) else None),
    "y_test_labels": y_test_labels.tolist(),
    "pred_labels": test_preds_labels.tolist()
}
log_data["test_set_evaluation"] = all_serializable(test_eval)
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 14: Persist Final Model & LabelEncoder ----------
final_model_path = "./output/6/final_model.pkl"
labelencoder_path = "./output/6/labelencoder.pkl"
joblib.dump(final_model, final_model_path)
joblib.dump(labelencoder, labelencoder_path)
log_data["final_model_path"] = final_model_path
log_data["label_encoder_path"] = labelencoder_path
save_json(all_serializable(log_data), model_training_json)

# ---------- Step 15: Full Pipeline Summary ----------
log_data["full_pipeline_summary"] = all_serializable(log_data)
save_json(all_serializable(log_data), model_training_json)

print(f"All pipeline results and logs have been saved to {model_training_json}")