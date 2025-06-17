import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support, confusion_matrix
import joblib
import time

# Try to import optional models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

def to_serializable(obj):
    if isinstance(obj, dict):
        return {to_serializable(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Index):
        return [to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, pd.Series):
        return [to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

# -------------- FILE PATHS ---------------
os.makedirs("./output/id_0/", exist_ok=True)
EDA_JSON = './output/id_0/eda_agent.json'
MODEL_JSON = './output/id_0/model_training.json'
RAW_DATA = 'data/iris.csv'
CLEANED_CSV = './output/id_0/iris_cleaned.csv'
OUTL_INCLUDED_CSV = './output/id_0/iris_outliers_included.csv'
OUTL_REMOVED_CSV = './output/id_0/iris_outliers_removed.csv'
TRAIN_CSV = './output/id_0/iris_train.csv'
TEST_CSV = './output/id_0/iris_test.csv'
FINAL_MODEL = './output/id_0/final_model.joblib'

# --------- LOGGING UTILS -------------
final_logs = []  # Accumulate to dump once at end

def log_json(log_dict):
    """Accumulate log dictionary to final_logs."""
    final_logs.append(log_dict)

def flush_logs():
    """Write all accumulated logs to MODEL_JSON."""
    with open(MODEL_JSON, 'w') as f:
        json.dump(to_serializable(final_logs), f, indent=2)

# ----------------------------------------------------
# 1. Load EDA output and log preprocessing insights
# ----------------------------------------------------
with open(EDA_JSON, 'r') as f:
    eda_output = json.load(f)

preprocessing_insights = {
    'data_types': eda_output.get('data_types', {}),
    'missing_values': eda_output.get('missing_values', {}),
    'outlier_indices': eda_output.get('outliers', {}).get('sepal.width', []),
    'duplicate_indices': eda_output.get('duplicates', []),
    'class_balance': eda_output.get('class_balance', {}),
    'feature_correlations': eda_output.get('feature_correlations', {}),
    'feature_summary': eda_output.get('feature_summary', {}),
}
log_json({"preprocessing_insights": preprocessing_insights})

# ----------------------------------------------------
# 2. Load and clean data
# ----------------------------------------------------
iris_df = pd.read_csv(RAW_DATA)
duplicate_indices = [101, 142]
cleaned_df = iris_df.drop(index=duplicate_indices, errors="ignore").reset_index(drop=True)
cleaned_df.to_csv(CLEANED_CSV, index=False)
log_json({
    "duplicate_removal": {
        "cleaned_dataset_path": CLEANED_CSV,
        "number_of_rows_after_removal": int(len(cleaned_df)),
        "indices_removed": [int(x) for x in duplicate_indices],
    }
})

# ----------------------------------------------------
# 3. Check missing values
# ----------------------------------------------------
missing_counts = preprocessing_insights['missing_values']
missing_imputation_required = not all((v == 0 for v in missing_counts.values()))
log_json({
    "missing_value_check": {
        "missing_value_counts": to_serializable(missing_counts),
        "imputation_required": bool(missing_imputation_required),
        "action": "No imputation required as there are no missing values." if not missing_imputation_required else "Check columns for imputation."
    }
})

# ----------------------------------------------------
# 4. Outlier handling: Save with and without outliers
# ----------------------------------------------------
outlier_indices = [int(i) for i in preprocessing_insights['outlier_indices']]
iris_with_outliers = cleaned_df.copy()
iris_with_outliers.to_csv(OUTL_INCLUDED_CSV, index=False)
# Outliers refer to original indices, but after removing duplicates and resetting, need to check their existence
safe_outlier_indices = [i for i in outlier_indices if i < len(cleaned_df)]
iris_without_outliers = cleaned_df.drop(index=safe_outlier_indices, errors="ignore").reset_index(drop=True)
iris_without_outliers.to_csv(OUTL_REMOVED_CSV, index=False)
log_json({
    "outlier_detection": {
        "outlier_column": "sepal.width",
        "outlier_indices": safe_outlier_indices,
        "with_outliers_dataset_path": OUTL_INCLUDED_CSV,
        "without_outliers_dataset_path": OUTL_REMOVED_CSV,
        "rows_with_outliers": int(len(iris_with_outliers)),
        "rows_without_outliers": int(len(iris_without_outliers))
    }
})

# ----------------------------------------------------
# 5. Feature selection and log
# ----------------------------------------------------
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
target = 'variety'
log_json({
    "feature_selection": {
        "input_features": features,
        "target_column": target
    }
})

# ----------------------------------------------------
# 6. Feature correlation strategy choices
# ----------------------------------------------------
all_features = features
high_corr_pairs = [("petal.length", "petal.width")]
exclude_feature = 'petal.length'
features_strategy1 = all_features.copy()
features_strategy2 = [feat for feat in all_features if feat != exclude_feature]
log_json({
    "feature_correlation_strategies": {
        "highly_correlated_pairs": high_corr_pairs,
        "strategy_1_all_features": features_strategy1,
        "strategy_2_exclude_petal_length": features_strategy2
    }
})

# ----------------------------------------------------
# 7. Label encoding
# ----------------------------------------------------
le = LabelEncoder()
cleaned_df[target] = le.fit_transform(cleaned_df[target])
label_mapping = {str(cls): int(le.transform([cls])[0]) for cls in le.classes_}
log_json({
    "label_encoding": {
        "target_column": target,
        "label_mapping": label_mapping
    }
})

# ----------------------------------------------------
# 8. Stratified train-test split
# ----------------------------------------------------
X = cleaned_df[features]
y = cleaned_df[target]
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_X = train_X.reset_index(drop=True)
test_X = test_X.reset_index(drop=True)
train_y = train_y.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)
iris_train = pd.concat([train_X, train_y], axis=1)
iris_test = pd.concat([test_X, test_y], axis=1)
iris_train.to_csv(TRAIN_CSV, index=False)
iris_test.to_csv(TEST_CSV, index=False)

unique_labels = list(le.classes_)
train_counts = np.bincount(train_y).tolist()
test_counts = np.bincount(test_y).tolist()
train_class_counts = {str(lbl): int(cnt) for lbl, cnt in zip(unique_labels, train_counts)}
test_class_counts = {str(lbl): int(cnt) for lbl, cnt in zip(unique_labels, test_counts)}
split_info = {
    "train_set_path": TRAIN_CSV,
    "test_set_path": TEST_CSV,
    "train_rows": int(len(train_X)),
    "test_rows": int(len(test_X)),
    "train_indices": [int(idx) for idx in train_X.index.tolist()],
    "test_indices": [int(idx) for idx in test_X.index.tolist()],
    "class_counts_train": train_class_counts,
    "class_counts_test": test_class_counts
}
log_json({"train_test_split": split_info})

# ----------------------------------------------------
# 9. Standardize features (fit on train only)
# ----------------------------------------------------
scaler = StandardScaler()
scaler.fit(train_X)
train_X_scaled = pd.DataFrame(scaler.transform(train_X), columns=features)
test_X_scaled = pd.DataFrame(scaler.transform(test_X), columns=features)
scaler_params = {
    "mean": [float(m) for m in scaler.mean_],
    "std": [float(s) for s in scaler.scale_],
    "feature_names": features
}
log_json({"scaler_parameters": scaler_params})

# ----------------------------------------------------
# 10. Candidate models (limit for runtime!)
# ----------------------------------------------------
# Start with just LogisticRegression and RandomForest to ensure quick runtimes.
candidates = [
    {"name": "LogisticRegression", "lib": "sklearn", 
     "estimator": LogisticRegression(solver="liblinear", multi_class='auto', max_iter=5000, random_state=42)},
    {"name": "RandomForest", "lib": "sklearn", 
     "estimator": RandomForestClassifier(n_jobs=-1, random_state=42)},
]
# Expand to SVC/XGB/LGB only for full production or later experimentation:
if HAS_XGB:
    candidates.append({"name": "XGBClassifier", "lib": "xgboost",
                       "estimator": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)})
if HAS_LGB:
    candidates.append({"name": "LGBMClassifier", "lib": "lightgbm", 
                       "estimator": lgb.LGBMClassifier(random_state=42, n_jobs=-1)})

log_json({
    "candidate_models": [
        {"model_name": m['name'], "library": m['lib']} for m in candidates
    ]
})

# ----------------------------------------------------
# 11. Hyperparameter search spaces (reduced for speed)
# ----------------------------------------------------
search_spaces = {
    "LogisticRegression": {
        "C": [0.1, 1.0]  # Keep very small
    },
    "RandomForest": {
        "n_estimators": [50, 100],  # Only two choices
        "max_depth": [None, 4],     # Only two choices
    },
    "XGBClassifier": {
        "n_estimators": [50],
        "max_depth": [3],
        "learning_rate": [0.1]
    },
    "LGBMClassifier": {
        "n_estimators": [50],
        "max_depth": [-1],
        "learning_rate": [0.1]
    },
}
log_json({
    "hyperparameter_search_spaces": {k:v for k,v in search_spaces.items() if k in [c['name'] for c in candidates]}
})

# ----------------------------------------------------
# 12. Cross-validation loop (very fast setup)
# ----------------------------------------------------
cv_results = []
strategies = [
    {"name": "all_features", "features": features_strategy1},
    {"name": "exclude_petal.length", "features": features_strategy2}
]

print("Starting cross-validation for each candidate model and feature strategy...")
for strategy in strategies:
    for candidate in candidates:
        model_name = candidate['name']
        estimator = candidate['estimator']
        sel_features = strategy['features']
        param_grid = search_spaces[model_name]
        print(f"Model: {model_name} | Features: {sel_features} | Searching params...")

        # Use GridSearch for very simple, use RandomizedSearch for bigger
        if model_name in ["RandomForest", "XGBClassifier", "LGBMClassifier"]:
            searcher = RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                scoring=["accuracy", "f1_macro", "neg_log_loss"],
                refit="f1_macro",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                return_train_score=False,
                n_iter=3,  # drastically reduced
                n_jobs=-1
            )
        else:
            searcher = GridSearchCV(
                estimator,
                param_grid,
                scoring=["accuracy", "f1_macro", "neg_log_loss"],
                refit="f1_macro",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                return_train_score=False,
                n_jobs=-1
            )

        start = time.time()
        searcher.fit(train_X_scaled[sel_features], train_y)
        end = time.time()
        print(f"  ...CV search complete ({end-start:.2f}s) - {model_name} / {strategy['name']}")

        best_params = searcher.best_params_
        cv_means = {}
        for metric in ["mean_test_accuracy", "mean_test_f1_macro", "mean_test_neg_log_loss"]:
            val = searcher.cv_results_[metric][searcher.best_index_]
            cv_means[metric] = float(val)

        entry = {
            "model": model_name,
            "features_used": sel_features,
            "strategy": strategy['name'],
            "cv_metrics": {
                "cv_accuracy": float(cv_means["mean_test_accuracy"]),
                "cv_f1_macro": float(cv_means["mean_test_f1_macro"]),
                "cv_log_loss": float(-cv_means["mean_test_neg_log_loss"]),
            },
            "best_hyperparameters": to_serializable(best_params)
        }
        cv_results.append(entry)

log_json({"cv_results": cv_results})

# ----------------------------------------------------
# 13. Select best model/feature set
# ----------------------------------------------------
cv_df = pd.DataFrame([{
    **r,
    "cv_f1_macro": r["cv_metrics"]["cv_f1_macro"],
    "cv_accuracy": r["cv_metrics"]["cv_accuracy"]
} for r in cv_results])
best_idx = cv_df.sort_values(['cv_f1_macro', 'cv_accuracy'], ascending=False).index[0]
best_model_entry = cv_results[best_idx]
log_json({"best_model": best_model_entry})

# ----------------------------------------------------
# 14. Train best model on full train set and save
# ----------------------------------------------------
best_model_name = best_model_entry['model']
best_features = best_model_entry['features_used']
best_params = best_model_entry['best_hyperparameters']
chosen_model = None
for candidate in candidates:
    if candidate['name'] == best_model_name:
        # Rebuild estimator with best hyperparams and refit
        # (ensure n_jobs=-1 passed if applicable)
        kw = best_params.copy()
        if best_model_name == 'RandomForest':
            kw['n_jobs'] = -1
        elif best_model_name == 'XGBClassifier' or best_model_name == 'LGBMClassifier':
            kw['n_jobs'] = -1
        chosen_model = candidate['estimator'].__class__(**kw)
        break
chosen_model.fit(train_X_scaled[best_features], train_y)
joblib.dump(chosen_model, FINAL_MODEL)
log_json({
    "final_model_file": FINAL_MODEL,
    "trained_model_name": best_model_name,
    "trained_model_params": to_serializable(best_params),
    "features_trained_on": best_features
})

# ----------------------------------------------------
# 15. Evaluate best model on test set
# ----------------------------------------------------
test_X_best = test_X_scaled[best_features]
test_pred = chosen_model.predict(test_X_best)
test_pred_prob = chosen_model.predict_proba(test_X_best) if hasattr(chosen_model, "predict_proba") else None

test_accuracy = float(accuracy_score(test_y, test_pred))
test_macro_f1 = float(f1_score(test_y, test_pred, average='macro'))
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_y, test_pred, average=None, zero_division=0)
conf_mat = confusion_matrix(test_y, test_pred)
test_lloss = float(log_loss(test_y, test_pred_prob)) if test_pred_prob is not None else None

log_json({
    "test_evaluation": {
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "per_class_precision": [float(x) for x in test_prec],
        "per_class_recall": [float(x) for x in test_rec],
        "per_class_f1": [float(x) for x in test_f1],
        "confusion_matrix": to_serializable(conf_mat.tolist()),
        "test_log_loss": test_lloss
    }
})

# ----------------------------------------------------
# 16. Final summary log
# ----------------------------------------------------
final_info = {
    "scaler_parameters": scaler_params,
    "label_mapping": label_mapping,
    "feature_selection_details": {
        "all_features": features_strategy1,
        "dropped_highly_correlated": features_strategy2,
        "selected_features": best_features
    },
    "model_hyperparameters": to_serializable(best_params),
    "final_model_file": FINAL_MODEL,
    "train_test_split": split_info,
    "test_evaluation": {
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "per_class_precision": [float(x) for x in test_prec],
        "per_class_recall": [float(x) for x in test_rec],
        "per_class_f1": [float(x) for x in test_f1],
        "confusion_matrix": to_serializable(conf_mat.tolist()),
        "test_log_loss": test_lloss
    }
}
log_json({"final_artifacts": final_info})

flush_logs()
print(f"\nAll artifacts saved and logged in: {MODEL_JSON}")