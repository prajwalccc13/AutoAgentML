import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
import joblib

# --- Optional Import for XGBoost ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Helper: Safely write JSON
def save_json(data, fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=4)

# Helper: Safely write CSV
def save_csv(df, fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    df.to_csv(fpath, index=False)

# Helper: Update and Save JSON with new key
def update_json_log(log_path, updates):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data.update(updates)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=4)
    return data

# File Paths and Key Columns
EDA_PATH = './output/2/eda_agent.json'
RAW_DATA_PATH = 'data/banana_quality.csv'
CLEANED_DATA_PATH = './output/2/cleaned_data.csv'
TRAIN_DATA_PATH = './output/2/train_data.csv'
TEST_DATA_PATH = './output/2/test_data.csv'
PRED_LOGREG_PATH = './output/2/predictions_logreg.csv'
PRED_RF_PATH = './output/2/predictions_rf.csv'
PRED_XGB_PATH = './output/2/predictions_xgb.csv'
MODEL_LOGREG_PATH = './output/2/model_logreg.pkl'
MODEL_RF_PATH = './output/2/model_rf.pkl'
MODEL_XGB_PATH = './output/2/model_xgb.pkl'
SCALER_PATH = './output/2/scaler_params.json'
MODEL_TRAINING_JSON = './output/2/model_training.json'

FEATURES = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']
TARGET = 'Quality'

log_dict = {}

####################
# 1. Load EDA JSON #
####################
eda_path = EDA_PATH
if os.path.exists(eda_path):
    with open(eda_path, 'r') as f:
        eda_results = json.load(f)
    eda_summary = {
        'data_types': eda_results.get('data_types', {}),
        'missing_values': eda_results.get('missing', {}),
        'cardinality': eda_results.get('cardinality', {}),
        'outliers': eda_results.get('outliers', {}),
        'distribution_summaries': eda_results.get('distribution_summaries', {}),
        'overall_dataset_info': eda_results.get('overall_info', {}),
        'column_insights': eda_results.get('column_insights', {}),
        'recommendations': eda_results.get('eda_recommendations', {})
    }
else:
    eda_summary = 'EDA file not found.'

log_dict['eda_summary'] = eda_summary
save_json(log_dict, MODEL_TRAINING_JSON)

#############################################
# 2. Load Raw Data and Log Basic Info       #
#############################################
df = pd.read_csv(RAW_DATA_PATH)
raw_data_status = {
    'shape': df.shape,
    'columns': list(df.columns),
    'dtypes': df.dtypes.astype(str).to_dict(),
    'head': df.head(3).to_dict(orient='records'),
    'value_counts_quality': df[TARGET].value_counts().to_dict()
}
update_json_log(MODEL_TRAINING_JSON, {'raw_data_status': raw_data_status})

###################################################
# 3. Outlier Treatment on Numeric Columns         #
###################################################
outlier_cols = FEATURES
df_num = df[outlier_cols]
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers
df_cleaned = df.copy()
outlier_records = {}
for col in outlier_cols:
    before_outliers = df_cleaned[(df_cleaned[col] < lower_bound[col]) | (df_cleaned[col] > upper_bound[col])]
    n_outliers = before_outliers.shape[0]
    outlier_records[col] = {
        'n_outliers': int(n_outliers),
        'outlier_indices': before_outliers.index.tolist()[:5]  # Only show a sample
    }
    df_cleaned[col] = np.where(df_cleaned[col] < lower_bound[col], lower_bound[col],
                        np.where(df_cleaned[col] > upper_bound[col], upper_bound[col], df_cleaned[col]))
# Log treatment summary
outlier_treatment = {
    "method": "Capped using 1.5*IQR rule.",
    "outlier_info_by_column": outlier_records,
    "original_shape": df.shape,
    "cleaned_shape": df_cleaned.shape,
    "removed_records_count": 0,  # Since capped not removed
}
save_csv(df_cleaned, CLEANED_DATA_PATH)
update_json_log(MODEL_TRAINING_JSON, {"outlier_treatment": outlier_treatment})

###############
# 4. Missing Treatment
###############
missing_info = df_cleaned.isnull().sum().to_dict()
missing_decision = {
    "missing_counts": missing_info,
    "decision": "No imputation performed as EDA and verification show zero missing values."
}
update_json_log(MODEL_TRAINING_JSON, {"missing_value_treatment": missing_decision})

##############################
# 5. Encode Target Column
##############################
le = LabelEncoder()
df_cleaned[TARGET+'_encoded'] = le.fit_transform(df_cleaned[TARGET])
# Store mapping explicitly
encoding_mapping = {c: int(le.transform([c])[0]) for c in le.classes_}
target_encoding = {
    "encoding_mapping": encoding_mapping,
    "class_value_counts": df_cleaned[TARGET+'_encoded'].value_counts().to_dict()
}
update_json_log(MODEL_TRAINING_JSON, {"target_encoding": target_encoding})

##############################
# 6. Extract Features/Target
##############################
selected_features_info = df_cleaned[FEATURES].dtypes.astype(str).to_dict()
feature_selection_log = {
    "selected_features": FEATURES,
    "data_types": selected_features_info,
    "target_column": TARGET+'_encoded'
}
update_json_log(MODEL_TRAINING_JSON, {"feature_selection": feature_selection_log})

###################################
# 7. Train-Test Split
###################################
X = df_cleaned[FEATURES]
y = df_cleaned[TARGET+'_encoded']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_test_info = {
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "train_target_distribution": y_train.value_counts(normalize=True).to_dict(),
    "test_target_distribution": y_test.value_counts(normalize=True).to_dict()
}
save_csv(pd.concat([X_train, y_train.reset_index(drop=True)], axis=1), TRAIN_DATA_PATH)
save_csv(pd.concat([X_test, y_test.reset_index(drop=True)], axis=1), TEST_DATA_PATH)
update_json_log(MODEL_TRAINING_JSON, {"train_test_split": train_test_info})

#########################################################
# 8. Feature Scaling (Standardize using StandardScaler)
#########################################################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_params = {
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
    "var_": scaler.var_.tolist(),
    "feature_names": FEATURES
}
save_json(scaler_params, SCALER_PATH)
feature_scaling_log = {
    "scaler": "StandardScaler",
    "fit_on": "Training data",
    "feature_stats_pre_scaling": {
        "train": {col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())} for col in FEATURES},
        "test": {col: {"mean": float(X_test[col].mean()), "std": float(X_test[col].std())} for col in FEATURES}
    },
    "feature_stats_post_scaling": {
        "train": {col: {"mean": float(X_train_scaled[:, i].mean()), "std": float(X_train_scaled[:, i].std())} for i, col in enumerate(FEATURES)},
        "test": {col: {"mean": float(X_test_scaled[:, i].mean()), "std": float(X_test_scaled[:, i].std())} for i, col in enumerate(FEATURES)}
    }
}
update_json_log(MODEL_TRAINING_JSON, {"feature_scaling": feature_scaling_log})

####################################
# 9. Train Models & Hyperparameter Tuning
####################################
models_and_params = [
    {
        'name': 'LogisticRegression',
        'estimator': LogisticRegression(solver='liblinear', max_iter=200),
        'param_grid': {"C": [0.01, 0.1, 1, 10]}
    },
    {
        'name': 'RandomForest',
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 4]
        }
    }
]
if XGBOOST_AVAILABLE:
    models_and_params.append(
        {
            'name': 'XGBoost',
            'estimator': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'param_grid': {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1]
            }
        }
    )

cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_training_results = {}
best_val_score = -np.inf
final_model_name = ""
final_model_path = ""
best_cv_results = None
best_estimator = None

for model_info in models_and_params:
    name = model_info['name']
    est = model_info['estimator']
    param_grid = model_info['param_grid']
    grid = GridSearchCV(est,
                        param_grid,
                        cv=cv_outer,
                        scoring='f1',
                        n_jobs=-1,
                        return_train_score=False)
    grid.fit(X_train_scaled, y_train)
    # Cross-validation scores for the best estimator
    y_val_pred = grid.predict(X_train_scaled)
    y_val_prob = grid.predict_proba(X_train_scaled)[:,1] if hasattr(grid, 'predict_proba') else None

    val_metrics = {
        "best_params": grid.best_params_,
        "cv_results": { 
            "mean_test_score": float(np.mean(grid.cv_results_['mean_test_score'])),
            "std_test_score": float(np.std(grid.cv_results_['mean_test_score'])),
        },
        "training_metrics": {
            "accuracy": float(accuracy_score(y_train, y_val_pred)),
            "precision": float(precision_score(y_train, y_val_pred)),
            "recall": float(recall_score(y_train, y_val_pred)),
            "f1_score": float(f1_score(y_train, y_val_pred)),
            "roc_auc": float(roc_auc_score(y_train, y_val_prob)) if y_val_prob is not None else None,
        }
    }
    # Save model
    if name == 'LogisticRegression':
        joblib.dump(grid.best_estimator_, MODEL_LOGREG_PATH)
        model_file_path = MODEL_LOGREG_PATH
    elif name == 'RandomForest':
        joblib.dump(grid.best_estimator_, MODEL_RF_PATH)
        model_file_path = MODEL_RF_PATH
    elif name == 'XGBoost':
        joblib.dump(grid.best_estimator_, MODEL_XGB_PATH)
        model_file_path = MODEL_XGB_PATH

    model_training_results[name] = {
        "model_type": name,
        "best_params": grid.best_params_,
        "mean_cv_f1": float(grid.best_score_),
        "cv_metrics": val_metrics,
        "model_path": model_file_path
    }
    # Track best model on F1-score (validation) and ROC-AUC if available
    validation_score = float(grid.best_score_)
    roc_auc = val_metrics["training_metrics"]["roc_auc"]
    effective_score = (validation_score + (roc_auc if roc_auc is not None else 0)) / 2
    if effective_score > best_val_score:
        best_val_score = effective_score
        final_model_name = name
        final_model_path = model_file_path
        best_cv_results = val_metrics
        best_estimator = grid.best_estimator_

update_json_log(MODEL_TRAINING_JSON, {"model_training_results": model_training_results})

##############################################
# 10. Evaluate Models on Test Set & Log
##############################################
test_set_evaluation = {}
for name in model_training_results:
    # Select model path
    if name == "LogisticRegression":
        model = joblib.load(MODEL_LOGREG_PATH)
    elif name == "RandomForest":
        model = joblib.load(MODEL_RF_PATH)
    elif name == "XGBoost" and XGBOOST_AVAILABLE:
        model = joblib.load(MODEL_XGB_PATH)
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
        "predicted_class_distribution": pd.Series(y_pred).value_counts().to_dict()
    }
    test_set_evaluation[name] = test_metrics
    # Save predictions
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob.tolist() if y_prob is not None else [None]*len(y_pred)
    })
    out_pred_path = f"./output/2/predictions_{name.lower()}.csv"
    save_csv(pred_df, out_pred_path)
update_json_log(MODEL_TRAINING_JSON, {"test_set_evaluation": test_set_evaluation})

#####################################################
# 11. Select Best Model and Log Rationale
#####################################################
best_model_metric = None
max_score = -1
for name, metrics in test_set_evaluation.items():
    # selection on f1 and roc_auc (equal weight)
    score = metrics['f1_score']
    if metrics['roc_auc'] is not None:
        score = (score + metrics['roc_auc']) / 2
    if score > max_score:
        max_score = score
        best_model_metric = metrics
        final_model_name = name
        final_model_path = model_training_results[name]['model_path']

final_model_selection = {
    "winning_model": final_model_name,
    "winning_model_path": final_model_path,
    "selection_rationale": "Highest combined F1-score and ROC-AUC on test set among all candidates.",
    "winning_test_metrics": best_model_metric
}
update_json_log(MODEL_TRAINING_JSON, {"final_model_selection": final_model_selection})

#####################################################
# 12. Pipeline Summary Outputs
#####################################################
pipeline_outputs = {
    "steps": [
        "EDA Summary Logged",
        "Raw dataset loaded and logged",
        "Capped outliers using 1.5*IQR rule",
        "Checked for missing values, no imputations performed",
        "Encoded target Quality using label encoding",
        f"Features selected: {FEATURES}",
        "Train-test split (80/20), stratified",
        "Standardized numeric features with StandardScaler",
        "Model training and tuning: Logistic Regression, Random Forest, XGBoost (if available)",
        "Evaluation on held-out test set",
        f"Best model selected: {final_model_name}",
    ],
    "files": {
        "eda_summary": EDA_PATH,
        "cleaned_data": CLEANED_DATA_PATH,
        "train_data": TRAIN_DATA_PATH,
        "test_data": TEST_DATA_PATH,
        "scaler_params": SCALER_PATH,
        "model_logreg": MODEL_LOGREG_PATH,
        "model_rf": MODEL_RF_PATH,
        "model_xgb": MODEL_XGB_PATH if XGBOOST_AVAILABLE else None,
        "predictions_logreg": PRED_LOGREG_PATH,
        "predictions_rf": PRED_RF_PATH,
        "predictions_xgb": PRED_XGB_PATH if XGBOOST_AVAILABLE else None,
        "model_training_json": MODEL_TRAINING_JSON
    }
}
update_json_log(MODEL_TRAINING_JSON, {"pipeline_outputs": pipeline_outputs})

print(f"Pipeline completed. Logs saved to {MODEL_TRAINING_JSON}")