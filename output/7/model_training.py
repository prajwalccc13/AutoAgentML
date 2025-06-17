import os
import json
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict
from collections import defaultdict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

warnings.filterwarnings('ignore')

# Paths
EDA_OUT = './output/7/eda_agent.json'
DATA_CSV = 'data/banana_quality.csv'
OUTPUT_DIR = './output/7/'
TRAINING_JSON = os.path.join(OUTPUT_DIR, 'model_training.json')
CLEANED_CSV_PATH = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_update(key: str, value: Any):
    try:
        if os.path.isfile(TRAINING_JSON):
            with open(TRAINING_JSON, 'r') as f:
                logs = json.load(f)
        else:
            logs = {}
        logs[key] = value
        with open(TRAINING_JSON, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"ERROR logging {key}: {e}")
        raise

# --- Task 1: Robustly Read and Parse EDA output ---
def parse_schema_from_eda_summary(eda_summary: dict, sample_df: pd.DataFrame = None) -> Dict[str, Dict]:
    """
    Try to extract a column-to-dtype mapping from the EDA summary.
    - Returns dict: {column_name: {'dtype': dtype_str}}
    - If no dtype is found, use 'object' unless a sample_df is provided for inference.
    """
    if 'schema' in eda_summary:
        schema = eda_summary['schema']
        # Some EDA tools output {col: dtype} or {col: {'dtype': ...}}, so standardize:
        result = {}
        for col, typ in schema.items():
            if isinstance(typ, dict) and 'dtype' in typ:
                result[col] = {'dtype': typ['dtype']}
            else:
                result[col] = {'dtype': str(typ)}
        return result

    # Fallback: check for 'columns' and try to extract both names and types
    columns = eda_summary.get('columns')
    if columns:
        schema = {}
        for colinfo in columns:
            if isinstance(colinfo, dict) and 'name' in colinfo and 'dtype' in colinfo:
                schema[colinfo['name']] = {'dtype': colinfo['dtype']}
            elif isinstance(colinfo, str):
                # fallback: try to infer from sample_df if provided
                fallback_dtype = 'object'
                if sample_df is not None and colinfo in sample_df.columns:
                    fallback_dtype = str(sample_df[colinfo].dtype)
                schema[colinfo] = {'dtype': fallback_dtype}
            # else: ignore or warn
        return schema

    # Absolute fallback: use DataFrame columns if provided
    if sample_df is not None:
        return {col: {'dtype': str(sample_df[col].dtype)} for col in sample_df.columns}
    raise ValueError("Unable to determine column schema from EDA summary.")


with open(EDA_OUT, 'r') as f:
    eda_summary = json.load(f)
log_update('eda_summary', eda_summary)

# --- Task 2: Load dataset with EDA datatypes ---

# Read once with default dtypes to allow for fallback inference if necessary
_sample_df = pd.read_csv(DATA_CSV, nrows=100)  # read a sample for type hints if needed

try:
    schema = parse_schema_from_eda_summary(eda_summary, sample_df=_sample_df)
except Exception as e:
    print(f"[WARN] Schema parsing failed: {e} -- falling back to sample DataFrame types.")
    schema = {col: {'dtype': str(_sample_df[col].dtype)} for col in _sample_df.columns}

column_dtypes = {col: (v['dtype'] if isinstance(v, dict) else v) for col, v in schema.items()}

# Map EDA-types to pandas types
def map_dtype(dtype_str):
    d = dtype_str.lower()
    if d.startswith('float'): return 'float'
    if d.startswith('int'): return 'int'
    if 'category' in d or d == 'object' or d == 'str': return 'object'
    return d

pd_col_dtypes = {col: map_dtype(t) for col, t in column_dtypes.items()}

# Read CSV with mapped dtypes for as much as possible
df = pd.read_csv(DATA_CSV, dtype=pd_col_dtypes)
dataset_info = {
    'shape': df.shape,
    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
}
log_update('dataset_info', dataset_info)

# --- Task 3: Data Quality Checks ---
data_quality_checks = {}

missing_vals = df.isnull().sum().to_dict()
data_quality_checks['missing_values'] = missing_vals

# Near-constant columns according to EDA (be robust for type/structure)
near_constant_cols = eda_summary.get('near_constant_columns', [])
if not near_constant_cols:
    # Fallback (robust): detect those with only one unique value
    near_constant_cols = [col for col in df.columns if df[col].nunique() == 1]
data_quality_checks['near_constant_columns'] = near_constant_cols

n_duplicates = df.duplicated().sum()
data_quality_checks['duplicated_rows'] = int(n_duplicates)

log_update('data_quality_checks', data_quality_checks)

# --- Task 4: Outlier Detection and Handling ---
outlier_handling = defaultdict(dict)
feature_outliers = defaultdict(list)
outlier_info = eda_summary.get('outliers', {})  # Should be {col: {indices, ...}} but check type
numerical_features = []
for col in df.columns:
    t = column_dtypes[col] if col in column_dtypes else str(df[col].dtype)
    if map_dtype(t) in ['float', 'int']:
        numerical_features.append(col)

for feat in numerical_features:
    # Outlier indices: from EDA if present and well-formed
    indices_val = None
    if feat in outlier_info:
        feat_outlier_obj = outlier_info[feat]
        # Check if this is a dict with indices key or a list of indices
        if isinstance(feat_outlier_obj, dict) and 'indices' in feat_outlier_obj:
            indices_val = feat_outlier_obj['indices']
        elif isinstance(feat_outlier_obj, list):
            indices_val = feat_outlier_obj
    if indices_val is None:
        # fallback: 1.5*IQR rule
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        indices_val = df[(df[feat] < lower) | (df[feat] > upper)].index.tolist()
    feature_outliers[feat] = indices_val
    outlier_handling[feat]['outlier_indices'] = indices_val
    outlier_handling[feat]['n_outliers'] = len(indices_val)

    # Choose cap or remove; logic can be customized here
    remove_or_cap = 'remove'
    if feat.lower() == 'sweetness' and len(indices_val) > 10:
        remove_or_cap = 'cap'
    outlier_handling[feat]['handling'] = remove_or_cap

    if remove_or_cap == 'remove' and len(indices_val) > 0:
        # Remove rows
        df = df.drop(indices_val)
    elif remove_or_cap == 'cap' and len(indices_val) > 0:
        # Cap at 1st/99th percentile
        low = df[feat].quantile(0.01)
        high = df[feat].quantile(0.99)
        df[feat] = np.clip(df[feat], low, high)
        outlier_handling[feat]['cap_low'] = float(low)
        outlier_handling[feat]['cap_high'] = float(high)
# Reset the index after outlier removal
df = df.reset_index(drop=True)
outlier_handling['resulting_shape'] = df.shape
df.to_csv(CLEANED_CSV_PATH, index=False)
outlier_handling['cleaned_csv_path'] = CLEANED_CSV_PATH

log_update('outlier_handling', outlier_handling)

# --- Task 5: Correlation Analysis ---
correlation_matrix = eda_summary.get('correlation_matrix', {})
if not (correlation_matrix and isinstance(correlation_matrix, dict)):
    corr = df[numerical_features].corr(method='pearson')
    correlation_matrix = corr.to_dict()
high_corr_pairs = []
already_seen = set()
for col1 in correlation_matrix:
    if not isinstance(correlation_matrix[col1], dict):
        continue
    for col2 in correlation_matrix[col1]:
        if col1 != col2 and abs(correlation_matrix[col1][col2]) > 0.85:
            pair = tuple(sorted([col1, col2]))
            if pair not in already_seen:
                high_corr_pairs.append(pair)
                already_seen.add(pair)
feature_decisions = {}
selected_features = numerical_features.copy()
for c1, c2 in high_corr_pairs:
    # By default drop second in the pair if not already removed
    if c2 in selected_features:
        selected_features.remove(c2)
    feature_decisions[f'{c1} vs {c2}'] = f"Drop '{c2}' due to high correlation with '{c1}'"
feature_selection = {
    'high_corr_pairs': high_corr_pairs,
    'decisions': feature_decisions,
    'selected_features': selected_features
}
log_update('feature_selection', feature_selection)

# --- Task 6: Identify Feature Types and Encode/Standardize ---
categorical_features = []
target_col = 'Quality'
for col in df.columns:
    if map_dtype(column_dtypes.get(col, str(df[col].dtype))) == 'object' and col != target_col:
        categorical_features.append(col)
# Use supported features for modeling
all_features = [col for col in selected_features if col != target_col] + categorical_features

# Encode target
df[target_col + "_orig"] = df[target_col]
df[target_col] = df[target_col].map({'Good': 1, 'Bad': 0})
if df[target_col].isnull().any():
    # fallback: LabelEncoder
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col + "_orig"])
else:
    le = None

# Standardize numerical features (selected)
scaler = StandardScaler()
df_std = df.copy()
scaler.fit(df_std[selected_features])
df_std[selected_features] = scaler.transform(df_std[selected_features])
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'features': selected_features
}
le_params = {
    'classes_': le.classes_.tolist() if le else ['Bad', 'Good'],
    'mapping': {'Bad': 0, 'Good': 1}
}
preprocessing = {
    'scaler_params': scaler_params,
    'label_encoder': le_params
}

# Save scaler and label encoder promptly
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
joblib.dump(le, encoder_path)
preprocessing['scaler_path'] = scaler_path
preprocessing['encoder_path'] = encoder_path
log_update('preprocessing', preprocessing)

# --- Task 7: Train/Test Split ---
X = df_std[all_features]
y = df_std[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)
split_indices = {
    'train_indices': X_train.index.tolist(),
    'test_indices': X_test.index.tolist(),
    'y_train_distribution': y_train.value_counts(normalize=True).to_dict(),
    'y_test_distribution': y_test.value_counts(normalize=True).to_dict()
}
log_update('train_test_split', split_indices)

# --- Task 8: Train models (LogReg, RF, XGB) with CV ---
trained_models = {}
cv_results = {}

# Logistic Regression
logreg = LogisticRegression(solver='liblinear', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logreg_cv_scores = cross_val_score(logreg, X_train, y_train, cv=cv, scoring='f1')
logreg.fit(X_train, y_train)
logreg_path = os.path.join(OUTPUT_DIR, 'logreg_model.pkl')
joblib.dump(logreg, logreg_path)
trained_models['logreg'] = {'model_path': logreg_path}
cv_results['logreg'] = {'cv_scores_f1': logreg_cv_scores.tolist()}

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1')
rf.fit(X_train, y_train)
rf_path = os.path.join(OUTPUT_DIR, 'rf_model.pkl')
joblib.dump(rf, rf_path)
trained_models['rf'] = {'model_path': rf_path}
cv_results['rf'] = {'cv_scores_f1': rf_cv_scores.tolist()}

# XGBoost (optional)
if xgb_available:
    xgb = XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='f1')
    xgb.fit(X_train, y_train)
    xgb_path = os.path.join(OUTPUT_DIR, 'xgb_model.pkl')
    joblib.dump(xgb, xgb_path)
    trained_models['xgb'] = {'model_path': xgb_path}
    cv_results['xgb'] = {'cv_scores_f1': xgb_cv_scores.tolist()}

model_training = {'trained_models': trained_models, 'cv_results': cv_results}
log_update('model_training', model_training)

# --- Task 9: Evaluate on test set ---
model_evaluation = {}
def get_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
# logreg
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]
model_evaluation['logreg'] = get_metrics(y_test, y_pred, y_prob)
# rf
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
model_evaluation['rf'] = get_metrics(y_test, y_pred, y_prob)
# xgb
if xgb_available:
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    model_evaluation['xgb'] = get_metrics(y_test, y_pred, y_prob)

log_update('model_evaluation', model_evaluation)

# --- Task 10: Select best model ---
# Based on F1-score first, then ROC-AUC
scores = []
for m in model_evaluation:
    scores.append((m, model_evaluation[m]['f1_score'], model_evaluation[m]['roc_auc']))
best = sorted(scores, key=lambda x: (x[1], x[2]), reverse=True)[0]
best_id = best[0]
best_rationale = (
    f"Selected '{best_id}' due to highest F1-score ({best[1]:.3f}); "
    f"ROC-AUC: {best[2]:.3f}."
)
best_model = {
    'selected_model': best_id,
    'selection_rationale': best_rationale,
    'performance': model_evaluation[best_id],
    'model_path': trained_models[best_id]['model_path']
}
log_update('best_model', best_model)

# --- Task 11: Save artifacts ---
final_artifacts = {
    'final_model_path': trained_models[best_id]['model_path'],
    'scaler_path': scaler_path,
    'encoder_path': encoder_path,
    'preprocessing_params': preprocessing,
    'selected_features': selected_features
}
log_update('final_artifacts', final_artifacts)

print("Workflow complete. Results stored in:", TRAINING_JSON)