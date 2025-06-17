import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import joblib
from datetime import datetime

# Utility Functions
def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# ========== PATHS ==========
OUTPUT_DIR = './output/id_2'
EDA_JSON_PATH = os.path.join(OUTPUT_DIR, 'eda_agent.json')
MODEL_JSON_PATH = os.path.join(OUTPUT_DIR, 'model_training.json')
DATA_PATH = 'data/banana_quality.csv'
TRAIN_CSV = os.path.join(OUTPUT_DIR, 'train.csv')
TEST_CSV = os.path.join(OUTPUT_DIR, 'test.csv')
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pkl')

# ========== STEP 1: LOAD DATA & EDA LOGS ==========
ensure_dir_exists(OUTPUT_DIR)
logs = {}

# Load EDA and data (with try/except to log/read issues)
eda_results = {}
eda_loading_issues = []
try:
    eda_results = load_json(EDA_JSON_PATH)
except Exception as e:
    eda_loading_issues.append(f"Could not load eda_agent.json: {e}")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load data from {DATA_PATH}: {e}")

logs['data_overview'] = {
    'dataset_path': DATA_PATH,
    'n_rows': int(df.shape[0]),
    'n_columns': int(df.shape[1]),
    'columns': list(df.columns),
}
if eda_loading_issues:
    logs['eda_load_warnings'] = eda_loading_issues

# --- Defensive feature types ---
feature_types = eda_results.get('feature_types')
if not feature_types or not isinstance(feature_types, dict) or len(feature_types) == 0:
    # Fallback: use pandas dtypes directly
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_types = {col: 'numeric' for col in numeric_cols}
    logs['feature_types_fallback'] = (
        "Used df dtypes to infer numeric features because EDA artifact was missing, empty or invalid."
    )
logs['feature_types'] = feature_types

# --- Defensive: check columns alignment ---
eda_cols = set(feature_types.keys())
df_cols = set(df.columns)
missing_in_eda = list(df_cols - eda_cols)
missing_in_df = list(eda_cols - df_cols)
if missing_in_eda:
    logs['feature_types_mismatch/missing_in_eda'] = missing_in_eda
if missing_in_df:
    logs['feature_types_mismatch/missing_in_df'] = missing_in_df

# Defensive: Log feature_types and df.columns pre-selection for diagnostics
logs['feature_types_keys'] = list(feature_types.keys())
logs['df_columns'] = list(df.columns)

missing_report = eda_results.get('missing_report', {})
logs['missing_values'] = missing_report
cardinality_report = eda_results.get('cardinality', {})
logs['cardinality'] = cardinality_report

outlier_indices = eda_results.get('outlier_indices', {}) if eda_results else {}
logs['outlier_indices'] = {col: indices for col, indices in outlier_indices.items() if indices}

save_json(logs, MODEL_JSON_PATH)

# ========== STEP 2: VALIDATE NO MISSING/DUPE ROWS ==========
validations = {}
# Defensive: missing_report = EDA or computed from df as fallback
if not missing_report:
    computed_missing = df.isnull().sum().to_dict()
    missing_report = computed_missing
n_duplicates = df.duplicated().sum()
no_missing = all((missing_report.get(col, 0)==0) for col in df.columns)

validations['no_missing'] = no_missing
validations['no_duplicate_rows'] = (n_duplicates == 0)
validations['n_duplicate_rows'] = int(n_duplicates)
validations['validated_and_cleaned_path'] = DATA_PATH
logs['data_validation'] = validations
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 3: OUTLIER INDICATORS ==========
outlier_info_log = {}
outlier_new_features = []

if not outlier_indices:
    logs['outlier_detection_warning'] = (
        "No outlier indices found in EDA JSON. No outlier indicator features will be created."
    )

for col, inds in outlier_indices.items() if outlier_indices else []:
    if not inds:
        continue
    col_feat = f"{col}_is_outlier"
    outlier_info_log[col] = {
        "n_outliers": len(inds),
        "indices_first_50": inds[:50]
    }
    outlier_new_features.append(col_feat)
    df[col_feat] = 0
    if len(inds):
        # Make sure indices are valid
        inds_valid = [ix for ix in inds if 0 <= ix < len(df)]
        df.loc[inds_valid, col_feat] = 1

logs['outlier_feature_info'] = outlier_info_log
logs['outlier_indicator_features'] = outlier_new_features
save_json(logs, MODEL_JSON_PATH)

for newcol in outlier_new_features:
    assert newcol in df.columns, f"Expected outlier indicator column {newcol} to appear in DataFrame after creation."

# ========== STEP 4: FEATURE SELECTION ANALYSIS ==========
# Defensive: Allow only those numeric features present in DF.
num_features = [col for col, typ in feature_types.items() if typ == 'numeric' and col in df.columns]
summary_stats = eda_results.get('summary_statistics', {})
correlations = eda_results.get('correlation_matrix', {})
target_col = 'Quality'

# Selected features are numeric + any outlier indicators, if available
selected_features = [f for f in (num_features + outlier_new_features) if f in df.columns]

logs['feature_selection_analysis_pre_assert'] = {
    'num_features': num_features,
    'outlier_new_features': outlier_new_features,
    'selected_features': selected_features
}
if not selected_features:
    logs['feature_selection_analysis_error'] = (
        "No features selected for modeling! "
        "num_features: {}, outlier_new_features: {}, df.columns: {}".format(
            num_features, outlier_new_features, list(df.columns)
        )
    )
    save_json(logs, MODEL_JSON_PATH)
    raise AssertionError("No features selected for modeling! See model_training.json for diagnostics.")

assert len(selected_features) > 0, "No features selected for modeling!"
for f in selected_features:
    assert f in df.columns, f"Selected feature {f} not in dataframe columns!"

logs['feature_selection_analysis'] = {
    'selected_numeric_features': num_features,
    'added_outlier_indicator_features': outlier_new_features,
    'final_selected_features': selected_features,
    'all_selected_actually_exist_in_df': [f for f in selected_features if f in df.columns],
    'summary_stats': {k: summary_stats[k] for k in selected_features if summary_stats and k in summary_stats},
    'correlations_with_target': {col: correlations.get(target_col, {}).get(col, None) if correlations else None for col in selected_features}
}
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 5: ENCODE TARGET ==========
target_map = {'Good': 1, 'Bad': 0}
if target_col not in df.columns:
    logs['target_encoding_error'] = (
        f"Target column '{target_col}' not present in dataframe columns: {list(df.columns)}"
    )
    save_json(logs, MODEL_JSON_PATH)
    raise ValueError(logs['target_encoding_error'])

df['Quality_binary'] = df[target_col].map(target_map)
if df['Quality_binary'].isnull().any():
    unmapped_vals = df[target_col][df['Quality_binary'].isnull()].unique().tolist()
    logs['target_encoding_error'] = (
        f"Encoding of target column failed! Unmapped values: {unmapped_vals}"
    )
    save_json(logs, MODEL_JSON_PATH)
    raise ValueError(logs['target_encoding_error'])

logs['target_encoding'] = {'column': target_col, 'encoding_map': target_map, 'new_column': 'Quality_binary'}
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 6: FEATURE TYPE VALIDATION ==========
feature_types_post = df[selected_features].dtypes.apply(lambda x: str(x)).to_dict()
encoding_status = all(np.issubdtype(df[f].dtype, np.number) for f in selected_features)
logs['feature_type_validation'] = {
    'selected_features': selected_features,
    'dtypes': feature_types_post,
    'all_numeric_or_binary': encoding_status
}
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 7: STRATIFIED TRAIN-TEST SPLIT ==========
X = df[selected_features]
y = df['Quality_binary']
assert X.shape[1] > 0, "X has zero columns!"
assert len(X) > 0, "X has zero rows!"
assert len(y) == len(X), "Feature matrix and target vector have mismatched shapes!"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
assert X_train.shape[0] == y_train.shape[0], "Train set X/y row mismatch"
assert X_test.shape[0] == y_test.shape[0], "Test set X/y row mismatch"

train_data = X_train.copy()
train_data['Quality_binary'] = y_train
test_data = X_test.copy()
test_data['Quality_binary'] = y_test

# Defensive: Each split has non-zero samples and all features as expected
assert train_data.shape[0] > 0 and test_data.shape[0] > 0, "Empty train or test set!"
assert list(X_train.columns) == selected_features, "Mismatch in train X columns vs. selected_features"
assert list(X_test.columns) == selected_features, "Mismatch in test X columns vs. selected_features"

train_data.to_csv(TRAIN_CSV, index=False)
test_data.to_csv(TEST_CSV, index=False)

split_stats = {
    'train_shape': train_data.shape,
    'test_shape': test_data.shape,
    'train_class_counts': train_data['Quality_binary'].value_counts().to_dict(),
    'test_class_counts': test_data['Quality_binary'].value_counts().to_dict(),
    'train_path': TRAIN_CSV,
    'test_path': TEST_CSV
}
logs['data_splits'] = split_stats
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 8: MODEL DEFINITIONS ==========
model_defs = {}

# Logistic Regression
model_defs['logistic_regression'] = {
    'type': 'LogisticRegression',
    'import': 'sklearn.linear_model.LogisticRegression',
    'hyperparameters': {'solver': 'liblinear', 'random_state': 42},
}

# Random Forest
model_defs['random_forest'] = {
    'type': 'RandomForestClassifier',
    'import': 'sklearn.ensemble.RandomForestClassifier',
    'hyperparameters': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
}

# XGBoost
model_defs['xgboost'] = {
    'type': 'XGBClassifier',
    'import': 'xgboost.XGBClassifier',
    'hyperparameters': {
        'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3,
        'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'
    }
}

logs['model_definitions'] = model_defs
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 9: MODEL TRAINING + CV ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_results_cv = {}
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

def get_model(kind):
    if kind == 'logistic_regression':
        return LogisticRegression(**model_defs[kind]['hyperparameters'])
    elif kind == 'random_forest':
        return RandomForestClassifier(**model_defs[kind]['hyperparameters'])
    elif kind == 'xgboost':
        return XGBClassifier(**model_defs[kind]['hyperparameters'])
    else:
        raise ValueError(f"Unknown model kind: {kind}")

for mname in model_defs.keys():
    model = get_model(mname)
    try:
        assert X_train.shape[1] > 0, f"For model {mname}: X_train has zero columns!"
        assert X_train.shape[0] > 0, f"For model {mname}: X_train has zero rows!"
        assert y_train.shape[0] > 0, f"For model {mname}: y_train has zero rows!"
        results = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        model_results_cv[mname] = {
            'test_accuracy': results['test_accuracy'].tolist(),
            'test_precision': results['test_precision'].tolist(),
            'test_recall': results['test_recall'].tolist(),
            'test_f1': results['test_f1'].tolist(),
            'test_roc_auc': results['test_roc_auc'].tolist(),
            'mean_metrics': {k: float(np.mean(v)) for k, v in results.items() if 'test_' in k}
        }
    except Exception as e:
        logs[f'cross_validate_error_{mname}'] = str(e)
        logs[f'cross_validate_error_{mname}_X_train_shape'] = X_train.shape
        logs[f'cross_validate_error_{mname}_X_train_columns'] = list(X_train.columns)
        logs[f'cross_validate_error_{mname}_y_train_summary'] = y_train.describe().to_dict()
        save_json(logs, MODEL_JSON_PATH)
        raise

logs['model_cross_validation'] = model_results_cv
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 10: FEATURE IMPORTANCES ==========
feature_importance_log = {}
for mname in model_defs.keys():
    model = get_model(mname)
    model.fit(X_train, y_train)
    if mname == 'logistic_regression':
        importance = dict(zip(selected_features, model.coef_[0].tolist()))
    elif mname in ['random_forest', 'xgboost']:
        importance = dict(zip(selected_features, model.feature_importances_.tolist()))
    else:
        importance = {}
    feature_importance_log[mname] = importance

logs['feature_importances'] = feature_importance_log
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 11: FINAL BEST MODEL (ROC-AUC/F1) ==========
cv_summary = {m: v['mean_metrics']['test_roc_auc'] for m, v in model_results_cv.items()}
best_model_name = max(cv_summary, key=cv_summary.get)

final_model = get_model(best_model_name)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:,1] if hasattr(final_model, "predict_proba") else None
test_metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}

logs['final_model_evaluation'] = {
    'best_model': best_model_name,
    'test_set_metrics': test_metrics
}
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 12: SAVE FINAL MODEL ==========
joblib.dump(final_model, BEST_MODEL_PATH)
best_model_info = {
    'model_file_path': BEST_MODEL_PATH,
    'model_type': best_model_name,
    'hyperparameters': model_defs[best_model_name]['hyperparameters'],
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
logs['final_model_artifact'] = best_model_info
save_json(logs, MODEL_JSON_PATH)

# ========== STEP 13: TOP-LEVEL SUMMARY ==========
logs['summary'] = {
    'data_split_files': {'train': TRAIN_CSV, 'test': TEST_CSV},
    'selected_features': selected_features,
    'model_types': list(model_defs.keys()),
    'model_hyperparameters': {m: model_defs[m]['hyperparameters'] for m in model_defs},
    'cv_evaluation_results': {m: model_results_cv[m]['mean_metrics'] for m in model_results_cv},
    'feature_importances': feature_importance_log.get(best_model_name, {}),
    'final_model': {
        'type': best_model_name,
        'model_file': BEST_MODEL_PATH,
        'hyperparameters': model_defs[best_model_name]['hyperparameters'],
        'test_metrics': test_metrics,
        'timestamp': best_model_info['timestamp']
    }
}
save_json(logs, MODEL_JSON_PATH)

print(f"All steps completed. Results saved to {MODEL_JSON_PATH}")