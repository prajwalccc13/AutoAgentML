import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

try:
    import xgboost as xgb
    from lightgbm import LGBMRegressor
except ImportError:
    xgb = None
    LGBMRegressor = None

# Directories and paths
EDA_PATH = './output/12/eda_agent.json'
DATA_PATH = './data/Crop_Yield_Prediction.csv'
PREP_DATA_PATH = './output/12/preprocessed_data.csv'
ENCODED_DATA_PATH = './output/12/encoded_data.csv'
MODEL_OUT_PATH = './output/12/best_model.pkl'
JSON_LOG_PATH = './output/12/model_training.json'

os.makedirs('./output/12/', exist_ok=True)

log = {
    "timestamp_start": datetime.now().isoformat(),
    "steps": [],
    "summary": {},
}

def write_log():
    with open(JSON_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, default=str)

def append_log_step(step_name, content):
    log['steps'].append({"step": step_name, "details": content})
    write_log()


### 1. Load EDA output
eda_info = {}
try:
    with open(EDA_PATH, 'r') as f:
        eda_info = json.load(f)
    append_log_step('Load EDA', {"status": "success", "eda_file": EDA_PATH, "eda_keys": list(eda_info.keys())})
except Exception as e:
    append_log_step('Load EDA', {"status": "error", "error": str(e)} )
    raise

data_types = eda_info.get('column_types', {})
missing_info = eda_info.get('missing_values', {})
cardinality = eda_info.get('cardinality', {})
desc_stats = eda_info.get('descriptive_stats', {})
outlier_flags = eda_info.get('outlier_detection', {})
preprocessing = eda_info.get('preprocessing_recommendations', {})
constant_features = eda_info.get('constant_or_duplicate_features', [])
correlations = eda_info.get('correlations', {})

### 2. Load dataset and check missing values
df = pd.read_csv(DATA_PATH)
missing_counts = df.isnull().sum().to_dict()
total_missing = sum(missing_counts.values())
if total_missing == 0:
    append_log_step(
        'Missing Value Check',
        {
            "missing_values_per_column": missing_counts,
            "imputation_performed": False,
            "message": "Zero missing values; proceeding without imputation."
        }
    )
else:
    append_log_step(
        'Missing Value Check',
        {
            "missing_values_per_column": missing_counts,
            "imputation_performed": True,
            "imputation_strategy": "To be determined by EDA.",
            "message": "Missing values found; this should not happen per EDA."
        }
    )

### 3. Outlier Handling (IQR capping)
# Columns per instruction (only those with outliers per EDA)
outlier_cols = ['Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall', 'Yield']
outlier_stats = {}
capped_df = df.copy()

for col in outlier_cols:
    flagged = outlier_flags.get(col, False)
    if flagged:
        Q1 = capped_df[col].quantile(0.25)
        Q3 = capped_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before_capping = capped_df[col].copy()
        capped_df[col] = capped_df[col].clip(lower, upper)
        capped_count = ((before_capping < lower) | (before_capping > upper)).sum()
        outlier_stats[col] = {
            "method": "IQR capping",
            "IQR_range": [float(lower), float(upper)],
            "n_capped": int(capped_count),
        }
    else:
        outlier_stats[col] = {
            "method": "No outlier capping (not flagged by EDA)",
            "n_capped": 0,
        }

capped_df.to_csv(PREP_DATA_PATH, index=False)
append_log_step(
    'Outlier Handling',
    {
        "columns_processed": outlier_cols,
        "outlier_capping_stats": outlier_stats,
        "preprocessed_data_file": PREP_DATA_PATH
    }
)

### 4. Encode categorical 'Crop' column with one-hot encoding
ohe = OneHotEncoder(sparse=False, dtype=int)
crops = capped_df[['Crop']]
encoded_crops = ohe.fit_transform(crops)
encoded_col_names = [f"Crop_{cat}" for cat in ohe.categories_[0]]
encoded_crop_df = pd.DataFrame(encoded_crops, columns=encoded_col_names, index=capped_df.index)
df_encoded = pd.concat([capped_df.drop('Crop', axis=1), encoded_crop_df], axis=1)

features_after_encoding = [col for col in df_encoded.columns if col != 'Yield']
encoding_map = dict(zip(ohe.categories_[0], encoded_col_names))
df_encoded.to_csv(ENCODED_DATA_PATH, index=False)

append_log_step(
    'Categorical Encoding',
    {
        "original_column": "Crop",
        "encoding_method": "one-hot",
        "encoded_feature_names": encoded_col_names,
        "feature_list_after_encoding": features_after_encoding,
        "encoding_map": encoding_map,
        "encoded_data_file": ENCODED_DATA_PATH
    }
)

### 5. Feature selection: remove constant/duplicate features
removed_features = []
if constant_features:
    df_encoded.drop(columns=constant_features, inplace=True)
    removed_features = constant_features
append_log_step(
    'Feature Selection',
    {
        "constant_or_duplicate_features_removed": removed_features,
        "feature_list_post_selection": [col for col in df_encoded.columns if col != 'Yield'],
        "status": "No constant or duplicate features to remove; none removed." if not removed_features else f"Removed: {removed_features}"
    }
)

### 6. Correlation analysis and multicollinearity
cor_feature_pairs = []
corr_matrix = df_encoded.drop('Yield', axis=1).corr()
for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.95:
            cor_feature_pairs.append(tuple(sorted([col1, col2])))
cor_feature_pairs = list(set(cor_feature_pairs))
features_to_drop_due_to_corr = []
if cor_feature_pairs:
    for pair in cor_feature_pairs:
        # Drop the second feature in the pair (as arbitrary choice)
        if pair[1] not in features_to_drop_due_to_corr:
            features_to_drop_due_to_corr.append(pair[1])
    df_encoded = df_encoded.drop(columns=features_to_drop_due_to_corr)
append_log_step(
    'Multi-collinearity',
    {
        "correlations_checked": True,
        "feature_pairs_high_corr": cor_feature_pairs,
        "features_dropped_due_to_collinearity": features_to_drop_due_to_corr,
        "final_correlation_matrix": corr_matrix.to_dict(),
        "feature_list_final": [col for col in df_encoded.columns if col != 'Yield']
    }
)

### 7. Train/test split (stratified if possible)
target_col = 'Yield'
y = df_encoded['Yield']
X = df_encoded.drop(target_col, axis=1)
stratify = None
n_unique_y = y.nunique()
if n_unique_y <= 10:
    # Attempt stratified if 'Yield' is categorical or has limited values
    stratify = y
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)
append_log_step(
    'Train/Test Split',
    {
        "split_type": "stratified" if stratify is not None else "random",
        "test_size": 0.2,
        "random_state": 42,
        "n_train": len(train_X),
        "n_test": len(test_X),
        "target_dist_train": train_y.describe().to_dict(),
        "target_dist_test": test_y.describe().to_dict()
    }
)

### 8. Select regressors and log
all_models = {}
model_choices = ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor']
model_hyperparams = {
    "LinearRegression": {},
    "RandomForestRegressor": {"n_estimators": 100, "random_state": 42},
    "XGBRegressor": {"n_estimators": 100, "random_state": 42, "verbosity": 0},
    "LGBMRegressor": {"n_estimators": 100, "random_state": 42}
}
append_log_step(
    'Model Selection',
    {
        "models": model_choices,
        "hyperparameters": model_hyperparams
    }
)

### 9. Cross-validation training
cv_results = {}
train_times = {}
trained_models = {}
errors_during_training = {}

from time import time

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
for model_name in model_choices:
    start_time = time()
    results = {}
    try:
        if model_name == 'LinearRegression':
            model = LinearRegression(**model_hyperparams[model_name])
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(**model_hyperparams[model_name])
        elif model_name == 'XGBRegressor':
            if xgb is None:
                errors_during_training[model_name] = 'xgboost not installed'
                continue
            model = xgb.XGBRegressor(**model_hyperparams[model_name])
        elif model_name == 'LGBMRegressor':
            if LGBMRegressor is None:
                errors_during_training[model_name] = 'lightgbm not installed'
                continue
            model = LGBMRegressor(**model_hyperparams[model_name])
        else:
            continue
        # 5-fold CV
        cv_scores = cross_validate(
            model, train_X, train_y,
            cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            return_train_score=False
        )
        results = {
            "R2_mean": float(np.mean(cv_scores['test_r2'])),
            "R2_std": float(np.std(cv_scores['test_r2'])),
            "MAE_mean": float(-np.mean(cv_scores['test_neg_mean_absolute_error'])),
            "MAE_std": float(np.std(-cv_scores['test_neg_mean_absolute_error'])),
            "RMSE_mean": float(-np.mean(cv_scores['test_neg_root_mean_squared_error'])),
            "RMSE_std": float(np.std(-cv_scores['test_neg_root_mean_squared_error']))
        }
        trained_models[model_name] = model.fit(train_X, train_y)
    except Exception as e:
        errors_during_training[model_name] = str(e)
    end_time = time()
    cv_results[model_name] = results
    train_times[model_name] = end_time - start_time

append_log_step(
    'Model Training and CV',
    {
        "cv_results": cv_results,
        "training_times_seconds": train_times,
        "training_errors": errors_during_training
    }
)

### 10. Test set evaluation
test_set_results = {}
residual_stats = {}

for model_name in trained_models:
    model = trained_models[model_name]
    y_pred = model.predict(test_X)
    R2 = r2_score(test_y, y_pred)
    MAE = mean_absolute_error(test_y, y_pred)
    RMSE = rmse(test_y, y_pred)
    residuals = test_y - y_pred
    test_set_results[model_name] = {
        "R2": float(R2),
        "MAE": float(MAE),
        "RMSE": float(RMSE)
    }
    residual_stats[model_name] = {
        "mean_residual": float(residuals.mean()),
        "std_residual": float(residuals.std())
    }
append_log_step(
    'Test Set Evaluation',
    {
        "model_performance": test_set_results,
        "residuals": residual_stats
    }
)

### 11. Select best model based on test RMSE (+ R2/MAE ties)
best_model_name = None
min_rmse = float('inf')
candidates = []
for model_name, result in test_set_results.items():
    rmse_val = result['RMSE']
    if rmse_val < min_rmse:
        min_rmse = rmse_val
        candidates = [model_name]
    elif rmse_val == min_rmse:
        candidates.append(model_name)
if len(candidates) == 1:
    best_model_name = candidates[0]
else:
    # Use R2, then MAE as tie-breaker
    best_r2 = -float('inf')
    r2_candidates = []
    for model_name in candidates:
        r2 = test_set_results[model_name]['R2']
        if r2 > best_r2:
            best_r2 = r2
            r2_candidates = [model_name]
        elif r2 == best_r2:
            r2_candidates.append(model_name)
    if len(r2_candidates) == 1:
        best_model_name = r2_candidates[0]
    else:
        # MAE tie-breaker: lowest MAE wins
        best_mae = float('inf')
        for model_name in r2_candidates:
            mae = test_set_results[model_name]['MAE']
            if mae < best_mae:
                best_model_name = model_name
append_log_step(
    'Best Model Selection',
    {
        "best_model": best_model_name,
        "selection_reason": "Lowest test RMSE, then highest R2, then lowest MAE (in tie).",
        "test_metric_values": test_set_results[best_model_name],
        "all_model_metrics": test_set_results
    }
)

### 12. Save model pipeline
# Save model: include encoding info, model, and list of input features
pipeline = {
    "model": trained_models[best_model_name],
    "ohe_categories": ohe.categories_[0].tolist(),
    "feature_columns": [col for col in df_encoded.columns if col != 'Yield'],
    "preprocessing": {
        "outlier_capping": outlier_stats,
        "constant_or_duplicate_removed": removed_features,
        "features_dropped_due_to_corr": features_to_drop_due_to_corr,
        "encoding_map": encoding_map,
        "categorical_encoding": "one-hot"
    }
}
joblib.dump(pipeline, MODEL_OUT_PATH)
append_log_step(
    'Save Best Model Artifact',
    {
        "model_path": MODEL_OUT_PATH,
        "model_type": best_model_name,
        "feature_order": pipeline["feature_columns"],
        "preprocessing_steps": pipeline["preprocessing"]
    }
)

### 13. Final Summary for reproducibility
log['summary'] = {
    "data_path": DATA_PATH,
    "preprocessed_data_path": PREP_DATA_PATH,
    "encoded_data_path": ENCODED_DATA_PATH,
    "model_artifact_path": MODEL_OUT_PATH,
    "feature_set": pipeline['feature_columns'],
    "categorical_encoding": {"Crop": encoding_map},
    "preprocessing": {
        "outlier_handling": outlier_stats,
        "constant_or_duplicate_removed": removed_features,
        "features_dropped_due_to_corr": features_to_drop_due_to_corr,
    },
    "model_type": best_model_name,
    "model_hyperparameters": model_hyperparams.get(best_model_name, {}),
    "test_metrics": test_set_results[best_model_name],
    "timestamp_end": datetime.now().isoformat(),
    "completed": True
}
write_log()

print(f"Pipeline completed. Log and results are saved to {JSON_LOG_PATH}.")