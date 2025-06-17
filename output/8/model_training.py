import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Optional: try to import XGBoost/LightGBM; fall back to GradientBoostingRegressor if not available
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    has_xgb = False

# --- Helper functions for JSON logging ---

def safe_write_json(filepath, data):
    """Write data dict to JSON file nicely."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def log_to_json(json_path, key, content):
    """Append/Update a key with content in JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = dict()
    all_data[key] = content
    safe_write_json(json_path, all_data)


def append_to_json(json_path, entries_dict):
    """Update multiple values in JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = dict()
    all_data.update(entries_dict)
    safe_write_json(json_path, all_data)

# --- Create output directory ---

os.makedirs('./output/8/', exist_ok=True)
json_log_path = './output/8/model_training.json'


# ==== 1. Load EDA results ====
eda_path = './output/8/eda_agent.json'
with open(eda_path, 'r') as f:
    eda_results = json.load(f)
log_to_json(json_log_path, 'eda_overview', eda_results)


# ==== 2. Load data & verify schema ====
csv_path = 'data/Crop_Yield_Prediction.csv'
df = pd.read_csv(csv_path)
schema_eda = sorted(eda_results.get('columns', eda_results.get('schema', {})).keys()
                    if 'columns' in eda_results or 'schema' in eda_results
                    else df.columns.tolist())
columns_dataset = df.columns.tolist()

schema_consistent = set(schema_eda) == set(columns_dataset)

data_schema = {
    "schema_consistent": schema_consistent,
    "eda_columns": schema_eda,
    "data_columns": columns_dataset,
    "shape": list(df.shape),
    "dtypes": df.dtypes.astype(str).to_dict()
}
log_to_json(json_log_path, 'data_schema', data_schema)


# ==== 3. Missing Value Check ====
has_missing_eda = None
if 'missing_value_summary' in eda_results:
    missing_df = pd.DataFrame.from_dict(eda_results['missing_value_summary'], orient='index')
    has_missing_eda = (missing_df['count'] > 0).any()
elif 'missing' in eda_results:
    has_missing_eda = eda_results['missing'].get('total_missing', 0) > 0
else:
    has_missing_eda = df.isnull().any().any()

missing_in_data = df.isnull().any().any()
missing_result = {
    "missing_as_per_eda": not has_missing_eda,
    "missing_in_data": not missing_in_data,
    "columns_with_missing": df.columns[df.isnull().any()].tolist()
}
log_to_json(json_log_path, 'missing_value_check', missing_result)


# ==== 4. Duplicate Rows Check ====
num_dupes = df.duplicated().sum()
dupes_as_per_eda = eda_results.get('duplicates', 0)
duplicate_result = {
    "duplicates_as_per_eda": dupes_as_per_eda,
    "duplicates_in_data": int(num_dupes)
}
log_to_json(json_log_path, 'duplicate_check', duplicate_result)


# ==== 5. Outlier Analysis (YIELD) ====
yield_col = 'Yield'
outlier_indices = df[(df[yield_col] > 100_000) | (df[yield_col] < 10)].index.tolist()
eda_outlier_report = eda_results.get('outlier_report', {})
outlier_summary = {
    "defined_criteria": "Yield > 100,000 or Yield < 10",
    "flagged_indices": outlier_indices,
    "num_flagged": len(outlier_indices),
    "eda_outlier_report": eda_outlier_report,
    "outlier_stats": {
        "count_extreme_high": int((df[yield_col] > 100_000).sum()),
        "count_extreme_low": int((df[yield_col] < 10).sum()),
        "min_Yield": float(df[yield_col].min()),
        "max_Yield": float(df[yield_col].max()),
        "mean_Yield": float(df[yield_col].mean()),
        "std_Yield": float(df[yield_col].std())
    }
}
log_to_json(json_log_path, 'outlier_analysis', outlier_summary)


# ==== 6. Data Cleaning (remove/cap outliers), Save cleaned CSV ====
if outlier_indices:
    # Cap outliers instead of drop (Alternatively, df = df.drop(index=outlier_indices))
    capped_df = df.copy()
    capped_df.loc[capped_df[yield_col] > 100_000, yield_col] = 100_000
    capped_df.loc[capped_df[yield_col] < 10, yield_col] = 10
    cleaning_steps = {
        "extreme_outliers": True,
        "method": "Winsorization: capped extreme values to 10/100,000",
        "flagged_indices": outlier_indices,
        "post_capping_stats": capped_df[yield_col].describe(percentiles=[.01,.99]).apply(float).to_dict()
    }
else:
    capped_df = df.copy()
    cleaning_steps = {
        "extreme_outliers": False,
        "method": "None needed (no extremes)",
        "flagged_indices": [],
        "post_capping_stats": capped_df[yield_col].describe(percentiles=[.01,.99]).apply(float).to_dict()
    }
capped_df.to_csv('./output/8/cleaned_data.csv', index=False)
log_to_json(json_log_path, 'cleaning_steps', cleaning_steps)


# ==== 7. Feature Engineering: One-hot, Robust Scale ====
# Infer categorical/numerical columns from data/EDA
categorical_features = ['Crop']
numerical_features = [col for col in capped_df.columns if col not in [yield_col] + categorical_features]

feature_pipeline = ColumnTransformer([
    ('num', RobustScaler(), numerical_features),
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
])

pipeline_desc = {
    "input_features": numerical_features + categorical_features,
    "categorical": categorical_features,
    "one_hot_encoded": categorical_features,
    "numerical": numerical_features,
    "scaler": "RobustScaler (for all numeric columns, to handle outliers)",
    "transformer_pipeline": ["one-hot on Crop", "robust-scaling all numericals"],
}
log_to_json(json_log_path, 'feature_engineering', pipeline_desc)


# ==== 8. Remove constant, quasi-constant, and highly correlated features ====
# 8.1 Constant/quasi-constant
constant_features = [col for col in capped_df.columns if capped_df[col].nunique() == 1]
quasi_constant_features = [col for col in capped_df.columns if 1 < capped_df[col].nunique() < max(10, int(0.01 * len(capped_df)))]

# 8.2 Highly correlated
corr_matrix = capped_df[numerical_features].corr().abs()
high_corr_pairs = [(col, other)
                   for col in corr_matrix.columns
                   for other in corr_matrix.columns
                   if col != other and corr_matrix.loc[col, other] > 0.98]
high_corr_features = list(set(j for i, j in high_corr_pairs))

excluded_features = list(set(constant_features + quasi_constant_features + high_corr_features))
selected_numerical = [col for col in numerical_features if col not in excluded_features]

final_features = selected_numerical + categorical_features
selected_features_log = {
    "constant": constant_features,
    "quasi_constant": quasi_constant_features,
    "highly_corr_above_0.98": high_corr_features,
    "all_excluded": excluded_features,
    "final_feature_list": final_features
}
log_to_json(json_log_path, 'selected_features', selected_features_log)


# ==== 9. Train/Test Split ====
X = capped_df[final_features]
y = capped_df[yield_col]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
indices_info = {
    "train_indices": list(Xtrain.index),
    "test_indices": list(Xtest.index),
    "train_size": int(Xtrain.shape[0]),
    "test_size": int(Xtest.shape[0])
}
log_to_json(json_log_path, 'data_split', indices_info)


# ==== 10. Regression Metrics ====
metrics_used = ["R2", "RMSE", "MAE"]
log_to_json(json_log_path, 'evaluation_metrics', metrics_used)


# ==== 11-12. Train 3 Models, Record Pipelines, 5-CV scores ====
model_training_runs = {}
cv_results = {}
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=1)

# Common feature preprocessor
preproc = ColumnTransformer([
    ('num', RobustScaler(), selected_numerical),
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
])

model_defs = {
    "ridge": {
        "model": Ridge(alpha=1.0, random_state=42),
        "name": "Ridge Regression",
        "params": {"alpha": 1.0, "random_state": 42}
    },
    "rf": {
        "model": RandomForestRegressor(n_estimators=100, random_state=42),
        "name": "Random Forest Regressor",
        "params": {"n_estimators": 100, "random_state": 42}
    }
}
if has_xgb:
    model_defs["xgb"] = {
        "model": XGBRegressor(n_estimators=100, random_state=42, eval_metric="rmse", n_jobs=1, verbosity=0),
        "name": "XGBoost Regressor",
        "params": {"n_estimators": 100, "random_state":42, "eval_metric":"rmse"}
    }
else:
    model_defs["gbr"] = {
        "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "name": "GradientBoostingRegressor",
        "params": {"n_estimators": 100, "random_state":42}
    }

for model_key, meta in model_defs.items():
    pipe = Pipeline([
        ('preproc', preproc),
        ('regressor', meta["model"])
    ])

    # 5-fold cross validation
    scores_r2 = cross_val_score(pipe, Xtrain, ytrain, cv=cv, scoring='r2')
    scores_rmse = -cross_val_score(pipe, Xtrain, ytrain, cv=cv, scoring='neg_root_mean_squared_error')
    scores_mae = -cross_val_score(pipe, Xtrain, ytrain, cv=cv, scoring='neg_mean_absolute_error')

    meta_result = {
        "model_name": meta["name"],
        "hyperparameters": meta["params"],
        "feature_pipeline": pipeline_desc,
        "selected_input_features": final_features
    }
    model_training_runs[model_key] = meta_result

    # Log CV results
    cv_results[model_key] = {
        "R2": {"mean": float(np.mean(scores_r2)), "std": float(np.std(scores_r2)), "fold_scores": [float(x) for x in scores_r2]},
        "RMSE": {"mean": float(np.mean(scores_rmse)), "std": float(np.std(scores_rmse)), "fold_scores": [float(x) for x in scores_rmse]},
        "MAE": {"mean": float(np.mean(scores_mae)), "std": float(np.std(scores_mae)), "fold_scores": [float(x) for x in scores_mae]},
    }

append_to_json(json_log_path, {
    "model_training_runs": model_training_runs,
    "cv_results": cv_results
})

# ==== 13. Test Evaluation for Each Model ====
test_performance = {}

for model_key, meta in model_defs.items():
    pipe = Pipeline([
        ('preproc', preproc),
        ('regressor', meta["model"])
    ])
    pipe.fit(Xtrain, ytrain)
    ypred = pipe.predict(Xtest)
    test_r2 = float(r2_score(ytest, ypred))
    test_rmse = float(mean_squared_error(ytest, ypred, squared=False))
    test_mae = float(mean_absolute_error(ytest, ypred))
    test_performance[model_key] = {
        "R2": test_r2,
        "RMSE": test_rmse,
        "MAE": test_mae
    }
append_to_json(json_log_path, {"test_performance": test_performance})


# ==== 14. Select Best Model ====
best_model_key = sorted(test_performance, key=lambda k: (-test_performance[k]['R2'], test_performance[k]['RMSE']))[0]
best_model_details = model_training_runs[best_model_key]
best_model_perf = test_performance[best_model_key]
best_model_details['test_performance'] = best_model_perf

append_to_json(json_log_path, {'best_model': {
    "model_key": best_model_key,
    "description": best_model_details,
    "test_set_performance": best_model_perf
}})

# ==== 15. Save model artifact + schema summary ====
# Refit best pipeline on all data (not just train; or use train if holdout needed)
final_pipe = Pipeline([
    ('preproc', preproc),
    ('regressor', model_defs[best_model_key]['model'])
])
final_pipe.fit(X, y)
model_artifact = './output/8/best_model.pkl'
joblib.dump(final_pipe, model_artifact)

pipeline_io_schema = {
    "input_features": final_features,
    "categorical": categorical_features,
    "one_hot_encoded": categorical_features,
    "numerical": selected_numerical,
    "output": yield_col,
    "expected_input_dtypes": {col: str(df[col].dtype) for col in final_features},
    "output_dtype": str(df[yield_col].dtype)
}
append_to_json(json_log_path, {
    'model_artifact': {
        "file_path": model_artifact,
        "pipeline_io_schema": pipeline_io_schema
    }
})

print("All steps complete. Outputs written to ./output/8/model_training.json and ./output/8/best_model.pkl.")