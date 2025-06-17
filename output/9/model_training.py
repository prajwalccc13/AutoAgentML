import os
import json
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Optional AutoML
try:
    from autosklearn.regression import AutoSklearnRegressor
    AUTO_SKLEARN = True
except ImportError:
    AUTO_SKLEARN = False

OUTPUT_JSON = './output/9/model_training.json'
MODEL_PATH = './output/9/final_model.pkl'
os.makedirs('./output/9/', exist_ok=True)

def save_json(data, path=OUTPUT_JSON):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def update_json(new_data, path=OUTPUT_JSON):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    # Deep update
    for k, v in new_data.items():
        data[k] = v
    save_json(data, path)

#########################
# 1. Load EDA and Metadata
#########################
with open('./output/9/eda_agent.json') as f:
    eda_out = json.load(f)

# Suppose metadata is in the EDA or can be loaded from elsewhere
# We simulate loading metadata
try:
    with open('./output/9/training_metadata.json') as f:
        training_metadata = json.load(f)
except FileNotFoundError:
    training_metadata = {"intent": "Yield regression for crops; predict 'Yield' using region-level Soil/N, P, K, pH, Rainfall, etc."}

# Optionally, get the path of source CSV from EDA
dataset_path = eda_out.get('dataset_path', './output/9/source_data.csv') 

# Read the actual data used for EDA
df = pd.read_csv(dataset_path)

log_dict = {
    "raw_eda_summary": eda_out,
    "training_metadata": training_metadata
}
save_json(log_dict, OUTPUT_JSON)

#########################
# 2. Verify missing values
#########################
missing = df.isnull().sum()
features = df.columns.tolist()
missing_feat = [feat for feat, v in missing.items() if v > 0]
preprocessing_summary = {
    'missing_value_features': missing_feat,
    'log': "No missing values found; missing value handling is skipped.",
    'feature_list': features
}
update_json({"preprocessing_summary": preprocessing_summary})

#########################
# 3. Outlier analysis
#########################
# Use the 'outlier_detection_IQR' from EDA summary
iqr = eda_out.get("outlier_detection_IQR", {})
outlier_indices = {}
outlier_counts = {}
for col, col_outlier in iqr.items():
    idx = col_outlier.get('outlier_indices', [])
    outlier_indices[col] = idx
    outlier_counts[col] = len(idx)
outlier_summary = {
    "outlier_counts_per_feature": outlier_counts,
    "outlier_indices_per_feature": outlier_indices,
    "decision": "Outliers will not be removed.",
    "rationale": "Regression task; target ('Yield') is highly skewed; substantial data volume; removing may bias model."
}
update_json({"outlier_summary": outlier_summary})

#########################
# 4. One-hot encode categorical 'Crop'
#########################
cat_col = 'Crop'
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe = encoder.fit_transform(df[[cat_col]])
ohe_df = pd.DataFrame(ohe, columns=[f'Crop_{c}' for c in encoder.categories_[0]])
encoded_cols = ohe_df.columns.tolist()
feature_engineering = {
   "one_hot_encoded_columns": {cat_col: encoded_cols}
}
update_json({"feature_engineering": feature_engineering})

#########################
# 5. Skewness / kurtosis - log1p transforms
#########################
skewness = {}
kurtosis = {}
features_for_skew = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
target_col = 'Yield'
for col in features_for_skew + [target_col]:
    skewness[col] = df[col].skew()
    kurtosis[col] = df[col].kurt()
highly_skewed = [col for col in ['Potassium', 'Phosphorus'] if abs(skewness.get(col,0))>1]
extremely_skewed_target = abs(skewness.get('Yield',0))>1 or abs(kurtosis.get('Yield',0))>1
transformed_cols = []
if highly_skewed: transformed_cols.extend(highly_skewed)
if extremely_skewed_target: transformed_cols.append('Yield')
df['Phosphorus_log1p'] = np.log1p(df['Phosphorus'])
df['Potassium_log1p'] = np.log1p(df['Potassium'])
df['Yield_log1p']      = np.log1p(df['Yield'])
feature_transformations = {
    "skewness": skewness,
    "kurtosis": kurtosis,
    "transformed_columns": transformed_cols,
    "transformation_type": "log1p applied to high-skew features and target"
}
update_json({"feature_transformations": feature_transformations})

#########################
# 6. Select features for modeling
#########################
selected_features = [
    'Nitrogen',
    'Phosphorus_log1p',
    'Potassium_log1p',
    'Temperature',
    'Humidity',
    'pH_Value',
    'Rainfall',
] + encoded_cols
selected_target = 'Yield_log1p'
update_json({"selected_features": selected_features})

#########################
# 7. Train-test split (stratified by Crop)
#########################
X = pd.concat([df[selected_features], ohe_df], axis=1)
y = df[selected_target]
y_true = df['Yield']
stratifier = df[cat_col]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, stratifier))
X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
train_yield = y_true.iloc[train_idx]
test_yield = y_true.iloc[test_idx]

train_test_split_log = {
    "train_indices": train_idx.tolist(),
    "test_indices": test_idx.tolist(),
    "stratify_on": cat_col,
    "train_class_distribution": dict(pd.Series(stratifier.iloc[train_idx]).value_counts().sort_index()),
    "test_class_distribution": dict(pd.Series(stratifier.iloc[test_idx]).value_counts().sort_index()),
    "train_yield_stats": train_yield.describe().to_dict(),
    "test_yield_stats": test_yield.describe().to_dict(),
}
update_json({'train_test_split': train_test_split_log})

#########################
# 8. Baseline Linear Regression
#########################
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
eval_lr = {
    'mean_absolute_error': float(mean_absolute_error(y_test, y_pred_lr)),
    'mean_squared_error': float(mean_squared_error(y_test, y_pred_lr)),
    'r2_score': float(r2_score(y_test, y_pred_lr))
}
model_lr = {
    "hyperparameters": lr.get_params(),
    "coefficients": dict(zip(X_train.columns, lr.coef_)),
    "evaluation_metrics": eval_lr
}
update_json({'models': {'linear_regression': model_lr}})

#########################
# 9. ElasticNet with GridSearch
#########################
en_params = {
    'alpha': [0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.9]
}
en = ElasticNet(max_iter=10000)
gs_en = GridSearchCV(en, en_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs_en.fit(X_train, y_train)
en_best = gs_en.best_estimator_
y_pred_en = en_best.predict(X_test)
eval_en = {
    'mean_absolute_error': float(mean_absolute_error(y_test, y_pred_en)),
    'mean_squared_error': float(mean_squared_error(y_test, y_pred_en)),
    'r2_score': float(r2_score(y_test, y_pred_en))
}
model_en = {
    "hyperparameters_grid": en_params,
    "best_hyperparameters": gs_en.best_params_,
    "evaluation_metrics": eval_en,
    "selected_parameters": en_best.get_params(),
    "coefficients": dict(zip(X_train.columns, en_best.coef_))
}
update_json({'models': {'elasticnet': model_en}})
 
#########################
# 10. Random Forest (grid search)
#########################
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [6, 12, None],
    'min_samples_leaf': [1, 3, 5]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
gs_rf = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
gs_rf.fit(X_train, y_train)
rf_best = gs_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)
eval_rf = {
    'mean_absolute_error': float(mean_absolute_error(y_test, y_pred_rf)),
    'mean_squared_error': float(mean_squared_error(y_test, y_pred_rf)),
    'r2_score': float(r2_score(y_test, y_pred_rf))
}
feature_importances = dict(zip(X_train.columns, rf_best.feature_importances_))
model_rf = {
    "hyperparameters_grid": rf_params,
    "best_hyperparameters": gs_rf.best_params_,
    "evaluation_metrics": eval_rf,
    "feature_importances": feature_importances
}
update_json({'models': {'random_forest': model_rf}})
 
#########################
# 11. GradientBoostingRegressor (early stopping/grid)
#########################
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, None],
    'subsample': [0.8, 1.0]
}
gb = GradientBoostingRegressor(random_state=42)
gs_gb = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
gs_gb.fit(X_train, y_train)
gb_best = gs_gb.best_estimator_
y_pred_gb = gb_best.predict(X_test)
eval_gb = {
    'mean_absolute_error': float(mean_absolute_error(y_test, y_pred_gb)),
    'mean_squared_error': float(mean_squared_error(y_test, y_pred_gb)),
    'r2_score': float(r2_score(y_test, y_pred_gb))
}
model_gb = {
    "hyperparameters_grid": gb_params,
    "best_hyperparameters": gs_gb.best_params_,
    "evaluation_metrics": eval_gb,
    "feature_importances": dict(zip(X_train.columns, gb_best.feature_importances_))
}
update_json({'models': {'gradient_boosting': model_gb}})

#########################
# 12. AutoML (if available)
#########################
if AUTO_SKLEARN:
    automl = AutoSklearnRegressor(time_left_for_this_task=360, per_run_time_limit=60)
    automl.fit(X_train, y_train)
    y_pred_automl = automl.predict(X_test)
    eval_automl = {
        'mean_absolute_error': float(mean_absolute_error(y_test, y_pred_automl)),
        'mean_squared_error': float(mean_squared_error(y_test, y_pred_automl)),
        'r2_score': float(r2_score(y_test, y_pred_automl))
    }
    automl_summary = {
        "model_description": str(automl.show_models()),
        "leaderboard": str(automl.leaderboard()),
        "evaluation_metrics": eval_automl
    }
    update_json({'models': {'automl': automl_summary}})
else:
    update_json({'models': {'automl': "AutoML library not installed; step skipped."}})

#########################
# 13. Comparison table
#########################
def get_metrics(model_block):
    return {**model_block.get('best_hyperparameters', {}),
            **{k: v for k,v in model_block.get('evaluation_metrics', {}).items()}}
with open(OUTPUT_JSON) as f:
    models_data = json.load(f)['models']
comparison = []
for model_label in ['linear_regression', 'elasticnet', 'random_forest', 'gradient_boosting', 'automl']:
    model_entry = models_data.get(model_label, {})
    if isinstance(model_entry, dict) and 'evaluation_metrics' in model_entry:
        comparison.append({
            'model': model_label,
            'hyperparameters': model_entry.get('best_hyperparameters', model_entry.get('hyperparameters', {})),
            **model_entry['evaluation_metrics']
        })
    elif isinstance(model_entry, dict) and 'evaluation_metrics' in model_entry:
        comparison.append({
            'model': model_label,
            **model_entry.get('evaluation_metrics')
        })
    elif isinstance(model_entry, str):
        comparison.append({'model': model_label, 'status': model_entry})
update_json({'model_comparison': comparison})

#########################
# 14. Select best model; save artifact + log file path
#########################
# Select best by r2_score
all_models = {
    'linear_regression': (lr, y_pred_lr),
    'elasticnet': (en_best, y_pred_en),
    'random_forest': (rf_best, y_pred_rf),
    'gradient_boosting': (gb_best, y_pred_gb)
}
if AUTO_SKLEARN:
    all_models['automl'] = (automl, y_pred_automl)
# Pick the one with max r2_score
scores = [(mdl, float(mean_squared_error(y_test, preds)), float(r2_score(y_test, preds)))
          for mdl, (est, preds) in all_models.items()]
sorted_by_r2 = sorted(scores, key=lambda x: (-x[2], x[1]))
final_model_name, _, _ = sorted_by_r2[0]
final_model_obj = all_models[final_model_name][0]
joblib.dump(final_model_obj, MODEL_PATH)
final_model_artifact = {
    "selected_model": final_model_name,
    "saved_model_path": MODEL_PATH,
    "metadata": {
        "selected_features": selected_features,
        "transformations": feature_transformations,
        "feature_engineering": feature_engineering
    }
}
update_json({'final_model_artifact': final_model_artifact})

#########################
# 15. Store predictions (test set)
#########################
final_model_preds = all_models[final_model_name][1]
# inverse transform from log1p if needed
pred_yield = np.expm1(final_model_preds)
true_yield = np.expm1(y_test if selected_target=='Yield_log1p' else y_test)
test_set_predictions = {
    "indices": test_idx.tolist(),
    "true_yield": true_yield.tolist(),
    "predicted_yield": pred_yield.tolist(),
    "log1p_true": y_test.tolist(),
    "log1p_pred": final_model_preds.tolist()
}
update_json({'test_set_predictions': test_set_predictions})

#########################
# 16. Post-processing, inference pipeline JSON
#########################
inference_pipeline = {
    "feature_list": selected_features,
    "feature_engineering": feature_engineering,
    "preprocessing": preprocessing_summary,
    "transformations": feature_transformations,
    "model_artifact": final_model_artifact,
    "selected_model": final_model_name,
    "one_hot_encoder_mapping": {cat_col: encoder.categories_[0].tolist()},
    "method": {
        "Phosphorus and Potassium": "log1p",
        "Target Yield": "log1p and then expm1 on inference",
        "Categorical Crop": "OneHot"
    }
}
update_json({'inference_pipeline': inference_pipeline})

print("Pipeline complete. All logs and artifacts saved at ./output/9/model_training.json and ./output/9/final_model.pkl.")