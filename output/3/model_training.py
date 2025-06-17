import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Logging and utility functions ---

LOG_PATH = './output/3/model_training.json'
os.makedirs('./output/3/', exist_ok=True)

def log_step(log_list, step_name, results_dict):
    '''Logs a step in the JSON process log.'''
    entry = {
        'step': step_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results_dict
    }
    log_list.append(entry)
    
def save_log(log_list):
    with open(LOG_PATH, 'w') as f:
        json.dump(log_list, f, indent=2)

# --- 1. Read EDA output and log summary ---

log = []
dfa = None
try:
    with open('./output/3/eda_agent.json', 'r') as f:
        eda = json.load(f)
except Exception as e:
    raise RuntimeError(f"EDA file could not be read: {str(e)}")

eda_summary = {
    'data_types': eda.get('data_types'),
    'missing_values': eda.get('missing_values'),
    'outliers': eda.get('outliers'),
    'cardinalities': eda.get('cardinalities'),
    'correlation_matrix': eda.get('correlation_matrix'),
    'target_metadata': eda.get('target_metadata'),
    'feature_recommendations': eda.get('feature_recommendations'),
}
log_step(log, 'EDA summary loaded', eda_summary)
save_log(log)

# --- 2. Validate CSV matches EDA summary (columns, types, row count) ---

df = pd.read_csv('data/Crop_Yield_Prediction.csv')
log_validation = {}

# a) Columns
eda_columns = list(eda['data_types'].keys())
df_columns = list(df.columns)
log_validation['columns_match'] = eda_columns == df_columns
log_validation['eda_columns'] = eda_columns
log_validation['csv_columns'] = df_columns
if not log_validation['columns_match']:
    log_validation['column_diff'] = {
        'in_eda_not_csv': sorted(list(set(eda_columns) - set(df_columns))),
        'in_csv_not_eda': sorted(list(set(df_columns) - set(eda_columns))),
    }

# b) Dtypes
df_types = df.dtypes.apply(lambda x: x.name).to_dict()
log_validation['dtypes_match'] = all(eda['data_types'][c] == df_types[c] for c in eda_columns if c in df_types)
log_validation['eda_types'] = eda['data_types']
log_validation['csv_types'] = df_types

# c) Row count
log_validation['row_count_match'] = eda['target_metadata'].get('n_rows') == len(df)
log_validation['eda_row_count'] = eda['target_metadata'].get('n_rows')
log_validation['csv_row_count'] = len(df)

log_step(log, 'Dataset validation', log_validation)
save_log(log)

# --- 3. Confirm no missing values (preprocessing check) ---

missing_check = {}
missing_check['has_missing_values'] = df.isnull().any().to_dict()
missing_check['rows_with_any_missing'] = int(df.isnull().any(axis=1).sum())
missing_check['all_features_no_missing'] = not df.isnull().any().any()
log_step(log, 'Missing values check', missing_check)
save_log(log)

# --- 4. Remove features flagged in EDA recommendations ---

target_column = eda['target_metadata']['target_column'] if 'target_column' in eda['target_metadata'] else 'Yield'
removal_reasons = eda['feature_recommendations'].get('remove', [])
info_leaks = eda['feature_recommendations'].get('info_leakage', [])
to_remove = set(removal_reasons + info_leaks)

selected_features = [col for col in eda_columns if col != target_column and col not in to_remove]
log_features = {'selected_features': selected_features, 'removed_features': sorted(list(to_remove)), 'target_column': target_column}
log_step(log, 'Feature selection based on EDA recommendations', log_features)
save_log(log)

# --- 5. Inspect Yield distribution, skewness, kurtosis for transformation plan ---

target_stats = eda['target_metadata'].get('distribution_stats', {})
transformation_plan = {
    'skewness': target_stats.get('skew', None),
    'kurtosis': target_stats.get('kurtosis', None),
    'transformation_needed': False,
    'recommended_transformation': None,
}
if target_stats:
    skew = float(target_stats.get('skew', 0))
    kurt = float(target_stats.get('kurtosis', 0))
    if abs(skew) > 2 or abs(kurt) > 10:
        transformation_plan['transformation_needed'] = True
        transformation_plan['recommended_transformation'] = 'log1p'
log_step(log, "Target ('Yield') distribution & transformation plan", transformation_plan)
save_log(log)

# Optional: apply transformation for further modeling if needed
def transform_target(y, plan):
    return np.log1p(y) if plan and plan.get('transformation_needed') else y

def inverse_transform_target(y, plan):
    return np.expm1(y) if plan and plan.get('transformation_needed') else y

# --- 6. Outlier treatment / robust scaling for numeric features with outliers ---

outliers_info = eda.get('outliers', {})
scaler_dict = {}
scaling_applied = {}
df_for_model = df.copy()
for col in selected_features:
    if col in outliers_info:
        outlier_indices = outliers_info[col].get('indices', [])
        n_outliers = len(outlier_indices)
        n_total = len(df)
        if n_outliers > 0 and n_outliers / n_total > 0.01: # >1% outliers
            # Apply robust scaler for this column
            scaler = RobustScaler()
            arr = df[[col]].values
            arr_scaled = scaler.fit_transform(arr)
            df_for_model[col] = arr_scaled
            scaler_dict[col] = scaler
            scaling_applied[col] = 'RobustScaler'
        else:
            scaling_applied[col] = 'none'
    else:
        scaling_applied[col] = 'none'

log_scaling = {
    'scaling_applied': scaling_applied,
    'outlier_columns': [col for col, v in scaling_applied.items() if v != 'none'],
}
# Save scaler(s)
if scaler_dict:
    joblib.dump(scaler_dict, './output/3/scaler.pkl')
log_step(log, 'Feature scaling/outlier treatment', log_scaling)
save_log(log)

# --- 7. Encode categorical column "Crop" ---

cat_col = 'Crop'
eda_cat_associations = eda.get('cat_num_associations', {})
cat_cardinality = eda.get('cardinalities', {}).get(cat_col, len(df[cat_col].unique()))
encoding_result = {'encoding': None, 'mapping': None}
if cat_col in selected_features:
    # Strategy: 
    # - If cardinality <=8 --> one-hot
    # - Else if strong association to target --> target (mean) encoding
    # - Else ordinal encoding as fallback
    if cat_cardinality <= 8:
        encoder = OneHotEncoder(cols=[cat_col], use_cat_names=True)
        df_for_model = encoder.fit_transform(df_for_model)
        encoding_result['encoding'] = 'onehot'
        encoding_result['mapping'] = encoder.get_feature_names()
        joblib.dump(encoder, './output/3/encoder.pkl')
    elif eda_cat_associations.get(cat_col, {}).get(target_column, 0) > 0.2:  # arbitrary threshold of 'strong'
        encoder = TargetEncoder(cols=[cat_col])
        df_for_model[cat_col] = encoder.fit_transform(df_for_model[cat_col], df_for_model[target_column])
        encoding_result['encoding'] = 'target'
        encoding_result['mapping'] = dict(zip(
            encoder.mapping.index if hasattr(encoder.mapping, 'index') else [],
            encoder.mapping['mean'] if 'mean' in encoder.mapping else []
        )) # May need to adjust depending on encoder output
        joblib.dump(encoder, './output/3/encoder.pkl')
    else:
        encoder = OrdinalEncoder(cols=[cat_col])
        df_for_model[cat_col] = encoder.fit_transform(df_for_model[cat_col])
        encoding_result['encoding'] = 'ordinal'
        encoding_result['mapping'] = dict(encoder.ordinal_mapping[0]) if hasattr(encoder, 'ordinal_mapping') else None
        joblib.dump(encoder, './output/3/encoder.pkl')
    # Remove 'Crop' from selected_features if using one-hot encoding
    if encoding_result['encoding'] == 'onehot':
        selected_features = [f for f in selected_features if f != cat_col]
        selected_features += [c for c in df_for_model.columns if c.startswith(cat_col + '_')]
        
log_step(log, f'Categorical encoding for "{cat_col}"', encoding_result)
save_log(log)

# --- 8. High correlation analysis with Yield, remove redundant predictors ---

corr_matrix = pd.DataFrame(eda['correlation_matrix'])
correlation_info = {'removals': [], 'details': []}
to_keep = selected_features.copy()
for col in selected_features:
    if col in corr_matrix.index and target_column in corr_matrix.columns:
        corr_val = corr_matrix.loc[col, target_column]
        # Check for highly correlated (>|0.9|) non-target predictors
        if abs(corr_val) > 0.9 and col != target_column:
            correlation_info['removals'].append(col)
            to_keep.remove(col)
        correlation_info['details'].append({'feature': col, 'corr_with_target': corr_val})
selected_features = to_keep
correlation_info['final_selected_features'] = selected_features
log_step(log, "Feature correlation & redundancy check", correlation_info)
save_log(log)

# --- 9. Train-test split (stratify on 'Crop' if possible) ---

y_raw = df[target_column].values
y = transform_target(y_raw, transformation_plan)
X = df_for_model[selected_features].values

# Stratified if possible (stratify for >1 group and group not unique)
can_stratify = (cat_col in df.columns) and (1 < df[cat_col].nunique() < len(df))
train_idx, test_idx = None, None
split_info = {}
if can_stratify:
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, 
                                      stratify=df[cat_col], random_state=42)
    split_info['stratified'] = True
else:
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, 
                                           random_state=42)
    split_info['stratified'] = False

split_info['train_indices'] = train_idx.tolist()
split_info['test_indices'] = test_idx.tolist()
split_info['random_state'] = 42
log_step(log, "Train-test split", split_info)
save_log(log)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --- 10. Define candidate regression models with hyperparameters ---

models_dict = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {} # No hyperparams for vanilla LR
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100],
            'max_depth': [None, 10]
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100],
            'max_depth': [3, 7]
        }
    },
    'ExtraTreesRegressor': {
        'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100],
            'max_depth': [None, 10]
        }
    }
}
model_list_log = []
for name, obj in models_dict.items():
    model_list_log.append({'name': name, 'params': obj['params']})
log_step(log, "Model candidates and parameters", {'models': model_list_log})
save_log(log)

# --- 11. Model training: grid search or default ---

from collections import defaultdict
training_results = []
for name, model_dict in models_dict.items():
    print(f"Training {name}...")
    st_time = time.time()
    # If no param grid, fit directly; else use GridSearchCV (basic grid)
    model = model_dict['model']
    param_grid = model_dict['params'] if model_dict['params'] else None
    if param_grid:
        cv = 3 if X_train.shape[0] > 32 else 2
        gs = GridSearchCV(model, param_grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        params_used = gs.best_params_
        cv_score = -gs.best_score_
    else:
        model.fit(X_train, y_train)
        best_model = model
        params_used = {}
        cv_score = None
    et_time = time.time()
    train_pred = best_model.predict(X_train)
    val_r2 = r2_score(y_train, train_pred)
    val_rmse = mean_squared_error(y_train, train_pred, squared=False)
    val_mae = mean_absolute_error(y_train, train_pred)
    record = {
        'model': name,
        'params': params_used,
        'cv_score': cv_score,
        'train_time_sec': et_time - st_time,
        'train_r2': val_r2,
        'train_rmse': val_rmse,
        'train_mae': val_mae,
        'cls_saved': False # track in next step
    }
    # Save fitted model objects for later
    model_save_path = f'./output/3/{name}_fitted.pkl'
    joblib.dump(best_model, model_save_path)
    record['cls_saved'] = model_save_path
    training_results.append(record)

log_step(log, 'Model training and validation', {'training_results': training_results})
save_log(log)

# --- 12. Evaluate models on test set ---

eval_results = defaultdict(dict)
metrics_used = ['R2', 'RMSE', 'MAE']
for res in training_results:
    model_name = res['model']
    model_path = res['cls_saved']
    model = joblib.load(model_path)
    y_pred_test = model.predict(X_test)
    # Reverse transform if target was transformed
    y_pred_final = inverse_transform_target(y_pred_test, transformation_plan)
    y_true_final = inverse_transform_target(y_test, transformation_plan)
    eval_results[model_name]['R2'] = float(r2_score(y_true_final, y_pred_final))
    eval_results[model_name]['RMSE'] = float(mean_squared_error(y_true_final, y_pred_final, squared=False))
    eval_results[model_name]['MAE'] = float(mean_absolute_error(y_true_final, y_pred_final))

log_step(log, "Test set evaluation", {'metrics_used': metrics_used, 'results': dict(eval_results)})
save_log(log)

# --- 13. Model selection based on RMSE or R2 ---

primary_metric = 'RMSE'
# Lower RMSE is better
best_model = min(eval_results.items(), key=lambda x: x[1][primary_metric])
selection = {
    'selected_model': best_model[0],
    'selected_metrics': best_model[1],
    'selection_metric': primary_metric
}
log_step(log, "Best model selection", selection)
save_log(log)

# --- 14. Save final model, pipeline objects, test predictions ---

final_model = joblib.load(f'./output/3/{selection["selected_model"]}_fitted.pkl')
final_model_path = './output/3/final_model.pkl'
joblib.dump(final_model, final_model_path)

# Test predictions
test_predictions = {
    'index': test_idx.tolist(),
    'y_true': inverse_transform_target(y_test, transformation_plan).tolist(),
    'y_pred': inverse_transform_target(final_model.predict(X_test), transformation_plan).tolist()
}
pd.DataFrame(test_predictions).to_csv('./output/3/test_predictions.csv', index=False)

# Log all output paths
output_files = {
    'model_path': final_model_path,
    'scaler_path': './output/3/scaler.pkl' if scaler_dict else None,
    'encoder_path': './output/3/encoder.pkl' if os.path.exists('./output/3/encoder.pkl') else None,
    'test_predictions_csv': './output/3/test_predictions.csv'
}
log_step(log, "Saving final outputs", output_files)
save_log(log)

print(f'Process completed, all results logged to: {LOG_PATH}')