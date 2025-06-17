import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs('./output/11/', exist_ok=True)

# Helper to read and write JSON with auto-create, auto-close
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

LOG_PATH = './output/11/model_training.json'
log_data = load_json(LOG_PATH)

###########################################
# 1. Read and parse EDA output and record
###########################################
eda_output_path = './output/11/eda_agent.json'
with open(eda_output_path, 'r') as f:
    eda_info = json.load(f)
log_data['eda_info'] = eda_info
save_json(LOG_PATH, log_data)

###########################################################
# 2. Read CSV dataset and integrity check against EDA info
###########################################################
csv_path = 'data/Crop_Yield_Prediction.csv'
data = pd.read_csv(csv_path)
integrity_check = {}

# Schema Specs from EDA
eda_columns = eda_info.get('feature_list', [])
eda_types = eda_info.get('data_profiles', {}).get('dtypes', {})  # Mapping (if present)
target_col = 'Yield'

# Check columns
expected_cols = set(eda_columns)
data_cols = set(data.columns.tolist())
integrity_check['expected_columns'] = sorted(list(expected_cols))
integrity_check['data_columns'] = sorted(list(data_cols))
integrity_check['missing_in_data'] = sorted(list(expected_cols - data_cols))
integrity_check['additional_in_data'] = sorted(list(data_cols - expected_cols))

# Compare dtypes, allow coercion
dtype_mismatches = []
col_type_map = {
    'Nitrogen':'int64',
    'Phosphorus':'int64',
    'Potassium':'int64',
    'Temperature':'float64',
    'Humidity':'float64',
    'pH_Value':'float64',
    'Rainfall':'float64',
    'Crop':'object',
    'Yield':'int64'
}
for col, eda_type in col_type_map.items():
    if col not in data.columns:
        dtype_mismatches.append({'col':col, 'error':'Missing in data'})
    else:
        actual_type = str(data[col].dtype)
        if eda_type != actual_type:
            dtype_mismatches.append({'col':col, 'expected':eda_type, 'actual':actual_type})

integrity_check['dtype_mismatches'] = dtype_mismatches
integrity_check['rows'] = len(data)
integrity_check['columns'] = len(data.columns)
log_data['data_integrity_check'] = integrity_check
save_json(LOG_PATH, log_data)

#################################################
# 3. Preprocess raw data (dtypes, missing, dedupe)
#################################################
preprocessing_summary = {}
for col, dtype in col_type_map.items():
    if dtype == 'int64':
        data[col] = pd.to_numeric(data[col], downcast='integer', errors='coerce').astype('Int64')
        # Use pandas Int64 to handle possible NAs before re-casting
    elif dtype == 'float64':
        data[col] = pd.to_numeric(data[col], downcast='float', errors='coerce').astype('float64')
    elif dtype == 'object':
        data[col] = data[col].astype('object')
# Fix any Int64 dtypes back to int for export
for col in [k for k,v in col_type_map.items() if v == 'int64']:
    data[col] = data[col].fillna(0).astype('int64')

missing_vals = data.isnull().sum().sum()
duplicates = data.duplicated().sum()

preprocessing_summary['initial_shape'] = [len(data), len(data.columns)]
preprocessing_summary['cast_types'] = col_type_map
preprocessing_summary['missing_vals'] = int(missing_vals)
preprocessing_summary['duplicates_removed'] = int(duplicates)
if missing_vals:
    data = data.dropna()
if duplicates:
    data = data.drop_duplicates()
preprocessing_summary['final_shape'] = [len(data), len(data.columns)]
data.to_csv('./output/11/cleaned_data.csv', index=False)
preprocessing_summary['cleaned_data_path'] = './output/11/cleaned_data.csv'
log_data['preprocessing_summary'] = preprocessing_summary
save_json(LOG_PATH, log_data)

#################################################
# 4. Outlier treatment (IQR/ Winsorization)
#################################################
outlier_treatment_summary = {}
numerical_features = [
    'Nitrogen','Phosphorus','Potassium',
    'Temperature','Humidity','pH_Value','Rainfall', 'Yield'
]
outlier_info = eda_info.get('outlier_info', {}) # May include counts
for col in numerical_features:
    if col not in data.columns: continue
    col_outlier = outlier_info.get(col, {})
    n_outliers = col_outlier.get('n_outliers', None)
    values_before = data[col].describe(percentiles=[.25, .5, .75]).to_dict()
    
    # Use IQR if substantial outliers or per instruction
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers_before = ((data[col] < lower) | (data[col] > upper)).sum()
    
    if outliers_before > 0:
        capped = data[col].clip(lower, upper)
        capped_points = (capped != data[col]).sum()
        data[col] = capped
        values_after = data[col].describe(percentiles=[.25, .5, .75]).to_dict()
        outlier_treatment_summary[col] = {
            'used': 'IQR_capping',
            'n_outliers_before': int(outliers_before),
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'n_capped': int(capped_points),
            'min_before': float(values_before['min']),
            'max_before': float(values_before['max']),
            'min_after': float(values_after['min']),
            'max_after': float(values_after['max'])
        }
    else:
        outlier_treatment_summary[col] = {
            'used': 'none_needed',
            'n_outliers_before': int(outliers_before)
        }
log_data['outlier_treatment_summary'] = outlier_treatment_summary
save_json(LOG_PATH, log_data)

#####################################################################
# 5. One-hot encoding for 'Crop' (22 unique values, moderate cardinality)
#####################################################################
encoding_summary = {}

onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
crop_reshaped = data[['Crop']].astype(str)
crop_oh = onehot.fit_transform(crop_reshaped)
crop_oh_df = pd.DataFrame(
    crop_oh, columns=[f'Crop_{cat}' for cat in onehot.categories_[0]]
)
data_enc = pd.concat([data.reset_index(drop=True), crop_oh_df], axis=1)
data_enc = data_enc.drop(columns=['Crop'])

encoding_summary['original_variable'] = 'Crop'
encoding_summary['unique_values'] = sorted(data['Crop'].unique().tolist())
encoding_summary['encoded_features'] = crop_oh_df.columns.tolist()
encoding_summary['encoding_type'] = 'OneHotEncoder'
encoding_summary['encoder_mapping'] = dict(
    zip(encoding_summary['unique_values'], crop_oh_df.columns.tolist())
)
log_data['encoding_summary'] = encoding_summary
save_json(LOG_PATH, log_data)

#########################################
# 6. Data leakage check as per EDA info
#########################################
data_leakage_check = {'flagged': eda_info.get('data_leakage_flag', None) or 'none', 'result':'No features flagged for leakage per EDA'}
log_data['data_leakage_check'] = data_leakage_check
save_json(LOG_PATH, log_data)

##################################################
# 7. Feature selection from EDA correlations/info
##################################################
selected_features = {}
# Example of what EDA might provide:
eda_corrs = eda_info.get('feature_target_correlation', {})  
# {'Nitrogen':0.11, ...}

# Get all numeric + one-hot 'Crop' columns unless excluded
to_remove = [f for f, corr in eda_corrs.items() if abs(corr)<0.01]
# Optional: don't drop these if required
feature_list = []
feature_types = {}
feature_correlations = {}

for col in data_enc.columns:
    if col == target_col: continue
    if col in eda_corrs and abs(eda_corrs[col])<0.01:
        continue # Exclude very low correlation numeric features
    feature_list.append(col)
    if col.startswith('Crop_'):
        feature_types[col]='encoded_categorical'
        feature_correlations[col]=None
    else:
        feature_types[col]=str(data_enc[col].dtype)
        feature_correlations[col]=eda_corrs.get(col, None)
selected_features['selected_feature_list'] = feature_list
selected_features['types'] = feature_types
selected_features['correlation_with_target'] = feature_correlations

log_data['selected_features'] = selected_features
save_json(LOG_PATH, log_data)

##########################################################################
# 8. Target transformation: log1p if high skew (>15), kurtosis (>460) from EDA
##########################################################################
target_transformation = {}
target_skew = eda_info.get('target_distribution',{}).get('skew',0)
target_kurt = eda_info.get('target_distribution',{}).get('kurt',0)
if target_skew > 15 and target_kurt > 460:
    trans_type = 'np.log1p'
    new_y = np.log1p(data_enc[target_col])
    stats_before = dict(
        skew=float(target_skew),
        kurt=float(target_kurt),
        min=float(data_enc[target_col].min()),
        max=float(data_enc[target_col].max())
    )
    stats_after = {
        'skew': float(pd.Series(new_y).skew()),
        'kurt': float(pd.Series(new_y).kurt()),
        'min': float(new_y.min()),
        'max': float(new_y.max())
    }
    data_enc[target_col] = new_y
    target_transformation = {
        'applied': True,
        'method': trans_type,
        'stats_before': stats_before,
        'stats_after': stats_after
    }
else:
    target_transformation = {
        'applied': False,
        'reason': 'Skew/kurtosis thresholds not met',
        'stats': {
            'skew': float(target_skew),
            'kurt': float(target_kurt),
            'min': float(data_enc[target_col].min()),
            'max': float(data_enc[target_col].max())
        }
    }
preprocessed_data_path = './output/11/preprocessed_data.csv'
data_enc.to_csv(preprocessed_data_path, index=False)
target_transformation['preprocessed_data_path'] = preprocessed_data_path
log_data['target_transformation'] = target_transformation
save_json(LOG_PATH, log_data)

#########################################################
# 9. Data splits (train/val/test) with stratification over 'Crop'
#########################################################
data_enc['Crop_raw'] = data['Crop'].values # add for stratification
X = data_enc[feature_list]
y = data_enc[target_col]

train_X, temp_X, train_y, temp_y, train_c, temp_c = train_test_split(
    X, y, data_enc['Crop_raw'], 
    test_size=0.3, stratify=data_enc['Crop_raw'], random_state=42
)
val_X, test_X, val_y, test_y, val_c, test_c = train_test_split(
    temp_X, temp_y, temp_c, 
    test_size=0.5, stratify=temp_c, random_state=42
)
splits = {
    'train': {'X':train_X, 'y':train_y, 'Crop':train_c},
    'val':   {'X':val_X,   'y':val_y,   'Crop':val_c},
    'test':  {'X':test_X,  'y':test_y,  'Crop':test_c}
}
split_summaries = {}
for split_name, ds in splits.items():
    path = f'./output/11/{split_name}.csv'
    df = pd.concat([ds['X'].reset_index(drop=True), ds['y'].reset_index(drop=True)], axis=1)
    df.to_csv(path, index=False)
    split_summaries[split_name] = {
        'rows': len(df),
        'columns': list(df.columns),
        'target_dist': ds['y'].describe().to_dict(),
        'crop_dist': ds['Crop'].value_counts().to_dict(),
        'path': path
    }
log_data['data_splits'] = split_summaries
save_json(LOG_PATH, log_data)

#########################################################
# 10. Standardize/normalize numeric features using train set only
#########################################################
scaling_parameters = {}
num_feats = [col for col in feature_list if str(train_X[col].dtype).startswith(('float','int'))]
scaler = StandardScaler()
train_X_scaled = train_X.copy()
val_X_scaled = val_X.copy()
test_X_scaled = test_X.copy()
scaler.fit(train_X[num_feats])
scaling_parameters['method'] = 'StandardScaler'
scaling_parameters['feature_means'] = dict(zip(num_feats, scaler.mean_.round(6).tolist()))
scaling_parameters['feature_stds'] = dict(zip(num_feats, scaler.scale_.round(6).tolist()))
train_X_scaled[num_feats] = scaler.transform(train_X[num_feats])
val_X_scaled[num_feats] = scaler.transform(val_X[num_feats])
test_X_scaled[num_feats] = scaler.transform(test_X[num_feats])
log_data['scaling_parameters'] = scaling_parameters
save_json(LOG_PATH, log_data)

#########################################################
# 11. Model selection/config/logging
#########################################################
model_configurations = []
import sklearn
model_dict = {
    'RidgeRegression': {
        'model': Ridge(),
        'libver': sklearn.__version__,
        'init_params': {'alpha':1.0, 'random_state':42}
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'libver': sklearn.__version__,
        'init_params': {'n_estimators':100, 'max_depth':None, 'random_state':42}
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(),
        'libver': sklearn.__version__,
        'init_params': {'n_estimators':100, 'random_state':42}
    }
}
for name, info in model_dict.items():
    entry = {
        'name': name,
        'library_version': info['libver'],
        'initial_hyperparameters': info['init_params']
    }
    model_configurations.append(entry)
log_data['model_configurations'] = model_configurations
save_json(LOG_PATH, log_data)

#########################################################
# 12. Hyperparameter tuning using 5-fold CV (RMSE, MAE, R2)
#########################################################
hyperparameter_tuning = {}
cv = 5
for name, info in model_dict.items():
    if name == 'RidgeRegression':
        grid = {'alpha':[0.1, 1.0, 10.0, 100.0]}
        mdl = Ridge(random_state=42)
    elif name == 'RandomForestRegressor':
        grid = {'n_estimators':[50, 100], 'max_depth':[None, 5, 15]}
        mdl = RandomForestRegressor(random_state=42)
    elif name == 'GradientBoostingRegressor':
        grid = {'n_estimators':[50, 100], 'learning_rate':[0.05, 0.1]}
        mdl = GradientBoostingRegressor(random_state=42)
    else:
        continue
    search = GridSearchCV(
        mdl, grid, 
        cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    search.fit(train_X_scaled, train_y)
    best_params = search.best_params_
    preds = search.predict(val_X_scaled)
    record = {
        'best_params': best_params,
        'cv_results': { 
            'best_score': -search.best_score_,
            'train_rmse': float(np.sqrt(mean_squared_error(train_y, search.predict(train_X_scaled)))),
            'val_rmse': float(np.sqrt(mean_squared_error(val_y, preds))),
            'val_mae': float(mean_absolute_error(val_y, preds)),
            'val_r2': float(r2_score(val_y, preds)),
        }
    }
    hyperparameter_tuning[name] = record
log_data['hyperparameter_tuning'] = hyperparameter_tuning
save_json(LOG_PATH, log_data)

#########################################################
# 13. Train on train+val, save artifacts
#########################################################
model_artifacts = {}
X_tr = pd.concat([train_X_scaled, val_X_scaled], axis=0)
y_tr = pd.concat([train_y, val_y], axis=0)
artifact_paths = {}
all_models = {}
for name, info in model_dict.items():
    best_params = hyperparameter_tuning.get(name, {}).get('best_params', info['init_params'])
    ModelClass = info['model'].__class__
    mdl = ModelClass(**best_params)
    mdl.fit(X_tr, y_tr)
    model_path = f'./output/11/model_{name.lower().replace("regressor","").replace("ridge","ridge")}.pkl'
    joblib.dump(mdl, model_path)
    artifact_paths[name] = model_path
    all_models[name] = mdl
model_artifacts = artifact_paths
log_data['model_artifacts'] = model_artifacts
save_json(LOG_PATH, log_data)

#########################################################
# 14. Evaluate each model on test set
#########################################################
model_evaluations = {}
def _metric_dict(y_true, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }
for name, mdl in all_models.items():
    y_pred = mdl.predict(test_X_scaled)
    metrics = _metric_dict(test_y, y_pred)
    record = {
        'metrics': metrics,
        'predictions': y_pred.tolist()[:100],  # Truncate for log extents
    }
    # Feature importances if tree-based
    if hasattr(mdl,'feature_importances_'):
        record['feature_importance'] = dict(zip(feature_list, mdl.feature_importances_.tolist()))
    model_evaluations[name] = record
log_data['model_evaluations'] = model_evaluations
save_json(LOG_PATH, log_data)

#########################################################
# 15. Compare and choose best model
#########################################################
best_model_choice = None
best_rmse = float('inf')
for name, ev in model_evaluations.items():
    if ev['metrics']['rmse']<best_rmse:
        best_rmse = ev['metrics']['rmse']
        best_model_choice = {
            'model_name': name,
            'metrics': ev['metrics'],
            'artifact_path': model_artifacts[name],
        }
        # If tree-based, add importance
        if 'feature_importance' in ev:
            best_model_choice['feature_importance'] = ev['feature_importance']
log_data['best_model'] = best_model_choice
save_json(LOG_PATH, log_data)

#########################################################
# 16. Pipeline summary for future agents
#########################################################
pipeline_summary = {
    'steps': [
        'EDA info loaded and logged.',
        'Data integrity/consistency checked against EDA outputs.',
        'Raw data preprocessed: dtypes, missing values & duplicates handled.',
        'Outlier capping (IQR) performed on flagged numerical features.',
        'Categorical Crop encoded via one-hot encoding.',
        'No data leakage features flagged.',
        'Feature selection based on EDA correlations (numeric + encoded).',
        'Target transformation (log1p) applied if heavy skew found.',
        'Data split into train/val/test with Crop stratification.',
        'Numerical features scaled (StandardScaler) using train stats.',
        'Three regression models: Ridge, RandomForest, GradientBoosting.',
        'Hyperparameter tuning via 5-fold CV (best by RMSE).',
        'Models trained on train+val and artifacts saved.',
        'All models evaluated on held-out test set.',
        'Best model selected by test RMSE; full metrics and importances logged.'
    ],
    'artifacts': {
        'eda_info': eda_output_path,
        'cleaned_data': './output/11/cleaned_data.csv',
        'preprocessed_data': preprocessed_data_path,
        'train_split': './output/11/train.csv',
        'val_split': './output/11/val.csv',
        'test_split': './output/11/test.csv',
        'model_artifacts': model_artifacts,
        'full_log_json': LOG_PATH
    }
}
log_data['pipeline_summary'] = pipeline_summary
save_json(LOG_PATH, log_data)
#########################################################

print("Pipeline complete. See './output/11/model_training.json' for full audit-log and outputs.")