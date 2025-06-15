import json
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import joblib

# Load the data_analysis_log.json file and parse insights
with open('data_analysis_log.json', 'r') as f:
    eda_log = json.load(f)

# Assume eda_log is a dictionary with keys like:
# 'data_types', 'missing_values', 'categorical_features', 'distribution_summaries', 'outlier_info', 'core_features', 'target_variable', 'task_type'

# Step 2: Read irish.csv into DataFrame
df = pd.read_csv('iris.csv')  # assuming 'iris.csv' exists
temp_dir = tempfile.mkdtemp()
raw_csv_path = os.path.join(temp_dir, 'data_raw.csv')
df.to_csv(raw_csv_path, index=False)

# Step 3: Handle missing values
# Identify features with missingness from the EDA log
missing_info = eda_log.get('missing_values', {})
features_with_missing = [feat for feat, val in missing_info.items() if val > 0]

# Determine feature types
data_types = eda_log.get('data_types', {})  # {'feat1': 'numeric', 'feat2': 'categorical', ...}

# Prepare imputer strategies
for feature in features_with_missing:
    if data_types.get(feature) == 'numeric':
        median_val = df[feature].median()
        df[feature].fillna(median_val, inplace=True)
    else:
        mode_val = df[feature].mode()[0]
        df[feature].fillna(mode_val, inplace=True)

# Save cleaned data
imputed_csv_path = os.path.join(temp_dir, 'data_imputed.csv')
df.to_csv(imputed_csv_path, index=False)

# Step 4: Normalize numerical features and encode categorical features
numerical_features = [feat for feat, dtype in data_types.items() if dtype == 'numeric']
categorical_features = [feat for feat, dtype in data_types.items() if dtype == 'categorical']

# Define transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Fit and transform data
X = df.drop(columns=['target'])  # assuming 'target' is the target; adjust as needed
y = df['target']

X_processed = preprocessor.fit_transform(X)

# Save processed data
processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
processed_df['target'] = y.values
processed_csv_path = os.path.join(temp_dir, 'data_processed.csv')
processed_df.to_csv(processed_csv_path, index=False)

# Step 5: Select core features based on distribution and cardinality info
distribution_summaries = eda_log.get('distribution_summaries', {})
cardinality_info = eda_log.get('cardinality', {})  # e.g., {'feat1': 10, ...}
core_features = eda_log.get('core_features', [])  # assuming list provided

# Save core features to JSON
selected_features_path = 'selected_features.json'
with open(selected_features_path, 'w') as f:
    json.dump(core_features, f)

# Step 6: Split data into train/test
X_core = df[core_features]
y_core = df['target']

# Determine if it's classification or regression from EDA log
task_type = eda_log.get('task_type', 'classification')  # default to classification

if task_type == 'classification':
    stratify = y_core
else:
    stratify = None

X_train, X_test, y_train, y_test = train_test_split(
    X_core, y_core, test_size=0.2, stratify=stratify
)

# Save train/test data
X_train.to_csv('train_data.csv', index=False)
X_test.to_csv('test_data.csv', index=False)

# Step 7: Encode target variable if classification
if task_type == 'classification':
    label_enc = LabelEncoder()
    y_train_enc = label_enc.fit_transform(y_train)
    y_test_enc = label_enc.transform(y_test)
    # Save encoded y
    with open('y_train.json', 'w') as f:
        json.dump({'y_train': y_train_enc.tolist()}, f)
    with open('y_test.json', 'w') as f:
        json.dump({'y_test': y_test_enc.tolist()}, f)
    y_train, y_test = y_train_enc, y_test_enc
else:
    # For regression, save as is
    pd.DataFrame({'y_train': y_train}).to_json('y_train.json')
    pd.DataFrame({'y_test': y_test}).to_json('y_test.json')

# Step 8: Configure model pipeline with hyperparameters
hyperparams = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
}
# Save hyperparameters
with open('model_params.json', 'w') as f:
    json.dump(hyperparams, f)

# Define model
if task_type == 'classification':
    model = RandomForestClassifier(**hyperparams)
else:
    model = RandomForestRegressor(**hyperparams)

# Save model parameters
# (Already saved as 'model_params.json')

# Step 9: Train the model
model.fit(X_train, y_train)
# Save trained model
joblib.dump(model, 'final_model.pkl')

# Step 10: Evaluate the model
y_pred = model.predict(X_test)

if task_type == 'classification':
    eval_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
else:
    eval_metrics = {
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'R2': r2_score(y_test, y_pred)
    }

# Save evaluation metrics
with open('evaluation_metrics.json', 'w') as f:
    json.dump(eval_metrics, f)

# Step 11: Log paths to master JSON
training_log = {
    'cleaned_data_path': imputed_csv_path,
    'processed_data_path': processed_csv_path,
    'feature_list': selected_features_path,
    'model_params': 'model_params.json',
    'trained_model': 'final_model.pkl',
    'evaluation_metrics': 'evaluation_metrics.json',
    'train_data_path': 'train_data.csv',
    'test_data_path': 'test_data.csv'
}

with open('training_log.json', 'w') as f:
    json.dump(training_log, f)

print("All steps completed successfully.")