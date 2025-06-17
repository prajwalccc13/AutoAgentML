import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import joblib
import sklearn

# === File/path settings ===
output_dir = './output/1/'
os.makedirs(output_dir, exist_ok=True)
eda_path = os.path.join(output_dir, 'eda_agent.json')
data_path = './data/iris.csv'
log_path = os.path.join(output_dir, 'model_training.json')

# === Step 1: Load EDA and data, log summary ===

with open(eda_path, 'r') as f:
    eda_json = json.load(f)
df = pd.read_csv(data_path)

log = {}
log['eda_summary'] = eda_json
log['data_parameters'] = {'shape': df.shape, 'columns': df.columns.tolist()}

# === Step 2: Check missing values & cardinality. Clean if needed. Log actions ===

cleaning_actions = []
missing_info = df.isnull().sum().to_dict()
high_card_cols = [col for col in df.columns if df[col].nunique() > 30]  # e.g. 30+ unique for small datasets

if sum(missing_info.values()) == 0:
    cleaning_actions.append('No missing values found.')
else:
    cleaning_actions.append(f"Missing values in columns: { {k:v for k,v in missing_info.items() if v>0} }")
    df = df.dropna()
    cleaning_actions.append('Rows with missing values dropped.')

if len(high_card_cols) == 0:
    cleaning_actions.append('No high cardinality columns (>30 uniques) found.')
else:
    cleaning_actions.append(f"High cardinality columns identified: {high_card_cols}")
    # For iris dataset, this is unlikely; just an example
    for col in high_card_cols:
        df = df.drop(columns=[col])
        cleaning_actions.append(f"Column '{col}' dropped due to high cardinality.")

log['data_cleaning'] = cleaning_actions

# === Step 3: Feature selection ===

all_numerical_features = [col for col in df.columns if col not in ['variety'] and pd.api.types.is_numeric_dtype(df[col])]
feature_sets = {
    'sepal_width_only': ['sepal.width'],
    'all_numerical': all_numerical_features,
    'all_except_sepal_width': [f for f in all_numerical_features if f != 'sepal.width'],
}
log['feature_sets'] = feature_sets

# === Step 4: Encode categorical target, save mapping ===

label_enc = LabelEncoder()
df['variety_encoded'] = label_enc.fit_transform(df['variety'])
target_mapping = dict(zip(map(str, label_enc.transform(label_enc.classes_)), label_enc.classes_))
log['target_encoding'] = {
    'classes': label_enc.classes_.tolist(),
    'mapping': target_mapping
}

# === Step 5: Preprocessing (scale numerical features), log scaler params ===

scalers = {}
scaler_params = {}
for fs_name, fs_feats in feature_sets.items():
    scaler = StandardScaler()
    scaler.fit(df[fs_feats])
    scalers[fs_name] = scaler
    scaler_params[fs_name] = {
        col: {'mean': scaler.mean_[i], 'std': scaler.scale_[i]}
        for i, col in enumerate(fs_feats)
    }
log['scaler_params'] = scaler_params

# === Step 6: Stratified split, log indices and sizes ===

X = df[all_numerical_features]
y = df['variety_encoded']
train_idx, test_idx = train_test_split(df.index, stratify=y, test_size=0.2, random_state=42)
split_metadata = {
    'train_indices': train_idx.tolist(),
    'test_indices': test_idx.tolist(),
    'train_size': len(train_idx),
    'test_size': len(test_idx),
    'stratify': True
}
log['split_metadata'] = split_metadata

# === Step 7: Model training (LogReg, Tree, RF) w/ CV, log hyperparams + CV ===

models = {
    'logistic_regression': LogisticRegression(max_iter=200, random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
}
model_training = {}

for fs_name, fs_feats in feature_sets.items():
    X_all = df[fs_feats]
    X_train = X_all.loc[train_idx]
    y_train = y.loc[train_idx]
    scaler = scalers[fs_name]
    X_train_scaled = scaler.transform(X_train)
    model_training[fs_name] = {}
    for model_name, model in models.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='accuracy')
        model_training[fs_name][model_name] = {
            'hyperparameters': model.get_params(),
            'cv_scores': cv_scores.tolist(),
            'mean_cv_accuracy': float(np.mean(cv_scores)),
            'std_cv_accuracy': float(np.std(cv_scores)),
        }
log['model_training'] = model_training

# === Step 8: Evaluate on test set (metrics, predictions), log results ===

evaluation = {}
for fs_name, fs_feats in feature_sets.items():
    X_all = df[fs_feats]
    X_train, X_test = X_all.loc[train_idx], X_all.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    scaler = scalers[fs_name]
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    evaluation[fs_name] = {}
    for model_name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        evaluation[fs_name][model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_predictions': y_pred.tolist(),
            'test_true': y_test.tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=label_enc.classes_, output_dict=True)
        }
log['evaluation'] = evaluation

# === Step 9: Compare and select best model/feature set, log rationale ===

# Find highest test accuracy (and ties broken by f1-score), log selection
max_acc = -1
max_f1 = -1
final_choice = (None, None)
for fs_name in evaluation:
    for model_name in evaluation[fs_name]:
        entry = evaluation[fs_name][model_name]
        acc = entry['accuracy']
        f1 = entry['f1_score']
        if (acc > max_acc) or (acc == max_acc and f1 > max_f1):
            max_acc = acc
            max_f1 = f1
            final_choice = (fs_name, model_name)

best_fs, best_model_name = final_choice
best_summary = {
    "selected_feature_set": best_fs,
    "selected_model": best_model_name,
    "accuracy": evaluation[best_fs][best_model_name]["accuracy"],
    "f1_score": evaluation[best_fs][best_model_name]["f1_score"],
    "rationale": (
        f"Selected '{best_model_name}' with feature set '{best_fs}' "
        "because it achieved the highest test accuracy, breaking ties with f1-score."
    ),
    "metrics": evaluation[best_fs][best_model_name]
}
log['best_model'] = best_summary

# === Step 10: Save model, scaler, input features, encoding; log file paths ===

# Train on the whole train set (not CV) for final model to save
final_scaler = scalers[best_fs]
final_model_obj = models[best_model_name]
features_to_use = feature_sets[best_fs]
# Refit to all train data
final_scaler.fit(df.loc[train_idx, features_to_use])
final_model_obj.fit(final_scaler.transform(df.loc[train_idx, features_to_use]), y.loc[train_idx])
# Save
artifact_paths = {}
# Model
model_path = os.path.join(output_dir, f'best_model_{best_model_name}_{best_fs}.joblib')
joblib.dump(final_model_obj, model_path)
artifact_paths['model'] = model_path
# Scaler
scaler_path = os.path.join(output_dir, f'best_scaler_{best_fs}.joblib')
joblib.dump(final_scaler, scaler_path)
artifact_paths['scaler'] = scaler_path
# Features
features_path = os.path.join(output_dir, 'input_features.json')
with open(features_path, 'w') as f:
    json.dump(features_to_use, f)
artifact_paths['input_features'] = features_path
# Target encoding
target_enc_path = os.path.join(output_dir, 'target_encoding.json')
with open(target_enc_path, 'w') as f:
    json.dump(log['target_encoding'], f)
artifact_paths['target_encoding'] = target_enc_path

# Versioning
artifact_paths['sklearn_version'] = sklearn.__version__

log['artifacts'] = artifact_paths

# === Save log JSON file ===
with open(log_path, 'w') as f:
    json.dump(log, f, indent=2)

print(f'All results logged to {log_path}')
print(f'Artifacts saved: {", ".join(list(artifact_paths.values()))}')