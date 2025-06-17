import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# --- 1. Load dataset ---
csv_path = 'data/Crop_Yield_Prediction.csv'
output_dir = './output/9'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'eda_agent.json')

result = {}

df = pd.read_csv(csv_path)

# --- 2. Basic dataset info ---
result['basic_info'] = {
    'n_rows': int(df.shape[0]),
    'n_cols': int(df.shape[1]),
    'columns': list(df.columns)
}

# --- 3. Data types ---
result['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}

# --- 4. Missing values ---
missing_values = {}
for col in df.columns:
    total = int(df[col].isnull().sum())
    percent = float(100 * df[col].isnull().mean())
    missing_values[col] = {'total_missing': total, 'percent_missing': percent}

result['missing_values'] = missing_values

# --- Identify numerical and categorical columns ---
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in df.columns if col not in numerical_cols]

# --- 5. Descriptive stats for numerical columns ---
desc_stats = {}
for col in numerical_cols:
    series = df[col]
    desc = {
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        '25th_percentile': float(series.quantile(0.25)),
        '50th_percentile': float(series.quantile(0.5)),
        '75th_percentile': float(series.quantile(0.75))
    }
    desc_stats[col] = desc
result['descriptive_statistics'] = desc_stats

# --- 6. Categorical columns: value counts and cardinality ---
categorical_info = {}
for col in categorical_cols:
    # Convert to string for categorical treatment
    value_counts = df[col].astype(str).value_counts(dropna=False)
    value_counts_json = value_counts.head(50).to_dict()  # Shows top 50 categories if many
    categorical_info[col] = {
        'cardinality': int(df[col].nunique(dropna=False)),
        'value_counts_top50': {str(k): int(v) for k, v in value_counts_json.items()}
    }
result['categorical_info'] = categorical_info

# --- 7. Outlier detection (IQR) for numerical columns ---
outlier_info = {}
for col in numerical_cols:
    series = df[col]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    outlier_info[col] = {
        'outlier_count': int(outliers.shape[0]),
        'outlier_indices': outliers.index.tolist()[:100]  # Only log first 100 indices for brevity
    }
result['outlier_detection_IQR'] = outlier_info

# --- 8. Skewness and Kurtosis ---
skew_kurt = {}
for col in numerical_cols:
    series = df[col]
    # Drop NA for stats
    skew_val = float(skew(series.dropna()))
    kurt_val = float(kurtosis(series.dropna()))
    skew_kurt[col] = {'skewness': skew_val, 'kurtosis': kurt_val}
result['skewness_kurtosis'] = skew_kurt

# --- 9. Pairwise correlation matrix ---
if 'Yield' in numerical_cols:
    sub_df = df[numerical_cols]
else:
    sub_df = df[numerical_cols + ['Yield']] if 'Yield' in df.columns else df[numerical_cols]
corr_matrix = sub_df.corr().to_dict()
# Convert numpy.float64 -> float for JSON
for col1 in corr_matrix:
    for col2 in corr_matrix[col1]:
        corr_matrix[col1][col2] = float(corr_matrix[col1][col2])
result['correlation_matrix'] = corr_matrix

# --- 10. Constant and quasi-constant features ---
constant_features = []
quasi_constant_features = []
for col in df.columns:
    counts = df[col].nunique(dropna=False)
    if counts == 1:
        constant_features.append(col)
    elif df[col].value_counts(normalize=True, dropna=False).values[0] > 0.98:
        quasi_constant_features.append(col)
result['constant_features'] = constant_features
result['quasi_constant_features'] = quasi_constant_features

# --- 11. Target column 'Yield' summary ---
target_col = 'Yield'
yield_summary = {}
if target_col in df.columns:
    series = df[target_col]
    yield_summary['type'] = str(df[target_col].dtype)
    yield_summary['distribution'] = {
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        '25th_percentile': float(series.quantile(0.25)),
        '50th_percentile': float(series.quantile(0.5)),
        '75th_percentile': float(series.quantile(0.75)),
        'skewness': float(skew(series.dropna())),
        'kurtosis': float(kurtosis(series.dropna())),
    }
    yield_summary['n_negative'] = int((series < 0).sum())
    yield_summary['n_zero'] = int((series == 0).sum())
    yield_summary['domain_notes'] = (
        "Crop yield ('Yield') should realistically be positive. Presence of negative or zero yields "
        "may indicate missing, erroneous, or placeholder values. Please check domain definitions."
    )
else:
    yield_summary['error'] = "Target column 'Yield' not found in dataset."
result['target_summary'] = yield_summary

# --- Save to JSON ---
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"EDA results successfully saved to {output_path}")