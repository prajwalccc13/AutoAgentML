import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# Ensure output directory exists
output_dir = './output/3'
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, 'eda_agent.json')
results = {}

#### 1. Load dataset & log basic info ####
df = pd.read_csv('data/Crop_Yield_Prediction.csv')

results['basic_info'] = {
    'num_rows': int(df.shape[0]),
    'num_columns': int(df.shape[1]),
    'column_names': df.columns.tolist(),
    'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
}

#### 2. Detect & log data types ####
def detect_col_type(series):
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'boolean'
    elif pd.api.types.is_numeric_dtype(dtype):
        return 'numerical'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'
    elif pd.api.types.is_categorical_dtype(series) or series.nunique() < 50 or series.dtype == object:
        return 'categorical'
    else:
        return 'other'
data_types = {}
for col in df.columns:
    data_types[col] = detect_col_type(df[col])
results['detected_column_types'] = data_types

#### 3. Missing values ####
missing_values = {}
for col in df.columns:
    n_missing = int(df[col].isnull().sum())
    pct_missing = float(n_missing) / df.shape[0]
    missing_values[col] = {
        'count': n_missing,
        'percent': round(pct_missing * 100, 4)
    }
results['missing_values'] = missing_values

#### 4. Descriptive statistics (numerical, including target "Yield") ####
desc_stats = {}
numeric_cols = [col for col, v in data_types.items() if v == 'numerical']
if 'Yield' in df.columns and 'Yield' not in numeric_cols:
    numeric_cols.append('Yield')
for col in numeric_cols:
    dat = df[col].dropna()
    stats_dict = {
        'count': int(dat.count()),
        'mean': float(dat.mean()),
        'median': float(dat.median()),
        'std': float(dat.std()),
        'min': float(dat.min()),
        'max': float(dat.max()),
        '25%': float(dat.quantile(0.25)),
        '50%': float(dat.quantile(0.50)),
        '75%': float(dat.quantile(0.75))
    }
    desc_stats[col] = stats_dict
results['descriptive_statistics'] = desc_stats

#### 5. Cardinality (number unique values) ####
cardinality = {}
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    sample_uniques = unique_vals[:10].tolist()
    cardinality[col] = {
        'num_unique': int(df[col].nunique(dropna=True)),
        'sample_unique_values': [str(val) for val in sample_uniques]
    }
results['cardinality'] = cardinality

#### 6. Frequency distribution (categorical) ####
cat_freq = {}
for col, col_type in data_types.items():
    if col_type == 'categorical':
        counts = df[col].value_counts(dropna=False)
        sample = counts.head(20).to_dict()
        cat_freq[col] = {str(k): int(v) for k, v in sample.items()}
results['categorical_frequency_distribution'] = cat_freq

#### 7. Outlier detection (IQR and Z-score) ####
outliers = {}
for col in numeric_cols:
    x = df[col].dropna()
    if x.empty:
        continue
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_outlier_idx = x[(x < lower) | (x > upper)].index.tolist()
    z_scores = np.abs(stats.zscore(x))
    z_outlier_idx = x.index[z_scores > 3].tolist()
    outliers[col] = {
        'IQR_outliers': {
            'count': len(iqr_outlier_idx),
            'indices': iqr_outlier_idx[:100],  # Limiting if many
        },
        'z_score_outliers': {
            'count': len(z_outlier_idx),
            'indices': z_outlier_idx[:100],
        }
    }
results['outliers'] = outliers

#### 8. Distribution shape (skewness, kurtosis) ####
distribution_shape = {}
for col in numeric_cols:
    x = df[col].dropna()
    if x.empty:
        continue
    distribution_shape[col] = {
        'skewness': float(stats.skew(x)),
        'kurtosis': float(stats.kurtosis(x))
    }
results['distribution_shape'] = distribution_shape

#### 9. Pairwise correlation matrix ####
corr_matrix = df[numeric_cols].corr().to_dict()
# Make JSON serializable
corr_matrix = {k: {j: float(v) for j, v in d.items()} for k, d in corr_matrix.items()}
results['correlation_matrix'] = corr_matrix

#### 10. Categorical-numerical associations (mean Yield per category) ####
cat_num_associations = {}
if 'Yield' in df.columns:
    for col, col_type in data_types.items():
        if col_type == 'categorical':
            cat_avg = df.groupby(col)['Yield'].mean().sort_values(ascending=False)
            cat_num_associations[col] = {str(idx): float(val) for idx, val in cat_avg.head(20).items()}
results['cat_num_associations'] = cat_num_associations

#### 11. Feature suitability & recommendations ####
feature_recommendations = {}
for col in df.columns:
    col_info = {}
    # Constant column
    if df[col].nunique(dropna=False) == 1:
        col_info['warning'] = 'Constant column'
        col_info['recommendation'] = 'Drop - provides no useful information.'
    # High missing rate
    elif missing_values[col]['percent'] > 40:
        col_info['warning'] = f'High missing rate ({missing_values[col]["percent"]:.2f}%)'
        col_info['recommendation'] = 'Evaluate imputation or drop if not useful.'
    # Low variance
    elif data_types[col]=='numerical' and df[col].std() < 1e-5:
        col_info['warning'] = 'Low variance'
        col_info['recommendation'] = 'Potentially uninformative; consider dropping.'
    # Potential leakage (simple heuristic: if column contains 'yield' or 'target' in its name)
    elif any(token in col.lower() for token in ['yield', 'target']):
        col_info['warning'] = 'Target leakage possible'
        col_info['recommendation'] = 'Review if this feature leaks information about target.'
    else:
        col_info['recommendation'] = 'Suitable for modeling (subject to further preprocessing and domain check)'
    feature_recommendations[col] = col_info
results['feature_recommendations'] = feature_recommendations

# Save to JSON
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"EDA summary saved to: {json_path}")