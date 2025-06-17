import pandas as pd
import numpy as np
import os
import json

# File paths
data_path = 'data/iris.csv'
output_dir = './output/6'
output_file = os.path.join(output_dir, 'eda_agent.json')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load Dataset
df = pd.read_csv(data_path)

# Prepare dictionary to hold all results
eda_results = {}

# 1. Log the shape and column names
eda_results['dataset_overview'] = {
    'shape': {'rows': df.shape[0], 'columns': df.shape[1]},
    'column_names': df.columns.tolist()
}

# 2. Identify data types per column
def col_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numerical'
    elif pd.api.types.is_categorical_dtype(dtype):
        return 'categorical'
    elif pd.api.types.is_object_dtype(dtype):
        return 'categorical'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'
    else:
        return str(dtype)

eda_results['column_types'] = {
    col: col_type(dt) for col, dt in df.dtypes.items()
}

# 3. Check for missing values in each column
missing_info = []
n_rows = len(df)
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_pct = float(missing_count) / n_rows * 100
    missing_info.append({
        'column': col,
        'missing_count': int(missing_count),
        'missing_percentage': round(missing_pct, 4)
    })
eda_results['missing_values'] = missing_info

# 4. Statistical summary for numerical columns
num_cols = [col for col, t in eda_results['column_types'].items() if t == 'numerical']
num_summary = {}
for col in num_cols:
    series = df[col].dropna()
    num_summary[col] = {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        '25%': series.quantile(0.25),
        '50%': series.quantile(0.5),
        '75%': series.quantile(0.75)
    }
eda_results['numerical_summary'] = num_summary

# 5. Cardinality for categorical columns (including target)
cat_cols = [col for col, t in eda_results['column_types'].items() if t == 'categorical']
cat_cardinality = {}
for col in cat_cols:
    card = df[col].nunique(dropna=False)
    cat_cardinality[col] = int(card)
eda_results['categorical_cardinality'] = cat_cardinality

# 6. Frequency distribution for target: 'variety'
if 'variety' in df.columns:
    target_counts = df['variety'].value_counts(dropna=False)
    target_pct = df['variety'].value_counts(normalize=True, dropna=False) * 100
    freq_dist = []
    for val in target_counts.index:
        freq_dist.append({
            'class': val,
            'count': int(target_counts[val]),
            'percentage': round(target_pct[val], 4)
        })
    eda_results['target_frequency_distribution'] = freq_dist

# 7. Outlier detection using IQR (for each numerical column)
outliers_info = {}
for col in num_cols:
    series = df[col].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outliers_info[col] = {
        'outlier_indices': outliers.index.tolist(),
        'outlier_values': outliers.values.tolist(),
        'n_outliers': int(len(outliers))
    }
eda_results['iqr_outliers'] = outliers_info

# 8. Pairwise correlation coefficients for numerical columns
if len(num_cols) > 1:
    corr_matrix = df[num_cols].corr()
    eda_results['correlation_matrix'] = corr_matrix.round(6).to_dict()
else:
    eda_results['correlation_matrix'] = {}

# 9. Aggregated stats by target for numerical features
by_target_agg = {}
if 'variety' in df.columns:
    grouped = df.groupby('variety')
    for col in num_cols:
        by_target_agg[col] = {}
        by_target_agg[col]['mean'] = grouped[col].mean().round(6).to_dict()
        by_target_agg[col]['std']  = grouped[col].std().round(6).to_dict()
        by_target_agg[col]['min']  = grouped[col].min().to_dict()
        by_target_agg[col]['max']  = grouped[col].max().to_dict()
eda_results['numerical_by_target'] = by_target_agg

# 10. Check for duplicate rows
duplicates_mask = df.duplicated(keep=False)
duplicate_rows = df[duplicates_mask]
eda_results['duplicates'] = {
    'n_duplicates': int(duplicate_rows.shape[0]),
    'duplicate_indices': duplicate_rows.index.tolist()
}

# 11. Near-zero variance columns (here: variance < 1e-4)
nzv_cols = []
nzv_values = {}
for col in num_cols:
    var = df[col].var()
    if var < 1e-4:
        nzv_cols.append(col)
        nzv_values[col] = var
eda_results['near_zero_variance_columns'] = {
    'columns': nzv_cols,
    'variance_values': {col: float(var) for col, var in nzv_values.items()}
}

# 12. Columns with constant (single unique) values
constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
eda_results['constant_columns'] = constant_cols

# Save all results to JSON
with open(output_file, 'w') as f:
    json.dump(eda_results, f, indent=2, default=str)