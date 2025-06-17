import os
import json
import pandas as pd
import numpy as np

output_dir = './output/id_3'
output_file = os.path.join(output_dir, 'eda_agent.json')
input_csv = 'data/Crop_Yield_Prediction.csv'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

results = {}

# 1. Load dataset, basic info
df = pd.read_csv(input_csv)
results['basic_info'] = {
    'n_rows': int(df.shape[0]),
    'n_columns': int(df.shape[1]),
    'column_names': df.columns.tolist(),
}

# 2. Data type for each column
results['column_data_types'] = {}
for col in df.columns:
    dtype = str(df[col].dtype)
    if dtype == 'object':
        inferred_type = 'categorical'
    elif np.issubdtype(df[col].dtype, np.number):
        inferred_type = 'numeric'
    elif np.issubdtype(df[col].dtype, np.datetime64):
        inferred_type = 'datetime'
    else:
        inferred_type = dtype
    results['column_data_types'][col] = {
        'pandas_dtype': dtype,
        'inferred_type': inferred_type,
    }

# 3. Missing values (number and %)
results['missing_values'] = {}
for col in df.columns:
    n_missing = int(df[col].isnull().sum())
    perc_missing = float((n_missing / results['basic_info']['n_rows']) * 100)
    results['missing_values'][col] = {
        'n_missing': n_missing,
        'perc_missing': perc_missing
    }

# 4. Summary statistics for numeric columns, including 'Yield'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Yield' not in numeric_cols and 'Yield' in df.columns:
    numeric_cols.append('Yield')
results['numeric_summary_statistics'] = {}
for col in numeric_cols:
    if col not in df:
        continue
    descr = df[col].describe(percentiles=[.25, .5, .75])
    results['numeric_summary_statistics'][col] = {
        'mean': float(descr['mean']) if 'mean' in descr else None,
        'median': float(df[col].median(skipna=True)),
        'std': float(descr['std']) if 'std' in descr else None,
        'min': float(descr['min']) if 'min' in descr else None,
        '25%': float(descr['25%']) if '25%' in descr else None,
        '50%': float(descr['50%']) if '50%' in descr else None,
        '75%': float(descr['75%']) if '75%' in descr else None,
        'max': float(descr['max']) if 'max' in descr else None,
        'count': int(descr['count']) if 'count' in descr else None
    }

# 5. Unique value count (cardinality), distinguish low/high cardinality
# Define "low" as <=10 unique values (can adjust)
results['column_cardinality'] = {}
CARDINALITY_THRESHOLD = 10
for col in df.columns:
    n_unique = df[col].nunique(dropna=False)
    if n_unique <= CARDINALITY_THRESHOLD:
        cardinality_type = 'low'
    else:
        cardinality_type = 'high'
    results['column_cardinality'][col] = {
        'n_unique': int(n_unique),
        'cardinality_type': cardinality_type
    }

# 6. Frequency of each category for categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
results['categorical_frequencies'] = {}
for col in cat_cols:
    freq = df[col].value_counts(dropna=False).to_dict()
    freq_serializable = {str(k): int(v) for k, v in freq.items()}
    results['categorical_frequencies'][col] = freq_serializable

# 7. Duplicated rows (number and indices)
duplicated_mask = df.duplicated(keep=False)
n_duplicates = int(duplicated_mask.sum())
duplicate_indices = df.index[duplicated_mask].tolist()
results['duplicates'] = {
    'n_duplicates': n_duplicates,
    'duplicate_indices': duplicate_indices
}

# 8. Outlier stats for each numeric column (using IQR)
results['outliers'] = {}
for col in numeric_cols:
    if col not in df:
        continue
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_indices = df.index[outlier_mask].tolist()
    results['outliers'][col] = {
        'iqr': float(iqr),
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'n_outliers': int(outlier_mask.sum()),
        'outlier_indices': outlier_indices
    }

# 9. Correlation matrix for numeric features plus Yield
numeric_corr_df = df[numeric_cols].corr()
results['correlation_matrix'] = numeric_corr_df.round(4).fillna('').to_dict()

# 10. Missing values segmented by target quartiles ('Yield')
results['missing_by_yield_quartile'] = {}
if 'Yield' in df.columns:
    yield_col = df['Yield']
    # Quartile bins: 0-25%, 25-50%, 50-75%, 75-100%
    qbins = pd.qcut(yield_col, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    seg_result = {}
    for col in df.columns:
        seg_result_col = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            seg_mask = (qbins == q)
            seg_df = df[seg_mask]
            n_missing = int(seg_df[col].isnull().sum())
            seg_result_col[q] = n_missing
        seg_result[col] = seg_result_col
    results['missing_by_yield_quartile'] = seg_result

# 11. Data quality issues
data_quality_issues = []

# Columns with >50% missing
for col, mv in results['missing_values'].items():
    if mv['perc_missing'] > 50:
        data_quality_issues.append(f"Column '{col}' has more than 50% missing values.")

# Columns with only a single value (or all NaN)
for col in df.columns:
    if df[col].nunique(dropna=True) <= 1:
        data_quality_issues.append(f"Column '{col}' has only a single unique value (not useful for modeling).")
    if df[col].isnull().all():
        data_quality_issues.append(f"Column '{col}' contains only null values.")

# Numeric columns with zero std (constant)
for col in numeric_cols:
    if col not in df:
        continue
    if df[col].std(skipna=True) == 0:
        data_quality_issues.append(f"Numeric column '{col}' is constant (zero standard deviation).")

# Categorical columns with very high cardinality
for col in cat_cols:
    if results['column_cardinality'][col]['cardinality_type'] == 'high':
        data_quality_issues.append(f"Categorical column '{col}' has high cardinality ({results['column_cardinality'][col]['n_unique']} unique values).")

# Columns with suspicious distributions (e.g. majority missing or a single dominant value)
for col in df.columns:
    most_freq = df[col].value_counts(dropna=False).iloc[0]
    perc_mf = most_freq / results['basic_info']['n_rows']
    if perc_mf > 0.95:
        data_quality_issues.append(f"Column '{col}' has a single value appearing in more than 95% of rows (possible data entry issue or low variability).")

results['data_quality_issues'] = data_quality_issues

# === Save to JSON ===
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"EDA results saved to {output_file}")