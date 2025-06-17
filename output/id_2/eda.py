import os
import json
import pandas as pd
import numpy as np

# Paths
INPUT_PATH = 'data/banana_quality.csv'
OUTPUT_DIR = './output/id_2/'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'eda_agent.json')

# Ensure output path exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the data
df = pd.read_csv(INPUT_PATH)
results = {}

# 1. Log shape and column names
results['shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}
results['column_names'] = list(df.columns)

# 2. Inferred data types
results['data_types'] = df.dtypes.astype(str).to_dict()

# 3. Missing values
missing_counts = df.isnull().sum().to_dict()
missing_percent = (df.isnull().mean() * 100).to_dict()
results['missing_values'] = {
    col: {
        'count': int(missing_counts[col]),
        'percent': float(round(missing_percent[col], 3))
    }
    for col in df.columns
}

# 4. Unique value counts (cardinality)
results['unique_value_counts'] = df.nunique().to_dict()

# 5. Numeric column statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
stats_numeric = {}
for col in numeric_cols:
    s = df[col]
    desc = {
        'count': int(s.count()),
        'mean': float(np.nanmean(s)),
        'median': float(np.nanmedian(s)),
        'std': float(np.nanstd(s, ddof=1)),
        'min': float(np.nanmin(s)),
        '25%': float(np.nanpercentile(s, 25)),
        '75%': float(np.nanpercentile(s, 75)),
        'max': float(np.nanmax(s))
    }
    stats_numeric[col] = desc
results['numeric_column_statistics'] = stats_numeric

# 6. Top 10 frequencies for categorical columns
cat_cols = df.select_dtypes(include=['object','category']).columns
cat_freqs = {}
for col in cat_cols:
    val_counts = df[col].value_counts(dropna=False).head(10)
    cat_freqs[col] = {str(k): int(v) for k,v in val_counts.items()}
results['top_10_frequencies_categorical'] = cat_freqs

# 7. Duplicate row detection
duplicates = df.duplicated()
num_duplicates = int(duplicates.sum())
duplicate_indices = df[duplicates].index.tolist()
results['duplicate_rows'] = {
    'num_duplicates': num_duplicates,
    'duplicate_indices': duplicate_indices
}

# 8. Outlier detection using IQR
iqr_outliers = {}
for col in numeric_cols:
    col_out = {}
    s = df[col]
    q1 = np.nanpercentile(s, 25)
    q3 = np.nanpercentile(s, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out_idx = df[(s < lower) | (s > upper)].index.tolist()
    col_out['num_outliers'] = len(out_idx)
    col_out['outlier_indices'] = out_idx
    iqr_outliers[col] = col_out
results['iqr_outliers'] = iqr_outliers

# 9. Numeric correlation matrix
corr_mat = df[numeric_cols].corr().round(4)
corr_json = corr_mat.replace({np.nan: None}).to_dict()
results['numeric_correlation_matrix'] = corr_json

# 10. Target column class distribution ('Quality')
target_col = 'Quality'
if target_col in df.columns:
    tar_counts = df[target_col].value_counts(dropna=False)
    tar_perc = (df[target_col].value_counts(normalize=True, dropna=False)*100).round(2)
    results['target_class_distribution'] = {
        'counts': {str(k): int(v) for k, v in tar_counts.items()},
        'percentages': {str(k): float(v) for k, v in tar_perc.items()}
    }
else:
    results['target_class_distribution'] = 'Column not found.'

# 11. Univariate relationships with target
univariate_stats = {}
for col in df.columns:
    if col == target_col:
        continue
    if col in numeric_cols:
        # Group by target, show mean/std/min/max/median/25th/75th
        grp_stats = df.groupby(target_col)[col].agg([
            'count', 'mean', 'std', 'min', 'median', 'max',
            lambda x: np.nanpercentile(x, 25),
            lambda x: np.nanpercentile(x, 75)
        ])
        grp_stats.columns = ['count', 'mean', 'std', 'min', 'median', 'max', '25%', '75%']
        formatted = grp_stats.round(4).replace({np.nan: None}).to_dict(orient='index')
        univariate_stats[col] = formatted
    else:
        # Cross-tabulation
        ctab = pd.crosstab(df[col], df[target_col], dropna=False)
        ctab = ctab.astype(int)
        univariate_stats[col] = ctab.to_dict()
results['univariate_relationships'] = univariate_stats

# 12. Identify problematic columns
problematic = {'high_cardinality': [], 'high_missingness': [], 'constant': []}
num_rows = df.shape[0]
for col in df.columns:
    # High cardinality: >50 or more than 10% of rows
    if df[col].nunique(dropna=False) > 50 or df[col].nunique(dropna=False) > 0.1 * num_rows:
        problematic['high_cardinality'].append(col)
    # High missingness: >30%
    if (df[col].isnull().mean() > 0.3):
        problematic['high_missingness'].append(col)
    # Constant
    if df[col].nunique(dropna=False) <= 1:
        problematic['constant'].append(col)
results['problematic_columns'] = problematic

# Save to JSON
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"EDA results saved to {OUTPUT_PATH}")