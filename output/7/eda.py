import os
import json
import pandas as pd
import numpy as np

# === 1. Load dataset ===
data_path = 'data/banana_quality.csv'
df = pd.read_csv(data_path)

# Prepare result dict
eda_results = {}

# === 2. Basic Info ===
shape = df.shape
eda_results['shape'] = {'rows': shape[0], 'columns': shape[1]}

columns = list(df.columns)
eda_results['columns'] = columns

dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
eda_results['dtypes'] = dtypes

# === 3. Missing Values ===
missing_count = df.isnull().sum().to_dict()
missing_pct = (df.isnull().mean()*100).round(2).to_dict()
eda_results['missing_values'] = {'count': missing_count, 'percentage': missing_pct}

# === 4. Statistical Summaries for Numeric Columns ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
stats_summary = {}
for col in numeric_cols:
    summary = df[col].describe(percentiles=[.25, .5, .75]).to_dict()
    # Ensure inclusion of median
    summary['median'] = float(df[col].median())
    stats_summary[col] = {k: float(v) for k, v in summary.items()}
eda_results['numeric_summary'] = stats_summary

# === 5. Unique Values & Categorical Columns ===
unique_counts = {}
unique_values = {}
cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
for col in df.columns:
    unique_counts[col] = int(df[col].nunique())
    if col in cat_cols:
        unique_values[col] = df[col].dropna().unique().tolist()
eda_results['unique_values'] = {
    'counts': unique_counts,
    'categorical': unique_values
}

# === 6. Target 'Quality' Distribution ===
target = 'Quality'
if target in df.columns:
    target_counts = df[target].value_counts(dropna=False)
    target_percentages = df[target].value_counts(normalize=True, dropna=False) * 100
    quality_distribution = {
        'count': target_counts.to_dict(),
        'percentage': target_percentages.round(2).to_dict()
    }
    eda_results['quality_distribution'] = quality_distribution

# === 7. Top 10 Value Counts for Categorical Columns ===
cat_value_counts = {}
for col in cat_cols:
    counts = df[col].value_counts(dropna=False).head(10)
    cat_value_counts[col] = counts.to_dict()
eda_results['top_10_value_counts_categorical'] = cat_value_counts

# === 8. Outliers by IQR in Numeric Columns ===
iqr_outliers = {}
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_indices = df.index[outlier_mask].tolist()
    iqr_outliers[col] = {
        'count': int(np.sum(outlier_mask)),
        'indices': outlier_indices if len(outlier_indices) < 1000 else outlier_indices[:1000] # prevent too big JSON
    }
eda_results['iqr_outliers'] = iqr_outliers

# === 9. Correlation Matrix (Numeric Columns, Pearson) ===
if numeric_cols:
    corr = df[numeric_cols].corr(method='pearson')
    corr_dict = corr.round(3).to_dict()
    eda_results['correlation_matrix_pearson'] = corr_dict
else:
    eda_results['correlation_matrix_pearson'] = {}

# === 10. Near-Constant Columns (most frequent value >95%) ===
near_constants = []
for col in df.columns:
    top_freq = df[col].value_counts(normalize=True, dropna=False).max()
    if top_freq > 0.95:
        near_constants.append(col)
eda_results['near_constant_columns'] = {
    'count': len(near_constants),
    'names': near_constants
}

# === 11. Duplicated Rows ===
dup_mask = df.duplicated()
dup_indices = df.index[dup_mask].tolist()
eda_results['duplicated_rows'] = {
    'count': int(np.sum(dup_mask)),
    'indices': dup_indices if len(dup_indices) < 1000 else dup_indices[:1000] # prevent too big JSON
}

# === 12. Write results to JSON ===
os.makedirs('./output/7/', exist_ok=True)
output_file = './output/7/eda_agent.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(eda_results, f, indent=4)

print(f"EDA successfully saved to {output_file}")