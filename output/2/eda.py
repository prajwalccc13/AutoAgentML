import os
import json
import numpy as np
import pandas as pd

# ==== 1. Load the dataset ====
data_path = 'data/banana_quality.csv'
df = pd.read_csv(data_path)

output_dir = './output/2/'
output_file = os.path.join(output_dir, 'eda_agent.json')
os.makedirs(output_dir, exist_ok=True)

eda_results = {}

# ==== 2. Dataset Shape ====
eda_results['dataset_shape'] = {
    'n_rows': int(df.shape[0]),
    'n_columns': int(df.shape[1])
}

# ==== 3. Columns - Names, Data Types, Type Flags ====
column_info = {}
for col in df.columns:
    dtype = str(df[col].dtype)
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    is_categorical = pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
    type_flag = (
        'numeric' if is_numeric else
        'datetime' if is_datetime else
        'categorical' if is_categorical else
        'other'
    )
    column_info[col] = {
        'dtype': dtype,
        'inferred_type': type_flag
    }
eda_results['columns_info'] = column_info

# ==== 4. Missing Values ====
missing_info = {}
n_rows = df.shape[0]
for col in df.columns:
    n_missing = int(df[col].isnull().sum())
    pct_missing = float(n_missing / n_rows * 100)
    missing_info[col] = {
        'missing_count': n_missing,
        'missing_pct': pct_missing
    }
eda_results['missing_values'] = missing_info

# ==== 5. Numeric Columns - Summaries ====
numeric_summary = {}
numeric_cols = [col for col, info in column_info.items() if info['inferred_type'] == 'numeric']
for col in numeric_cols:
    col_data = df[col].dropna()
    description = col_data.describe(percentiles=[.25, .5, .75])
    numeric_summary[col] = {
        'mean': float(description['mean']),
        'median': float(col_data.median()),
        'std': float(description['std']),
        'min': float(description['min']),
        'max': float(description['max']),
        '25%': float(description['25%']),
        '50%': float(description['50%']),
        '75%': float(description['75%']),
        'n_unique': int(col_data.nunique())
    }
eda_results['numeric_summaries'] = numeric_summary

# ==== 6. Categorical Columns - Summaries ====
cat_summary = {}
cat_cols = [col for col, info in column_info.items() if info['inferred_type'] == 'categorical']
for col in cat_cols:
    col_data = df[col].astype('str').fillna('nan')
    value_counts = col_data.value_counts()
    total = len(col_data)
    top10 = value_counts.head(10)
    frequencies = (value_counts / total).to_dict()
    rare_count = int((value_counts / total < 0.01).sum())
    cat_summary[col] = {
        'n_unique': int(value_counts.shape[0]),
        'top_10': [{ 'category': k, 'count': int(v)} for k, v in top10.items()],
        'n_rare_categories': rare_count
    }
eda_results['categorical_summaries'] = cat_summary

# ==== 7. Target Column "Quality" - Distribution ====
quality_info = {}
if 'Quality' in df.columns:
    q_data = df['Quality'].astype('str').fillna('nan')
    q_counts = q_data.value_counts().sort_values(ascending=False)
    total = len(q_data)
    q_percent = (q_counts / total * 100).round(2)
    underrepresented = [cls for cls, pct in q_percent.items() if pct < 5]
    
    quality_info = {
        'unique_classes': list(q_counts.index),
        'class_distribution': [
            {'class': k, 'count': int(v), 'pct': float(q_percent[k])}
            for k, v in q_counts.items()
        ],
        'underrepresented_classes': underrepresented
    }
eda_results['quality_column'] = quality_info

# ==== 8. High Cardinality Categorical Columns ====
high_cardinality = []
for col in cat_cols:
    n_uniques = int(df[col].nunique())
    if n_uniques > 50:
        high_cardinality.append({'column': col, 'cardinality': n_uniques})
eda_results['high_cardinality_categoricals'] = high_cardinality

# ==== 9. Numeric Columns - Outliers via IQR ====
outlier_info = {}
for col in numeric_cols:
    col_data = df[col].dropna()
    if col_data.empty:
        outlier_info[col] = {'outliers': []}
        continue
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    mask = (col_data < lower) | (col_data > upper)
    outlier_indices = col_data.index[mask].tolist()
    outlier_values = col_data[mask].tolist()
    outlier_info[col] = {
        'outlier_indices': [int(idx) for idx in outlier_indices],
        'outlier_values': [float(val) for val in outlier_values]
    }
eda_results['outliers_iqr'] = outlier_info

# ==== 10. Pairwise Correlations ====
pairwise_corr = None
corr_with_target = None
most_correlated = None
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr(method='pearson')
    # JSON serializable version:
    pairwise_corr = corr_matrix.round(4).fillna('').to_dict()
else:
    pairwise_corr = 'Not enough numeric columns for correlation.'

# Correlation with Target:
correlated_features = []
target_col = 'Quality'
if target_col in df.columns and target_col in numeric_cols:
    corrs = df[numeric_cols].corrwith(df[target_col]).drop(target_col)
    corr_with_target = corrs.round(4).to_dict()
    most_correlated = sorted(corr_with_target.items(),
                             key=lambda x: abs(x[1]),
                             reverse=True)
    most_correlated = [{'feature': k, 'correlation_with_target': v} for k,v in most_correlated]
else:
    corr_with_target = 'Target not numeric or not present.'
    most_correlated = 'Target not numeric or not present.'
eda_results['pairwise_correlations'] = pairwise_corr
eda_results['correlation_with_target'] = corr_with_target
eda_results['most_correlated_with_target'] = most_correlated

# ==== 11. Feature Engineering Recommendations ====
feature_recommendations = {}
for col, info in column_info.items():
    dtype = info['inferred_type']
    n_missing = missing_info[col]['missing_count']

    # Missing value handling
    if dtype == 'numeric':
        if n_missing > 0:
            missing_strategy = 'Impute with median or mean depending on distribution/skewness.'
        else:
            missing_strategy = 'No missing values.'
        outlier_treatment = 'Consider capping or removing points outside 1.5*IQR.' \
            if outlier_info.get(col, {}).get('outlier_indices') else 'No outliers detected.'
        encoding = 'No encoding needed for numeric.'
    elif dtype == 'categorical':
        if n_missing > 0:
            missing_strategy = "Impute missing as 'Unknown' or most frequent category."
        else:
            missing_strategy = 'No missing values.'
        n_uniques = int(df[col].nunique())
        if n_uniques > 50:
            encoding = 'Use target/ordinal encoding or grouping rare categories for high cardinality.'
        else:
            encoding = 'One-hot encoding or label encoding suitable.'
        outlier_treatment = 'Not applicable.'
    elif dtype == 'datetime':
        missing_strategy = "Consider fill with median or mode date, or flag as missing."
        outlier_treatment = "Investigate outliers as possibly erroneous timestamps."
        encoding = "Extract features (year, month, day, etc.) or encode as ordinal."
    else:
        missing_strategy = 'Custom handling required.'
        outlier_treatment = 'Custom handling required.'
        encoding = 'Custom handling required.'
    feature_recommendations[col] = {
        'missing_handling': missing_strategy,
        'outlier_handling': outlier_treatment,
        'encoding': encoding
    }
eda_results['feature_recommendations'] = feature_recommendations

# ==== 12. Save to JSON ====
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(eda_results, f, indent=2, ensure_ascii=False)