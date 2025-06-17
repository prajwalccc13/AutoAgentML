import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Directory and file paths
output_dir = './output/12/'
output_file = os.path.join(output_dir, 'eda_agent.json')

# Prepare the output dictionary
eda_results = {}

# 1. Load the dataset, check for errors, log shape
try:
    df = pd.read_csv('data/Crop_Yield_Prediction.csv')
    load_status = 'success'
    error_msg = ''
    shape = df.shape
except Exception as e:
    df = None
    load_status = 'fail'
    error_msg = str(e)
    shape = None

eda_results['dataset_loading'] = {
    'status': load_status,
    'shape': {'rows': shape[0], 'columns': shape[1]} if shape else None,
    'error': error_msg
}

# If loading failed, write and exit.
if df is None:
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(eda_results, f, indent=4)
    exit()

# 2. Extract and log column names and data types
cols_and_dtypes = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
eda_results['columns_and_types'] = cols_and_dtypes

# 3. Missing values: percentage and count per column
missing_info = {
    col: {
        'missing_count': int(df[col].isna().sum()),
        'missing_pct': float((df[col].isna().mean()) * 100)
    } for col in df.columns
}
eda_results['missing_values'] = missing_info

# 4. Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
eda_results['numerical_columns'] = numerical_cols
eda_results['categorical_columns'] = categorical_cols

# 5. Descriptive statistics for numerical columns
desc_stats = {}
for col in numerical_cols:
    col_data = df[col].dropna()
    if not col_data.empty:
        desc_stats[col] = {
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            '25%': float(col_data.quantile(0.25)),
            '50%': float(col_data.quantile(0.50)),
            '75%': float(col_data.quantile(0.75)),
        }
    else:
        desc_stats[col] = None
eda_results['numerical_descriptive_stats'] = desc_stats

# 6. Unique values for categorical columns
cat_cardinality = {col: int(df[col].nunique(dropna=True)) for col in categorical_cols}
eda_results['categorical_cardinality'] = cat_cardinality

# 7. Constant features (single unique value)
single_unique = []
for col in df.columns:
    if df[col].nunique(dropna=False) == 1:
        single_unique.append(col)
eda_results['constant_features'] = single_unique

# 8. Yield column distribution summary
target_col = 'Yield'
if target_col in numerical_cols:
    col_data = df[target_col].dropna()
    target_dist = {
        'min': float(col_data.min()),
        'max': float(col_data.max()),
        'mean': float(col_data.mean()),
        'std': float(col_data.std()),
        'skewness': float(skew(col_data)),
        'kurtosis': float(kurtosis(col_data)),
        'percentiles': {str(p): float(col_data.quantile(p/100.)) for p in [1, 5, 25, 50, 75, 95, 99]}
    }
else:
    target_dist = 'Yield column is not numerical.'
eda_results['Yield_distribution'] = target_dist

# 9. Outlier detection (1.5*IQR)
outlier_counts = {}
for col in numerical_cols:
    col_data = df[col].dropna()
    if col_data.empty:
        outlier_counts[col] = None
        continue
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    outlier_counts[col] = int(outliers.count())
eda_results['outlier_counts_1.5xIQR'] = outlier_counts

# 10. Pairwise Pearson correlations
corr_matrix = df[numerical_cols].corr(method='pearson').round(4)
eda_results['pearson_correlations'] = corr_matrix.fillna('').to_dict()

# 11. Duplicate row count
dup_count = int(df.duplicated().sum())
eda_results['duplicate_rows'] = dup_count

# 12. Highly imbalanced categorical features (>90% in one category)
imbalanced_cats = {}
for col in categorical_cols:
    non_null_cnt = df[col].notnull().sum()
    if non_null_cnt == 0:
        continue
    top_freq = df[col].value_counts(dropna=True).iloc[0] / non_null_cnt
    if top_freq > 0.90:
        imbalanced_cats[col] = {
            'most_freq_value': str(df[col].value_counts().index[0]),
            'frequency': float(top_freq * 100)
        }
eda_results['highly_imbalanced_categorical'] = imbalanced_cats

# 13. Likely data-entry errors in numerical columns (e.g., negatives)
numerical_entry_issues = {}
for col in numerical_cols:
    negatives = None
    if 'yield' in col.lower() or 'area' in col.lower() or 'rain' in col.lower() or 'temperature' in col.lower():
        negatives = int((df[col] < 0).sum())
        if negatives > 0:
            numerical_entry_issues[col] = {
                'negative_value_count': negatives,
                'note': 'Check if negatives are valid for this column.'
            }
eda_results['numerical_entry_possible_issues'] = numerical_entry_issues

# 14. Preprocessing or cleaning action summary
preprocessing_recommendations = []

# Missing value handling
for col, missing_stat in missing_info.items():
    if missing_stat['missing_pct'] > 0:
        if col in numerical_cols:
            preprocessing_recommendations.append(
                f"Consider imputing missing values in '{col}' with mean/median or suitable method."
            )
        elif col in categorical_cols:
            preprocessing_recommendations.append(
                f"Consider imputing missing values in '{col}' with mode or special category."
            )

# Constant columns
if single_unique:
    preprocessing_recommendations.append(
        f"Remove constant columns: {', '.join(single_unique)}"
    )

# Suggest removing duplicates
if dup_count > 0:
    preprocessing_recommendations.append(
        f"Remove {dup_count} duplicate rows."
    )

# Imbalanced categoricals
if imbalanced_cats:
    for col in imbalanced_cats:
        preprocessing_recommendations.append(
            f"Consider combining infrequent categories or removing feature '{col}' due to imbalance."
        )

# Datatype conversions
for col, dtype in cols_and_dtypes.items():
    if dtype == 'object' and col not in categorical_cols:
        preprocessing_recommendations.append(
            f"Check if column '{col}' should be categorical."
        )

# Outlier actions
for col, out_count in outlier_counts.items():
    if out_count is not None and out_count > 0:
        preprocessing_recommendations.append(
            f"Investigate and handle {out_count} detected outliers in '{col}'."
        )

# Entry issues
if numerical_entry_issues:
    for col in numerical_entry_issues:
        preprocessing_recommendations.append(
            f"Check for and correct invalid negative values in '{col}'."
        )

eda_results['preprocessing_recommendations'] = preprocessing_recommendations

# Write the results JSON
os.makedirs(output_dir, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(eda_results, f, indent=4)

print(f"EDA results saved to {output_file}")