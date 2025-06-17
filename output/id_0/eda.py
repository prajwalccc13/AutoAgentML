import os
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# 1. Create output directory if it doesn't exist
output_dir = './output/id_0'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'eda_agent.json')

# 2. Load data
df = pd.read_csv('data/iris.csv')

eda_result = {}

# Dataset overview
eda_result["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
eda_result["columns"] = list(df.columns)
eda_result["sample_rows"] = df.head(5).replace(np.nan, None).to_dict(orient='records')

# 3. Data types
type_map = {}
for col in df.columns:
    dtype = df[col].dtype
    if pd.api.types.is_bool_dtype(dtype):
        type_map[col] = 'boolean'
    elif pd.api.types.is_numeric_dtype(dtype):
        type_map[col] = 'numerical'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        type_map[col] = 'datetime'
    else:
        type_map[col] = 'categorical'
eda_result["column_types"] = type_map

# 4. Missing and unique values
missing_and_unique = {}
for col in df.columns:
    missing_and_unique[col] = {
        "missing_count": int(df[col].isnull().sum()),
        "unique_count": int(df[col].nunique(dropna=True))
    }
eda_result["missing_and_unique"] = missing_and_unique

# Numerical and categorical columns
numerical_cols = [col for col, typ in type_map.items() if typ == 'numerical']
categorical_cols = [col for col, typ in type_map.items() if typ == 'categorical']

# 5. Statistical summaries for numerical columns
num_stats = {}
for col in numerical_cols:
    stats = {
        "mean": float(df[col].mean(skipna=True)),
        "median": float(df[col].median(skipna=True)),
        "std": float(df[col].std(skipna=True)),
        "min": float(df[col].min(skipna=True)),
        "max": float(df[col].max(skipna=True)),
        "25%": float(df[col].quantile(0.25)),
        "50%": float(df[col].quantile(0.50)),
        "75%": float(df[col].quantile(0.75)),
    }
    num_stats[col] = stats
eda_result["numerical_summaries"] = num_stats

# 6. Categorical summaries
cat_stats = {}
for col in categorical_cols:
    value_counts = df[col].value_counts(dropna=False).to_dict()
    value_counts = {str(k): int(v) for k, v in value_counts.items()}
    cat_stats[col] = {
        "cardinality": int(df[col].nunique(dropna=True)),
        "frequency_distribution": value_counts
    }
eda_result["categorical_summaries"] = cat_stats

# 7. Outlier detection using 1.5*IQR
outlier_info = []
for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
    if outliers:
        outlier_info.append({"column": col, "outlier_indices": outliers})
eda_result["numerical_outliers"] = outlier_info

# 8. Class balance for "variety"
if "variety" in df.columns:
    class_balance = df["variety"].value_counts(dropna=False).to_dict()
    class_balance = {str(k): int(v) for k, v in class_balance.items()}
else:
    class_balance = None
eda_result["class_balance_variety"] = class_balance

# 9. Correlation matrix for numerical columns
if len(numerical_cols) > 1:
    corr_matrix = df[numerical_cols].corr().replace({np.nan: None}).to_dict()
else:
    corr_matrix = None
eda_result["numerical_correlation_matrix"] = corr_matrix

# 10. Skewness and kurtosis
skew_kurt_stats = {}
for col in numerical_cols:
    clean_col = df[col].dropna()
    skew_val = float(skew(clean_col)) if len(clean_col) > 0 else None
    kurt_val = float(kurtosis(clean_col)) if len(clean_col) > 0 else None
    skew_kurt_stats[col] = {"skewness": skew_val, "kurtosis": kurt_val}
eda_result["skewness_and_kurtosis"] = skew_kurt_stats

# 11. Highly correlated numerical feature pairs (>|0.9|)
high_corr_pairs = []
if corr_matrix:
    corr_df = df[numerical_cols].corr().abs()
    np.fill_diagonal(corr_df.values, 0)
    for col1 in corr_df.columns:
        for col2 in corr_df.columns:
            if col1 < col2:
                corr_val = corr_df.loc[col1, col2]
                if corr_val > 0.9:
                    high_corr_pairs.append({"feature_pair": [col1, col2], "correlation": float(df[[col1, col2]].corr().loc[col1, col2])})
eda_result["highly_correlated_features"] = high_corr_pairs

# 12. Top 5 most/least frequent categories for each categorical column
cat_profiles = {}
for col in categorical_cols:
    value_counts = df[col].value_counts(dropna=False)
    most_frequent = value_counts.head(5).to_dict()
    least_frequent = value_counts.tail(5).to_dict()
    cat_profiles[col] = {
        "top_5_most_frequent": {str(k): int(v) for k, v in most_frequent.items()},
        "top_5_least_frequent": {str(k): int(v) for k, v in least_frequent.items()},
    }
eda_result["categorical_feature_profiles"] = cat_profiles

# 13. Duplicate rows
duplicates = df[df.duplicated(keep=False)]
eda_result["duplicates"] = {
    "count": int(duplicates.shape[0]),
    "indices": duplicates.index.tolist()
}

# 14. Data quality report
quality_report = []
for col in df.columns:
    if missing_and_unique[col]['missing_count'] > 0:
        quality_report.append(f"Column '{col}' has {missing_and_unique[col]['missing_count']} missing values.")
    if col in [d['column'] for d in outlier_info]:
        quality_report.append(f"Column '{col}' contains outliers detected by 1.5*IQR rule.")
    if col in categorical_cols and missing_and_unique[col]['unique_count'] > 50:
        quality_report.append(f"Categorical column '{col}' has high cardinality ({missing_and_unique[col]['unique_count']}).")
if eda_result["duplicates"]["count"] > 0:
    quality_report.append(f"Dataset contains {eda_result['duplicates']['count']} duplicate rows.")
eda_result["data_quality_report"] = quality_report

# 15. Feature engineering mapping
fe_map = {}
for col in df.columns:
    if col == "variety":
        continue
    t = type_map[col]
    if t == "numerical":
        fe_map[col] = "numerical"
    elif t == "categorical":
        # Optionally check cardinality
        if missing_and_unique[col]['unique_count'] > 15:
            fe_map[col] = "categorical_high_cardinality"
        else:
            fe_map[col] = "categorical_encoding"
    elif t == "boolean":
        fe_map[col] = "boolean_encoding"
    elif t == "datetime":
        fe_map[col] = "datetime_encoding"
    else:
        fe_map[col] = "unknown"
eda_result["feature_engineering_mapping"] = fe_map

# Save all results to JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(eda_result, f, indent=2, ensure_ascii=False)