import os
import json
import pandas as pd
import numpy as np
from collections import Counter

# Ensure output directory exists
output_dir = './output/id_4'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'eda_agent.json')

# Load dataset
dataset_path = 'data/Crop_Yield_Prediction.csv'
df = pd.read_csv(dataset_path)

# Initialize results dict
results = {}

# 1. Basic metadata
results['metadata'] = {
    "dataset_path": dataset_path,
    "n_rows": int(df.shape[0]),
    "n_columns": int(df.shape[1]),
    "columns": list(df.columns)
}

# 2. Infer data types
inferred_types = {}
for col in df.columns:
    dtype = str(df[col].dtype)
    sample = df[col].dropna().iloc[:100]
    if pd.api.types.is_bool_dtype(df[col]):
        inferred = 'boolean'
    elif pd.api.types.is_numeric_dtype(df[col]):
        inferred = 'numerical'
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        inferred = 'datetime'
    elif pd.api.types.is_categorical_dtype(df[col]):
        inferred = 'categorical'
    else:
        # Heuristic for categoricals
        if sample.nunique() < 0.05*df.shape[0] or df[col].dtype == object:
            inferred = 'categorical'
        else:
            inferred = 'unknown'
    inferred_types[col] = {"pandas_dtype": dtype, "inferred_type": inferred}
results['data_types'] = inferred_types

# 3. Missing values
missing_per_col = {}
total_missing = int(df.isnull().sum().sum())
total_cells = df.shape[0] * df.shape[1]
for col in df.columns:
    missing = int(df[col].isnull().sum())
    percent = float(missing) / float(df.shape[0]) * 100
    missing_per_col[col] = {
        "missing_count": missing,
        "missing_percent": percent
    }
results['missing_values'] = {
    "per_column": missing_per_col,
    "total_missing": total_missing,
    "total_missing_percent": (total_missing / total_cells * 100)
}

# 4. Columns with constant values
constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
results['constant_columns'] = constant_cols

# 5. Cardinality analysis
cardinality = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
low_cardinality = [col for col, val in cardinality.items() if val <= 10]
high_cardinality = [col for col, val in cardinality.items() if val > df.shape[0]*0.5]
results['cardinality'] = {
    "cardinality_per_column": cardinality,
    "low_cardinality_columns": low_cardinality,
    "high_cardinality_columns": high_cardinality
}

# Helper for serialization (handles numpy types)
def serialize(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

# 6. Numerical column summary stats
num_cols = [col for col in df.columns if inferred_types[col]['inferred_type'] == 'numerical']
numerical_summary = {}
for col in num_cols:
    col_data = df[col].dropna()
    summary = {
        "count": int(col_data.count()),
        "mean": float(col_data.mean()) if not col_data.empty else None,
        "median": float(col_data.median()) if not col_data.empty else None,
        "mode": [serialize(x) for x in col_data.mode().tolist()] if not col_data.empty else [],
        "std": float(col_data.std()) if not col_data.empty else None,
        "min": float(col_data.min()) if not col_data.empty else None,
        "max": float(col_data.max()) if not col_data.empty else None,
        "percentiles": {f"{int(q*100)}%": float(col_data.quantile(q)) for q in [0, 0.25, 0.5, 0.75, 1.0]} if not col_data.empty else {},
    }
    numerical_summary[col] = summary
results['numerical_summary'] = numerical_summary

# 7. Categorical column stats
cat_cols = [col for col in df.columns if inferred_types[col]['inferred_type'] == 'categorical']
categorical_summary = {}
for col in cat_cols:
    value_counts = df[col].value_counts(dropna=False)
    mode_value = value_counts.idxmax() if not value_counts.empty else None
    summary = {
        "top_10_most_frequent": value_counts.head(10).to_dict(),
        "top_10_least_frequent": value_counts.tail(10).to_dict(),
        "mode": serialize(mode_value),
        "total_unique": int(df[col].nunique(dropna=False)),
        "value_counts": value_counts.to_dict() if value_counts.size <= 100 else "Too many unique values (see top_10_*)"
    }
    categorical_summary[col] = summary
results['categorical_summary'] = categorical_summary

# 8. Duplicate rows
n_dup = int(df.duplicated().sum())
pct_dup = n_dup / df.shape[0] * 100
results['duplicates'] = {
    "duplicate_count": n_dup,
    "duplicate_percent": pct_dup
}

# 9. Outliers detection using IQR for numerical columns
outlier_info = {}
for col in num_cols:
    col_data = df[col].dropna()
    if len(col_data) < 4:
        outlier_info[col] = {"count": 0, "indices": []}
        continue
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_idx = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
    outlier_info[col] = {"count": len(outlier_idx), "indices": [int(idx) for idx in outlier_idx]}
results['outliers'] = outlier_info

# 10. Skewness and kurtosis
skew_kurt = {}
for col in num_cols:
    col_data = df[col].dropna()
    skewness = float(col_data.skew()) if not col_data.empty else None
    kurtosis = float(col_data.kurt()) if not col_data.empty else None
    skew_kurt[col] = {
        "skewness": skewness,
        "kurtosis": kurtosis
    }
results['skewness_kurtosis'] = skew_kurt

# 11. Target column 'Yield' summary
target_col = 'Yield'
if target_col in df.columns:
    targ = df[target_col].dropna()
    target_info = {
        "count": int(targ.count()),
        "mean": float(targ.mean()) if not targ.empty else None,
        "median": float(targ.median()) if not targ.empty else None,
        "mode": [serialize(x) for x in targ.mode().tolist()] if not targ.empty else [],
        "std": float(targ.std()) if not targ.empty else None,
        "min": float(targ.min()) if not targ.empty else None,
        "max": float(targ.max()) if not targ.empty else None,
        "percentiles": {f"{int(q*100)}%": float(targ.quantile(q)) for q in [0, 0.25, 0.5, 0.75, 1.0]} if not targ.empty else {},
        "missing_count": int(df[target_col].isnull().sum()),
        "missing_percent": float(df[target_col].isnull().sum()) / df.shape[0] * 100,
        "outlier_count": outlier_info[target_col]["count"] if target_col in outlier_info else None,
        "outlier_indices": outlier_info[target_col]["indices"] if target_col in outlier_info else None,
        "skewness": skew_kurt[target_col]["skewness"] if target_col in skew_kurt else None,
        "kurtosis": skew_kurt[target_col]["kurtosis"] if target_col in skew_kurt else None
    }
    results['target_Yield_summary'] = target_info
else:
    results['target_Yield_summary'] = 'Yield column not found'

# 12. Correlation matrix
corr_methods = ['pearson', 'spearman']
correlation_results = {}
for method in corr_methods:
    try:
        cmat = df[num_cols].corr(method=method).replace({np.nan: None}).to_dict()
        correlation_results[method] = cmat
    except Exception:
        correlation_results[method] = "Error computing correlation matrix"
results['correlation_matrix'] = correlation_results

# 13. Highly correlated pairs
high_corr_pairs = {}
threshold = 0.85
for method in corr_methods:
    pairs = []
    if isinstance(correlation_results[method], dict):
        corr_df = pd.DataFrame(correlation_results[method])
        for i, col1 in enumerate(corr_df.columns):
            for j, col2 in enumerate(corr_df.columns):
                if i < j:
                    corr_val = corr_df.loc[col1, col2]
                    if corr_val is not None and abs(corr_val) > threshold:
                        pairs.append({
                            "pair": [col1, col2],
                            "correlation_value": float(corr_val)
                        })
    high_corr_pairs[method] = pairs
results['highly_correlated_pairs'] = high_corr_pairs

# 14. Data/data-entry error checks
data_entry_errors = {}
for col in num_cols:
    # Negative values check for non-negative columns
    if (df[col].min() is not None) and (df[col].min() >= 0):
        continue # skip if all values are non-negative
    negatives = df[df[col] < 0].index.tolist()
    if negatives:
        data_entry_errors[col] = {
            "negative_value_indices": [int(idx) for idx in negatives],
            "count": len(negatives)
        }

for col in cat_cols:
    # Unlikely strings (e.g., typos, blanks)
    value_counts = df[col].astype(str).value_counts()
    suspicious = value_counts[value_counts < 3].index.tolist()
    if suspicious:
        data_entry_errors.setdefault(col, {})
        data_entry_errors[col]['suspicious_category_values'] = suspicious
results['data_entry_errors'] = data_entry_errors

# 15. EDA summary and suggestions
findings = {
    "data_quality_issues": [],
    "transformation_needs": [],
    "feature_engineering_opportunities": []
}

# Data quality
if results['duplicates']['duplicate_count'] > 0:
    findings['data_quality_issues'].append("Duplicate rows detected.")
if any(col in results['constant_columns'] for col in df.columns):
    findings['data_quality_issues'].append("Columns with zero variance exist.")
for col, info in results['missing_values']['per_column'].items():
    if info['missing_percent'] > 5:
        findings['data_quality_issues'].append(f"High missing value percentage in column {col} ({info['missing_percent']:.2f}%)")
if data_entry_errors:
    findings['data_quality_issues'].append("Potential data entry errors detected (e.g., negative values or rare categories).")

# Transformation needs
for col, sk in skew_kurt.items():
    if sk['skewness'] is not None and abs(sk['skewness']) > 1:
        findings['transformation_needs'].append(f"Column {col} is highly skewed (skewness={sk['skewness']:.2f}), consider transformation.")
    if sk['kurtosis'] is not None and sk['kurtosis'] > 5:
        findings['transformation_needs'].append(f"Column {col} has high kurtosis (kurtosis={sk['kurtosis']:.2f}), consider handling outliers.")
for col in constant_cols:
    findings['transformation_needs'].append(f"Column {col} is constant, possibly removable.")

# Feature engineering
for method in corr_methods:
    if results['highly_correlated_pairs'][method]:
        findings['feature_engineering_opportunities'].append(f"Highly correlated features (>0.85) detected. Consider dimensionality reduction or feature selection for method '{method}'.")
if len(low_cardinality) > 0:
    findings['feature_engineering_opportunities'].append("Low-cardinality categorical features suitable for one-hot encoding or effect coding.")

results['eda_summary'] = findings

# --- Save Results ---
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4, default=serialize)

print(f"EDA summary written to {output_path}")