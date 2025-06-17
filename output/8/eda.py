import os
import json
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# === Step 1: Load data ===
DATA_PATH = 'data/Crop_Yield_Prediction.csv'
OUTPUT_DIR = './output/8/'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'eda_agent.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
result = {}

# === Step 2: Data Overview ===
data_overview = {
    "first_5_rows": df.head(5).to_dict(orient="records"),
    "shape": {"rows": df.shape[0], "columns": df.shape[1]},
    "column_names": list(df.columns)
}
result["data_overview"] = data_overview

# === Step 3: Column Types ===
column_types = {}
for col in df.columns:
    col_dtype = df[col].dtype
    if pd.api.types.is_bool_dtype(col_dtype):
        column_types[col] = "boolean"
    elif pd.api.types.is_numeric_dtype(col_dtype):
        column_types[col] = "numerical"
    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
        column_types[col] = "datetime"
    elif pd.api.types.is_categorical_dtype(col_dtype):
        column_types[col] = "categorical"
    else:
        unique_ratio = df[col].nunique() / float(df.shape[0])
        if unique_ratio < 0.5:
            column_types[col] = "categorical"
        else:
            column_types[col] = "other"
result["column_types"] = column_types

# === Step 4: Missing Values ===
missing_values = {}
for col in df.columns:
    num_missing = df[col].isna().sum()
    percent_missing = float(num_missing) / df.shape[0] * 100
    missing_values[col] = {
        "n_missing": int(num_missing),
        "percent_missing": round(percent_missing, 3)
    }
result["missing_values"] = missing_values

# === Step 5: Numerical Stats ===
numerical_features = [col for col, typ in column_types.items() if typ == 'numerical']
numerical_stats = {}
for col in numerical_features:
    col_stats = {
        "mean": df[col].mean(skipna=True),
        "median": df[col].median(skipna=True),
        "std": df[col].std(skipna=True),
        "min": df[col].min(skipna=True),
        "max": df[col].max(skipna=True),
        "25%": df[col].quantile(0.25),
        "50%": df[col].quantile(0.5),
        "75%": df[col].quantile(0.75),
        "n_unique": int(df[col].nunique(dropna=True))
    }
    # Ensure serializability (convert numpy types)
    for key in col_stats:
        if isinstance(col_stats[key], (np.generic, np.float32, np.float64, np.int64)):
            col_stats[key] = float(col_stats[key])
    numerical_stats[col] = col_stats
result["numerical_stats"] = numerical_stats

# === Step 6: Categorical Stats ===
categorical_features = [col for col, typ in column_types.items() if typ == 'categorical']
categorical_stats = {}
for col in categorical_features:
    vc = df[col].value_counts(dropna=False)
    top_values = list(vc.head(3).index.astype(str))
    frequencies = list(vc.head(3).values.tolist())
    col_stats = {
        "n_unique": int(df[col].nunique(dropna=True)),
        "top_3": [{"value": v, "count": int(c)} for v, c in zip(top_values, frequencies)]
    }
    categorical_stats[col] = col_stats
result["categorical_stats"] = categorical_stats

# === Step 7: Constant Features ===
constant_features = []
for col in df.columns:
    if df[col].nunique(dropna=False) == 1:
        constant_features.append(col)
result["constant_features"] = {
    "n_constant": len(constant_features),
    "columns": constant_features
}

# === Step 8: Outliers (IQR Method) ===
outliers = {}
for col in numerical_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    # Outlier threshold
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    # Boolean mask
    mask = (df[col] < lower) | (df[col] > upper)
    indices = df.index[mask].tolist()
    outliers[col] = {
        "outlier_count": int(np.sum(mask)),
        "outlier_indices": indices
    }
result["outliers"] = outliers

# === Step 9: Correlations (Pearson & Spearman, wrt 'Yield') ===
if 'Yield' in numerical_features:
    other_numericals = [f for f in numerical_features]
    pearson_corr = df[other_numericals].corr(method='pearson').to_dict()
    spearman_corr = df[other_numericals].corr(method='spearman').to_dict()
    result["correlations"] = {
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }
else:
    result["correlations"] = {}

# === Step 10: Distribution Stats ===
distribution_stats = {}
for col in numerical_features:
    vals = df[col].dropna()
    if len(vals) == 0:
        skew_val = 0
        kurt_val = 0
    else:
        skew_val = float(skew(vals))
        kurt_val = float(kurtosis(vals, fisher=True))
    distribution_stats[col] = {
        "skewness": skew_val,
        "kurtosis": kurt_val
    }
result["distribution_stats"] = distribution_stats

# === Step 11: Rare Categories ===
rare_categories = {}
n_rows = df.shape[0]
for col in categorical_features:
    value_counts = df[col].value_counts(dropna=False)
    rare = value_counts[value_counts < 0.01 * n_rows]
    rare_count = int(rare.sum())
    rare_percent = float(rare_count) / n_rows * 100
    rare_list = list(rare.index.astype(str))
    n_missing = df[col].isnull().sum()
    percent_missing = float(n_missing) / n_rows * 100
    rare_categories[col] = {
        "n_missing": int(n_missing),
        "percent_missing": round(percent_missing, 3),
        "n_rare": int(len(rare)),
        "rare_percent": round(rare_percent, 3),
        "rare_categories": rare_list
    }
result["rare_categories"] = rare_categories

# === Step 12: Duplicates ===
dupes = df.duplicated(keep=False)
duplicate_indices = df.index[dupes].tolist()
duplicate_count = int(dupes.sum())
result["duplicates"] = {
    "duplicate_count": duplicate_count,
    "duplicate_indices": duplicate_indices
}

# === Step 13: Target Analysis (Yield) ===
target_col = 'Yield'
if target_col in df.columns:
    vals = df[target_col].dropna()
    val_steps = {
        "count": int(vals.count()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std()),
    }
    counts = df[target_col].value_counts().head(5).to_dict()
    zero_count = int((df[target_col] == 0).sum())
    neg_count = int((df[target_col] < 0).sum())
    abnormal = {
        "extremely_low_value": float(vals.min()),
        "extremely_high_value": float(vals.max()),
        "comment": (
            "Check if values below {0} or above {1} are abnormal according to domain context"
            .format(vals.min(), vals.max())
        )
    }
    result["target_analysis"] = {
        "value_summary": val_steps,
        "top_5_counts": {str(k): int(v) for k, v in counts.items()},
        "count_zeros": zero_count,
        "count_negative": neg_count,
        "abnormal_range": abnormal
    }
else:
    result["target_analysis"] = {}

# === Step 14: Feature Lists (excluding target) classified by type ===
feature_lists = {
    "numerical": [col for col in numerical_features if col != target_col],
    "categorical": [col for col in categorical_features if col != target_col],
    "other": [col for col, typ in column_types.items() if typ not in ['numerical', 'categorical'] and col != target_col]
}
result["feature_lists"] = feature_lists

# === Save JSON ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"EDA results saved to: {OUTPUT_JSON}")