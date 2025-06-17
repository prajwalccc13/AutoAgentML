import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Paths
data_path = 'data/Crop_Yield_Prediction.csv'
output_dir = './output/11'
output_path = os.path.join(output_dir, 'eda_agent.json')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load Data
df = pd.read_csv(data_path)

# Collect results here
eda_results = {}

# 1. Shape, Columns, Data types
eda_results['basic_info'] = {
    'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
    'columns': df.columns.tolist(),
    'data_types': dict(df.dtypes.astype(str))
}

# 2. Missing Values
missing_vals = df.isnull().sum()
missing_perc = (missing_vals / len(df)) * 100
eda_results['missing_values'] = {
    'total_missing': missing_vals.to_dict(),
    'percent_missing': missing_perc.round(2).to_dict()
}

# 3. Cardinality (Unique values)
cardinality = df.nunique()
eda_results['cardinality'] = cardinality.to_dict()

# 4. Numerical Descriptive Statistics (including 'Yield')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
desc_stats = df[num_cols].describe().to_dict()
eda_results['numerical_descriptive_stats'] = desc_stats

# 5. Categorical column frequencies (top 10 values)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_value_counts = {}
for col in cat_cols:
    vc = df[col].value_counts(dropna=False).head(10)
    cat_value_counts[col] = vc.to_dict()
eda_results['categorical_top_frequencies'] = cat_value_counts

# 6. High Cardinality Columns (> 50 unique values)
high_card_cols = cardinality[cardinality > 50].index.tolist()
eda_results['high_cardinality_columns'] = high_card_cols

# 7. Pairwise correlation numerical features and 'Yield'
correlations = {}
if 'Yield' in num_cols:
    for col in num_cols:
        if col == 'Yield':
            continue
        # Exclude all-na columns for correlation
        if df[[col, 'Yield']].dropna().shape[0] > 0:
            corr = df[[col, 'Yield']].corr().iloc[0, 1]
            if pd.notna(corr):
                correlations[col] = float(corr)
eda_results['numerical_correlations_with_Yield'] = correlations

# 8. Outlier Detection (IQR)
outliers = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    indices = df[outlier_mask].index.tolist()
    outliers[col] = {
        'count': int(len(indices)),
        'indices': indices[:1000]  # log only first 1000 for brevity
    }
eda_results['outliers_IQR'] = outliers

# 9. Percentage of zero values for each numerical column
zero_percentage = {}
for col in num_cols:
    zero_count = (df[col] == 0).sum()
    zero_percentage[col] = round((zero_count / df.shape[0]) * 100, 2)
eda_results['zero_percentage_numerical'] = zero_percentage

# 10. Infer Potential Categorical Columns (object type or integer with low cardinality)
potential_cat_cols = []
for col in df.columns:
    unique = cardinality[col]
    dtype = str(df[col].dtype)
    # Object, category, or int/float with few (<15) unique values
    if dtype in ['object', 'category']:
        potential_cat_cols.append(col)
    elif dtype.startswith('int') and unique < 15:
        potential_cat_cols.append(col)
eda_results['potential_categorical_columns'] = potential_cat_cols

# 11. 'Yield' distribution stats
if 'Yield' in df.columns:
    yield_series = df['Yield'].dropna()
    eda_results['Yield_distribution'] = {
        'mean': float(yield_series.mean()),
        'std': float(yield_series.std()),
        'min': float(yield_series.min()),
        'max': float(yield_series.max()),
        'skewness': float(skew(yield_series)),
        'kurtosis': float(kurtosis(yield_series))
    }
else:
    eda_results['Yield_distribution'] = "Yield column not found."

# 12. Duplicate Rows
num_duplicates = df.duplicated().sum()
eda_results['duplicate_rows'] = {
    'total_duplicate_rows': int(num_duplicates)
}

# 13. Columns might require encoding or transformation
columns_to_transform = []
for col in df.columns:
    # Categorical
    if col in potential_cat_cols:
        columns_to_transform.append(col)
    # Binary columns (those with only 2 unique values)
    elif cardinality[col] == 2:
        columns_to_transform.append(col)
    # Skewed numeric features (skewness > 1 or < -1)
    elif col in num_cols and abs(skew(df[col].dropna())) > 1:
        columns_to_transform.append(col)
eda_results['columns_to_encode_or_transform'] = list(set(columns_to_transform))

# 14. Columns with potential data leakage risk (correlation with Yield > 0.8 or < -0.8)
potential_leakage = [col for col, corr in correlations.items() if abs(corr) >= 0.8]
eda_results['potential_data_leakage_columns'] = potential_leakage

# Save JSON output
with open(output_path, 'w') as f:
    json.dump(eda_results, f, indent=4)

print(f"EDA results saved to {output_path}")