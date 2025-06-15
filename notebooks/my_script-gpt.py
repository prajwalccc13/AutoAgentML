import pandas as pd
import json
import numpy as np

# Configuration
CSV_FILE_PATH = 'iris.csv'  # Replace with your CSV file path
JSON_LOG_PATH = 'data_analysis_log.json'
TARGET_VARIABLE = 'target'  # Set to None if no target variable
MISSING_VALUE_THRESHOLD = 0.5  # 50%
CORRELATION_THRESHOLD = 0.8
LOW_VARIANCE_THRESHOLD = 0.01  # 1%

# Load dataset
df = pd.read_csv(CSV_FILE_PATH)

# Initialize log dictionary
log = {}

# 1. Dataset shape
log['dataset_shape'] = df.shape

# 2. Data types
log['column_data_types'] = df.dtypes.astype(str).to_dict()

# 3. Missing values
missing_counts = df.isnull().sum()
missing_percentages = (missing_counts / len(df))
log['missing_value_counts_and_percentages'] = {
    col: {
        'missing_count': int(count),
        'missing_percentage': float(perc)
    }
    for col, count, perc in zip(df.columns, missing_counts, missing_percentages)
}

# 4. Numeric summary statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_summary = df[numeric_cols].describe().T
# Add median separately
numeric_summary['median'] = df[numeric_cols].median()
log['numeric_summary_statistics'] = numeric_summary.to_dict()

# 5. Categorical summary statistics
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_summary = {}
for col in categorical_cols:
    value_counts = df[col].value_counts(dropna=False)
    categorical_summary[col] = {
        'count': value_counts.to_dict(),
        'unique': df[col].nunique(),
        'top': value_counts.idxmax(),
        'top_freq': int(value_counts.max())
    }
log['categorical_summary_statistics'] = categorical_summary

# 6. Column cardinalities
column_cardinalities = df.nunique()
log['column_cardinalities'] = column_cardinalities.to_dict()

# 7. Columns with high missing ratio
high_missing_cols = missing_percentages[missing_percentages > MISSING_VALUE_THRESHOLD]
log['high_missing_value_columns'] = {
    col: float(perc)
    for col, perc in high_missing_cols.items()
}

# 8. Outliers in numeric columns (IQR method)
numeric_outliers = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_indices = df.index[outlier_mask].tolist()
    numeric_outliers[col] = {
        'outlier_count': int(outlier_mask.sum()),
        'outlier_indices': outlier_indices
    }
log['numeric_outliers'] = numeric_outliers

# 9. Top 5 categories per categorical column
categorical_top_categories = {}
for col in categorical_cols:
    value_counts = df[col].value_counts().head(5)
    categorical_top_categories[col] = value_counts.to_dict()
log['categorical_top_categories'] = categorical_top_categories

# 10. Low variance numeric columns
low_variance_cols = []
for col in numeric_cols:
    if df[col].var() < LOW_VARIANCE_THRESHOLD:
        low_variance_cols.append(col)
log['low_variance_numeric_columns'] = low_variance_cols

# 11. Target variable distribution (if specified)
if TARGET_VARIABLE and TARGET_VARIABLE in df.columns:
    target_counts = df[TARGET_VARIABLE].value_counts()
    # For classification, show class counts; for regression, maybe stats
    log['target_variable_distribution'] = target_counts.to_dict()

# 12. Correlation matrix of numeric features
correlation_matrix = df[numeric_cols].corr(method='pearson')
# Convert to nested list
log['numeric_feature_correlation_matrix'] = correlation_matrix.values.tolist()

# 13. Highly correlated features with target
high_corr_features = []
if TARGET_VARIABLE and TARGET_VARIABLE in df.columns:
    target_corrs = correlation_matrix[TARGET_VARIABLE]
    for col in numeric_cols:
        if col != TARGET_VARIABLE:
            corr_value = target_corrs[col]
            if abs(corr_value) > CORRELATION_THRESHOLD:
                high_corr_features.append(col)
log['highly_correlated_features_with_target'] = high_corr_features

# 14. Domain-specific data quality notes (manual or heuristic)
# For illustration, let's check for duplicated entries and suspicious patterns
duplicates = df.duplicated()
data_quality_notes = []
if duplicates.any():
    data_quality_notes.append(f"Found {duplicates.sum()} duplicated rows.")
# Check for inconsistent categories (e.g., presence of unexpected categories)
# Example (can be customized)
# For simplicity, assume no specific rules; expand as needed
log['data_quality_notes'] = data_quality_notes

# Save the log to JSON
with open(JSON_LOG_PATH, 'w') as f:
    json.dump(log, f, indent=4)

print(f"Analysis complete. Log saved to {JSON_LOG_PATH}.")