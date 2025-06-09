import pandas as pd
import numpy as np
import json
from scipy import stats # This import is not used in the provided snippet, but keeping it as it was in the original

# Load your CSV data
file_path = 'iris.csv' # Make sure 'iris.csv' exists in the same directory or provide the full path
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}' into a DataFrame.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the CSV file is in the correct directory.")
    exit() # Exit if the file isn't found, as subsequent tasks will fail.

# --- Helper function to convert NumPy types to native Python types ---
def convert_numpy_types(obj):
    """
    Recursively converts NumPy numeric types within a dictionary or list
    to native Python types (int, float).
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert NumPy arrays to Python lists
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

# 1. Data shape, first few rows, column data types
data_info = {
    'shape': df.shape,
    # Convert DataFrame head to JSON-friendly format
    'head': df.head().to_dict(orient='records'),
    'dtypes': df.dtypes.apply(lambda dt: dt.name).to_dict()
}

# 2. Missing values per column
missing_info = {}
total_rows = df.shape[0]
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_percent = (missing_count / total_rows) * 100
    missing_info[col] = {
        'missing_count': int(missing_count), # Ensure int type
        'missing_percent': float(missing_percent) # Ensure float type
    }

# 3. Data types and unique counts
dtype_unique_counts = {}
for col in df.columns:
    dtype_unique_counts[col] = {
        'dtype': str(df[col].dtype),
        'unique_count': int(df[col].nunique()) # Ensure int type
    }

# 4. Descriptive stats for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
numeric_stats = {}
for col in numerical_cols:
    data_series = df[col].dropna()
    numeric_stats[col] = {
        'mean': float(data_series.mean()),
        'median': float(data_series.median()),
        'std': float(data_series.std()),
        'min': float(data_series.min()),
        'max': float(data_series.max()),
        '25%': float(data_series.quantile(0.25)),
        '50%': float(data_series.quantile(0.50)),
        '75%': float(data_series.quantile(0.75))
    }

# 5. Descriptive stats for categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_stats = {}
for col in categorical_cols:
    data_series = df[col].dropna()
    mode_value = data_series.mode().iloc[0] if not data_series.mode().empty else None
    categorical_stats[col] = {
        'mode': mode_value,
        'unique_count': int(data_series.nunique()) # Ensure int type
    }

# 6. Cardinality of categorical columns
categorical_cardinality = {}
for col in categorical_cols:
    count_unique = df[col].nunique()
    pct_unique = (count_unique / total_rows) * 100
    categorical_cardinality[col] = {
        'unique_count': int(count_unique), # Ensure int type
        'percentage': float(pct_unique) # Ensure float type
    }

# 7. Outliers detection in numerical columns via IQR
outliers_info = {}
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_info[col] = {
        'outlier_count': int(outliers.shape[0]) # Ensure int type
    }

# 8. High missing ratio features (>50%)
high_missing_cols = [col for col in df.columns if (df[col].isnull().mean() * 100) > 50]

# 9. Correlation matrix and high correlation pairs
# Convert correlation matrix to dictionary, then apply conversion to values
corr_matrix = df[numerical_cols].corr()
corr_matrix_dict = {
    idx: {col: float(val) for col, val in row.items()}
    for idx, row in corr_matrix.to_dict(orient='index').items()
}

# Find pairs with correlation above threshold
threshold = 0.8
high_corr_pairs = []
for i in range(len(numerical_cols)):
    for j in range(i+1, len(numerical_cols)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            col1 = numerical_cols[i]
            col2 = numerical_cols[j]
            high_corr_pairs.append({ 'pair': (col1, col2), 'correlation': float(corr_value) }) # Ensure float type

# 10. Distribution of target variable, if present
target_column = 'species'  # Assuming 'species' is the target variable for Iris dataset
target_dist = None
if target_column in df.columns:
    target_dist = df[target_column].value_counts().to_dict()

# 11. High cardinality categorical features
high_cardinality_threshold = 100
high_card_cols = { col: int(count) for col, count in categorical_stats.items() if count['unique_count'] > high_cardinality_threshold }

# 12. Top N category frequencies per categorical feature
top_N = 10
top_categories = {}
for col in categorical_cols:
    top_categories[col] = {k: int(v) for k, v in df[col].value_counts().head(top_N).to_dict().items()} # Ensure int type

# 13. Duplicated rows
duplicates = df[df.duplicated()]
duplicated_info = {
    'duplicates_count': int(duplicates.shape[0]), # Ensure int type
    'sample_indices': duplicates.index.tolist()
}

# 14. Data ranges and unique value counts
data_ranges = {}
for col in numerical_cols:
    data_ranges[col] = {
        'min': float(df[col].min()), # Ensure float type
        'max': float(df[col].max())  # Ensure float type
    }
for col in categorical_cols:
    data_ranges[col] = {
        'unique_values': int(df[col].nunique()) # Ensure int type
    }

# 15. Percentage of missing data per feature
missing_percentage = {}
for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / total_rows) * 100
    missing_percentage[col] = {
        'missing_count': int(missing_count), # Ensure int type
        'missing_percent': float(missing_pct) # Ensure float type
    }

# 16. Low variance features (near-constant)
low_variance_features = {}
for col in df.columns:
    if df[col].nunique() == 1:
        variance = 0.0 # Use float for consistency
    elif pd.api.types.is_numeric_dtype(df[col]):
        variance = df[col].var()
    else:
        # For categorical, consider the proportion of the most frequent category
        variance = df[col].value_counts(normalize=True).iloc[0] if not df[col].isnull().all() else 0.0

    # Threshold: variance close to 0 (or proportion near 1 for categorical)
    if (isinstance(variance, (int, float)) and variance < 1e-5):
        low_variance_features[col] = float(variance) # Ensure float type

# 17. Data quality issues summary
# Example: check for inconsistent data types or unexpected values
# For simplicity, we'll prepare a placeholder. Real implementation would involve more checks.
data_quality_issues = []

# Compile all logs into a dictionary
all_logs = {
    'data_info': data_info,
    'missing_info': missing_info,
    'dtype_unique_counts': dtype_unique_counts,
    'numeric_stats': numeric_stats, # Add numeric stats to all_logs
    'categorical_stats': categorical_stats,
    'categorical_cardinality': categorical_cardinality,
    'outliers_info': outliers_info,
    'high_missing_cols': high_missing_cols,
    'correlation_matrix': corr_matrix_dict, # Use the converted dict
    'high_correlation_pairs': high_corr_pairs,
    'target_distribution': target_dist,
    'high_cardinality_features': high_card_cols,
    'top_categories': top_categories,
    'duplicated_rows': duplicated_info,
    'data_ranges': data_ranges,
    'missing_percentage': missing_percentage,
    'low_variance_features': low_variance_features,
    'data_quality_issues': data_quality_issues
}

# Apply the recursive conversion function to the entire logs dictionary
final_logs = convert_numpy_types(all_logs)

# Save all logs to JSON file
output_file_path = 'data_analysis_logs.json'
with open(output_file_path, 'w') as f:
    json.dump(final_logs, f, indent=4)

print(f"Data analysis logs saved to '{output_file_path}'")