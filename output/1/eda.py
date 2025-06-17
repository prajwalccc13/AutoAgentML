import os
import json
import pandas as pd
import numpy as np
from scipy import stats

# --------------- Helper Functions ----------------

def identify_feature_types(df):
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = 'numerical'
        else:
            types[col] = 'categorical'
    return types

def get_descriptive_stats(df):
    desc = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict = {}
        vals = df[col].dropna()
        stats_dict['mean'] = vals.mean()
        stats_dict['median'] = vals.median()
        stats_dict['std'] = vals.std()
        stats_dict['min'] = vals.min()
        stats_dict['max'] = vals.max()
        stats_dict['quartiles'] = {
            'q1': vals.quantile(0.25),
            'q2': vals.quantile(0.50),
            'q3': vals.quantile(0.75)
        }
        desc[col] = stats_dict
    return desc

def get_categorical_stats(df, feature_types):
    cat_stats = {}
    for col, col_type in feature_types.items():
        if col_type == 'categorical':
            value_counts = df[col].value_counts(dropna=False)  # include NaN
            stats = {
                'unique_values': sorted([str(x) for x in df[col].dropna().unique()]),
                'num_unique_values': df[col].nunique(dropna=True),
                'value_counts': value_counts.astype(int).to_dict()
            }
            cat_stats[col] = stats
    return cat_stats

def check_cardinality(df, feature_types, threshold=10):
    card = {}
    high_card_cols = []
    for col, col_type in feature_types.items():
        if col_type == 'categorical':
            n_unique = df[col].nunique(dropna=True)
            card[col] = n_unique
            if n_unique > threshold:
                high_card_cols.append(col)
    return card, high_card_cols

def detect_outliers_iqr(df):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col]
        q1 = np.percentile(vals.dropna(), 25)
        q3 = np.percentile(vals.dropna(), 75)
        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr
        outlier_indices = df.index[(vals < lower) | (vals > upper)].tolist()
        outliers[col] = outlier_indices
    return outliers

def detect_outliers_zscore(df, threshold=3):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col]
        zscores = np.abs(stats.zscore(vals.dropna(), nan_policy='omit'))
        outlier_indices = vals.dropna().index[zscores > threshold].tolist()
        outliers[col] = outlier_indices
    return outliers

def get_correlation(df):
    corr_matrix = df.select_dtypes(include=[np.number]).corr().to_dict()
    high_corr_pairs = []
    matrix = df.select_dtypes(include=[np.number]).corr()
    for col1 in matrix.columns:
        for col2 in matrix.columns:
            if col1 != col2:
                corr_val = matrix.loc[col1, col2]
                if abs(corr_val) > 0.8:
                    pair = tuple(sorted([col1, col2]))
                    if pair not in high_corr_pairs:
                        high_corr_pairs.append({'pair': pair, 'corr': corr_val})
    # Remove duplicates
    unique_high_corr_pairs = []
    seen = set()
    for d in high_corr_pairs:
        pair = tuple(d['pair'])
        if pair not in seen:
            unique_high_corr_pairs.append(d)
            seen.add(pair)
    return corr_matrix, unique_high_corr_pairs

def get_histogram_counts(df, bins=10):
    histograms = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        counts, bin_edges = np.histogram(vals, bins=bins)
        histograms[col] = {
            'bin_edges': bin_edges.tolist(),
            'counts': counts.tolist()
        }
    return histograms

def rare_classes_and_imbalance(df, feature_types, threshold=0.05, target='variety'):
    results = {}
    n = len(df)
    for col, typ in feature_types.items():
        if typ == 'categorical':
            counts = df[col].value_counts(dropna=False)
            props = counts / n
            rare_classes = props[props < threshold].to_dict()
            missing_prop = df[col].isnull().sum() / n
            results[col] = {
                'rare_classes': list(map(str, rare_classes.keys())),
                'rare_classes_proportion': {str(k): float(v) for k, v in rare_classes.items()},
                'missing_proportion': missing_prop
            }
    # Class imbalance in target
    if target in df.columns:
        target_counts = df[target].value_counts()
        target_props = (target_counts / n).to_dict()
        imbalance_stats = {
            'class_proportions': {str(k): float(v) for k, v in target_props.items()},
            'min_class_proportion': float(min(target_props.values())),
            'max_class_proportion': float(max(target_props.values())),
            'imbalance_ratio': float(max(target_props.values()) / min(target_props.values())) if min(target_props.values()) > 0 else None
        }
        results[target]['class_imbalance_summary'] = imbalance_stats
    return results

def recommended_features(df, missing_stats, high_card_cols, high_corr_pairs, feature_types, missing_thresh=0.2):
    # Exclude columns with too much missing data, high cardinality, or high correlation
    total = len(df)
    flags = {}
    for col in df.columns:
        flags[col] = {
            'high_missing': False,
            'high_cardinality': False,
            'high_correlation': False
        }
    # High missing
    for col, miss in missing_stats['missing_values'].items():
        if miss['percentage'] > missing_thresh * 100:
            flags[col]['high_missing'] = True
    # High cardinality
    for col in high_card_cols:
        flags[col]['high_cardinality'] = True
    # High correlation
    for pair in high_corr_pairs:
        for col in pair['pair']:
            flags[col]['high_correlation'] = True
    # Recommend those with all flags False
    recommended = [col for col, f in flags.items() if not any(f.values()) and feature_types[col] != 'categorical']
    summary = {
        'flags': flags,
        'recommended_features': recommended
    }
    return summary

# --------------- Main EDA Logic ----------------
def main():
    # Ensure output directory exists
    output_dir = './output/1'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eda_agent.json')

    eda_results = {}

    # 1. Load data and log metadata
    df = pd.read_csv('data/iris.csv')
    eda_results['dataset_metadata'] = {
        'n_rows': int(df.shape[0]),
        'n_columns': int(df.shape[1]),
        'column_names': df.columns.tolist(),
        'dtypes': {col: str(df[col].dtype) for col in df.columns}
    }

    # 2. Missing values
    missing_by_col = {}
    total_missing = 0
    for col in df.columns:
        num_missing = df[col].isnull().sum()
        percent_missing = (num_missing / len(df)) * 100
        missing_by_col[col] = {
            'count': int(num_missing),
            'percentage': percent_missing
        }
        total_missing += num_missing
    overall_missing_pct = (total_missing / (df.shape[0]*df.shape[1])) * 100
    eda_results['missing_values'] = {
        'missing_values': missing_by_col,
        'overall_percentage': overall_missing_pct
    }

    # 3. Feature types
    feature_types = identify_feature_types(df)
    eda_results['feature_types'] = feature_types

    # 4. Descriptive stats for numerical columns
    eda_results['numerical_stats'] = get_descriptive_stats(df)

    # 5. Categorical stats (including target)
    eda_results['categorical_stats'] = get_categorical_stats(df, feature_types)

    # 6. Categorical cardinality
    cardinality, high_card_cols = check_cardinality(df, feature_types)
    eda_results['categorical_cardinality'] = {
        'cardinality': cardinality,
        'high_cardinality_columns': high_card_cols
    }

    # 7. Outliers
    eda_results['outliers'] = {
        'IQR_method': detect_outliers_iqr(df),
        'zscore_method': detect_outliers_zscore(df)
    }

    # 8. Pearson correlation matrix and high-correlation pairs
    corr_matrix, high_corr_pairs = get_correlation(df)
    eda_results['correlation'] = {
        'pearson_correlation_matrix': corr_matrix,
        'high_correlation_pairs_abs_gt_0.8': high_corr_pairs
    }

    # 9. Histogram bin counts
    eda_results['numerical_histograms'] = get_histogram_counts(df, bins=10)

    # 10. Categorical rare classes (5% threshold) and target class imbalance
    eda_results['rare_classes_and_imbalance'] = rare_classes_and_imbalance(df, feature_types, threshold=0.05, target='variety')

    # 11. Recommended features summary
    eda_results['recommended_features'] = recommended_features(
        df,
        eda_results['missing_values'],
        high_card_cols,
        high_corr_pairs,
        feature_types,
        missing_thresh=0.2
    )

    # Save all results to JSON, ensuring serializability
    with open(output_path, 'w') as f:
        json.dump(eda_results, f, indent=2, default=str)

if __name__ == '__main__':
    main()