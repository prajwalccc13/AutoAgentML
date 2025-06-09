import pandas as pd

# Load the dataset from the specified path
dataset_path = '/path/to/your/dataset.csv'
data = pd.read_csv(dataset_path)

# Step 1: Dataset shape and types
print("Dataset Shape:", data.shape)
print("Data Types:")
for col in data.dtypes.keys():
    print(col, data[col].dtype)

# Step 2: Numeric columns
numeric_columns = [col for col in data.columns if data[col].dtypes == "int"]
print("\nNumeric Columns:")
for col in numeric_columns:
    print("- ", col, ": ", data[col].describe())

# Step 3: Categorical columns
categorical_columns = [col for col in data.columns if data[col].dtypes!= "object"]
print("\nCategorical Columns:")
for col in categorical_columns:
    print("- ", col, ": ", data[col].value_counts())

# Step 4: Missing data handling
def handle_missing(data):
    # Example of imputation using mean
    data['column_to_impute'] = data['column_to_impute'].fillna(data.mean())
    
    # Example of removing rows with missing values
    data.dropna(inplace=True)
    
    return data

data = handle_missing(data)
print("\nAfter Handling Missing Data:")
print(data.describe())

# Step 5: DateTime columns
def plot_datetime_series(series):
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values)
    plt.title('DateTime Series')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

plot_datetime_series(data['date_column'])