import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Check the number of classes in the DataFrame
num_classes = df.shape[1]

print(f"The number of classes in the CSV dataset is: {num_classes}")