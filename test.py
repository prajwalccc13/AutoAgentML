import pandas as pd

# 1. Load the Excel file
df = pd.read_excel("data\\bima_data.xlsx")

# 2. Get counts of all unique ICD codes
counts = df["ICD_CODE"].value_counts()

# 3. Print total unique counts
print("--- ALL ICD CODE COUNTS ---")
print(counts)
print("\n" + "=" * 30 + "\n")

# 4. Filter and print the lowest 5 (least frequent)
print("--- LOWEST 5 ICD CODES ---")
print(counts.head(100))
