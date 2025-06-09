class Prompts:
    def __init__(self):
        self.eda = """
                    You are a Python data scientist. Write python code to perform EDA tasks on the dataset at '{self.dataset_path}'.

                    The plan includes the following steps:
                    - Dataset shape and types
                    - Numeric columns: {plan["columns"]["numeric"]}
                    - Categorical columns: {plan["columns"]["categorical"]}
                    - DateTime columns: {plan["columns"]["datetime"]}
                    - Columns with missing values: {plan["columns"]["missing"]}

                    Based on this, your code should:
                    1. Visualize the numeric data distributions (e.g., histograms, box plots)
                    2. Visualize correlations between numeric features
                    3. Handle missing data (imputation, removal)
                    4. Visualize categorical data distributions (bar plots)
                    5. Visualize DateTime columns (trend, seasonality)
                    6. Generate a markdown report 'report.md' with findings in '{self.output_dir}'

                    Output Python code only. No explanations or extra text.

                    """