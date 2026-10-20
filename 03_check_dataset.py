
# Import required library
import pandas as pd
from pathlib import Path

# Set the dataset path
project_folder = Path(__file__).parent
data_path = project_folder / "data" / "healthcare_disease_prediction_2000.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Define disease targets
target_columns = ["Heart_Disease", "Diabetes", "Stroke"]

# Remove patient ID from input features
feature_columns = [
    column for column in df.columns
    if column not in ["Patient_ID"] + target_columns
]

print("\n========== DATASET CHECK ==========")

# Check duplicate records
print("\nDuplicate complete rows:", df.duplicated().sum())

# Check duplicate patient IDs
print("Duplicate patient IDs:", df["Patient_ID"].duplicated().sum())

# Check correlations between numeric features and targets
print("\n========== FEATURE-TARGET CORRELATIONS ==========")

correlation_data = df[feature_columns + target_columns]
correlations = correlation_data.corr(numeric_only=True)

for target in target_columns:
    print(f"\n{target}:")
    print(
        correlations[target]
        .drop(target_columns, errors="ignore")
        .sort_values(key=abs, ascending=False)
        .round(3)
        .to_string()
    )

# Compare feature averages for positive and negative cases
print("\n========== FEATURE AVERAGES ==========")

for target in target_columns:
    print(f"\n{target}:")

    averages = df.groupby(target)[feature_columns].mean(numeric_only=True)

    print(averages.round(2).to_string())

print("\nDataset checks completed!")
