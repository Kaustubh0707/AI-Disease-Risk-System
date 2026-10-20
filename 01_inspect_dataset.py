
from pathlib import Path
import pandas as pd

# Find the project and dataset folders
project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data"

print("Project folder:", project_folder)
print("Data folder:", data_folder)

# Check whether the data folder exists
if not data_folder.exists():
    print("ERROR: The data folder was not found.")
    raise SystemExit

# List CSV files inside the data folder
csv_files = list(data_folder.glob("*.csv"))

print("\nCSV files found:")

for file in csv_files:
    print("-", file.name)

# Stop if no CSV file exists
if not csv_files:
    print("\nERROR: No CSV file found inside the data folder.")
    raise SystemExit

# Load the first CSV file
dataset_path = csv_files[0]
df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("File:", dataset_path.name)
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Check disease target columns
target_columns = ["Heart_Disease", "Diabetes", "Stroke"]

print("\nDisease target distribution:")

for column in target_columns:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))
    else:
        print(f"\nWARNING: Target column '{column}' was not found.")
