
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

project_folder = Path(__file__).parent

training_results_path = project_folder / "model_results.csv"
bayesian_results_path = project_folder / "bayesian_network_results.csv"
cross_validation_path = project_folder / "cross_validation_results.csv"

output_path = project_folder / "all_models_comparison.csv"


# --------------------------------------------------
# 2. Load results from previous scripts
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========\n")

if not training_results_path.exists():
    raise FileNotFoundError(
        "model_results.csv not found. Run 02_train_models.py first."
    )

if not bayesian_results_path.exists():
    raise FileNotFoundError(
        "bayesian_network_results.csv not found. Run 06_bayesian_network.py first."
    )

training_results = pd.read_csv(training_results_path)
bayesian_results = pd.read_csv(bayesian_results_path)

print("Training results loaded successfully.")
print("Bayesian Network results loaded successfully.")


# --------------------------------------------------
# 3. Standardize column names
# --------------------------------------------------

# Rename the AUC column if the training script uses ROC_AUC
if "AUC" in training_results.columns:
    training_results = training_results.rename(
        columns={"AUC": "ROC_AUC"}
    )

# Make sure both result files contain the same metrics
required_columns = [
    "Disease",
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

for column in required_columns:
    if column not in training_results.columns:
        raise ValueError(
            f"Column '{column}' missing from model_results.csv. "
            f"Available columns: {list(training_results.columns)}"
        )

    if column not in bayesian_results.columns:
        raise ValueError(
            f"Column '{column}' missing from bayesian_network_results.csv. "
            f"Available columns: {list(bayesian_results.columns)}"
        )


# --------------------------------------------------
# 4. Combine all model results
# --------------------------------------------------

comparison = pd.concat(
    [
        training_results[required_columns],
        bayesian_results[required_columns]
    ],
    ignore_index=True
)

# Convert metric columns to numeric values
metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

for column in metric_columns:
    comparison[column] = pd.to_numeric(
        comparison[column],
        errors="coerce"
    )

comparison = comparison.sort_values(
    by=["Disease", "F1_Score"],
    ascending=[True, False]
).reset_index(drop=True)


# --------------------------------------------------
# 5. Display the complete comparison
# --------------------------------------------------

print("\n========== ALL MODELS ==========\n")

print(
    comparison.round(4).to_string(index=False)
)


# --------------------------------------------------
# 6. Identify the highest F1 score per disease
# --------------------------------------------------

print("\n========== TOP MODEL PER DISEASE ==========\n")

for disease in comparison["Disease"].unique():

    disease_results = comparison[
        comparison["Disease"] == disease
    ]

    best_model = disease_results.loc[
        disease_results["F1_Score"].idxmax()
    ]

    print(f"Disease: {disease}")
    print(f"Model:   {best_model['Model']}")
    print(f"F1 Score: {best_model['F1_Score']:.4f}")
    print(f"Recall:   {best_model['Recall']:.4f}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print("-" * 40)


# --------------------------------------------------
# 7. Save the comparison report
# --------------------------------------------------

comparison.to_csv(
    output_path,
    index=False
)

print(f"\nComparison saved to:\n{output_path}")
print("\nModel comparison completed successfully!")
