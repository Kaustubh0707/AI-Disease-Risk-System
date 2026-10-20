
# Import required libraries
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Set the dataset path
project_folder = Path(__file__).parent
data_path = project_folder / "data" / "healthcare_disease_prediction_2000.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Define disease targets
target_columns = ["Heart_Disease", "Diabetes", "Stroke"]

# Prepare input features
X = df.drop(columns=["Patient_ID"] + target_columns)

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# Prepare preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_columns)
])

# Define the models
models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Decision_Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# Create five cross-validation splits
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Select evaluation metrics
metrics = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1_Score": "f1",
    "ROC_AUC": "roc_auc"
}

# Store validation results
results = []

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

# Validate each model for every disease
for target in target_columns:

    print(f"\nDisease: {target}")
    y = df[target]

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cross_validation,
            scoring=metrics,
            n_jobs=1
        )

        result = {
            "Disease": target,
            "Model": model_name
        }

        for metric in metrics:
            score_name = "test_" + metric
            mean_score = scores[score_name].mean()
            std_score = scores[score_name].std()

            result[metric] = round(mean_score, 4)
            result[metric + "_Std"] = round(std_score, 4)

        results.append(result)

        print(f"\nModel: {model_name}")

        for metric in metrics:
            print(
                f"{metric}: {result[metric]:.4f} "
                f"(+/- {result[metric + '_Std']:.4f})"
            )

# Save the validation results
results_df = pd.DataFrame(results)
results_path = project_folder / "cross_validation_results.csv"

results_df.to_csv(results_path, index=False)

print("\n========== VALIDATION SUMMARY ==========")
print(
    results_df[
        ["Disease", "Model", "Accuracy", "Precision",
         "Recall", "F1_Score", "ROC_AUC"]
    ].to_string(index=False)
)

print("\nCross-validation completed!")
print("Results saved to:", results_path)
