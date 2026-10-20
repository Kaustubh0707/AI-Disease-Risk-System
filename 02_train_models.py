
# Import required libraries
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Set project paths
project_folder = Path(__file__).parent
data_path = project_folder / "data" / "healthcare_disease_prediction_2000.csv"
models_folder = project_folder / "models"

models_folder.mkdir(exist_ok=True)

# Load the dataset
df = pd.read_csv(data_path)

# Define disease targets
target_columns = [
    "Heart_Disease",
    "Diabetes",
    "Stroke"
]

# Remove patient ID and disease targets from input features
X = df.drop(columns=["Patient_ID"] + target_columns)

# Separate numerical and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# Prepare the input data
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ]
)

# Define the models
model_definitions = {
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

# Store evaluation results
results = []

print("\n========== MODEL TRAINING ==========")

# Train and evaluate each disease separately
for target in target_columns:

    print(f"\nDisease: {target}")

    y = df[target]

    # Split the data while preserving the target proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Train each model
    for model_name, model in model_definitions.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # Generate predictions
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        auc = roc_auc_score(y_test, probabilities)

        # Save the trained model
        model_filename = f"{target}_{model_name}.joblib"
        model_path = models_folder / model_filename

        joblib.dump(pipeline, model_path)

        # Save the evaluation results
        results.append({
            "Disease": target,
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })

        print(f"\nModel: {model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")

# Save all results to a CSV file
results_df = pd.DataFrame(results)

results_path = project_folder / "model_results.csv"
results_df.to_csv(results_path, index=False)

print("\n========== FINAL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nTraining completed successfully!")
print("Saved models folder:", models_folder)
print("Saved results file:", results_path)
