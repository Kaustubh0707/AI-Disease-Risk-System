
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

import joblib


# --------------------------------------------------
# 1. Set project paths and load the dataset
# --------------------------------------------------

project_folder = Path(__file__).parent
data_path = project_folder / "data" / "healthcare_disease_prediction_2000.csv"
models_folder = project_folder / "models"

models_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

print("\n========== BAYESIAN NETWORK ==========")
print(f"Dataset loaded: {df.shape[0]} records")


# --------------------------------------------------
# 2. Convert numerical features into categories
# --------------------------------------------------

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[-float("inf"), 60, float("inf")],
    labels=["Lower", "Higher"]
).astype(str)

df["BP_Group"] = pd.cut(
    df["Blood_Pressure_Systolic"],
    bins=[-float("inf"), 150, float("inf")],
    labels=["Lower", "Higher"]
).astype(str)

df["Cholesterol_Group"] = pd.cut(
    df["Cholesterol"],
    bins=[-float("inf"), 240, float("inf")],
    labels=["Lower", "Higher"]
).astype(str)

df["BMI_Group"] = pd.cut(
    df["BMI"],
    bins=[-float("inf"), 30, float("inf")],
    labels=["Lower", "Higher"]
).astype(str)

df["Glucose_Group"] = pd.cut(
    df["Glucose_Level"],
    bins=[-float("inf"), 140, float("inf")],
    labels=["Lower", "Higher"]
).astype(str)

df["Smoking_Group"] = df["Smoking"].map({
    0: "No",
    1: "Yes"
}).astype(str)


# --------------------------------------------------
# 3. Select features for each disease
# --------------------------------------------------

network_features = {
    "Heart_Disease": [
        "Age_Group",
        "BP_Group",
        "Cholesterol_Group"
    ],

    "Diabetes": [
        "BMI_Group",
        "Glucose_Group"
    ],

    "Stroke": [
        "Age_Group",
        "BP_Group",
        "Smoking_Group"
    ]
}


# --------------------------------------------------
# 4. Train and evaluate each Bayesian Network
# --------------------------------------------------

results = []

for target, features in network_features.items():

    print(f"\n{'=' * 50}")
    print(f"Disease: {target}")
    print(f"{'=' * 50}")

    # Select relevant features and target
    data = df[features + [target]].copy()

    # Convert binary target into categories
    data[target] = data[target].map({
        0: "No",
        1: "Yes"
    })

    # Split into training and testing datasets
    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[target]
    )

    # Create a network where selected features
    # point towards the disease target
    edges = [
        (feature, target)
        for feature in features
    ]

    model = DiscreteBayesianNetwork(edges)

    # Estimate conditional probability distributions
    estimator = BayesianEstimator(
        model,
        train_data
    )

    cpds = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    # Add estimated probability distributions
    model.add_cpds(*cpds)

    # Validate the network structure
    if not model.check_model():
        raise ValueError(
            f"Bayesian Network validation failed for {target}"
        )

    # Create inference engine
    inference = VariableElimination(model)

    # Store actual and predicted values
    y_true = []
    y_pred = []
    y_prob = []

    # Predict each test record
    for _, row in test_data.iterrows():

        evidence = {
            feature: row[feature]
            for feature in features
        }

        probability = inference.query(
            variables=[target],
            evidence=evidence,
            show_progress=False
        )

        # Identify the probability of disease = Yes
        state_names = probability.state_names[target]
        positive_index = state_names.index("Yes")

        risk = float(
            probability.values[positive_index]
        )

        predicted = "Yes" if risk >= 0.5 else "No"

        y_true.append(row[target])
        y_pred.append(predicted)
        y_prob.append(risk)

    # Convert categories into binary values for metrics
    y_true_binary = [
        1 if value == "Yes" else 0
        for value in y_true
    ]

    y_pred_binary = [
        1 if value == "Yes" else 0
        for value in y_pred
    ]

    # Calculate performance metrics
    accuracy = accuracy_score(
        y_true_binary,
        y_pred_binary
    )

    precision = precision_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    recall = recall_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    f1 = f1_score(
        y_true_binary,
        y_pred_binary,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true_binary,
        y_prob
    )

    # Display results
    print(f"Training records: {len(train_data)}")
    print(f"Testing records:  {len(test_data)}")
    print(f"Network edges:    {edges}")

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nConditional Probability Distributions:")

    for cpd in model.get_cpds():
        print(cpd)

    # Store results for the comparison report
    results.append({
        "Disease": target,
        "Model": "Bayesian_Network",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    # Save the trained Bayesian Network
    model_path = (
        models_folder /
        f"{target}_Bayesian_Network.joblib"
    )

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


# --------------------------------------------------
# 5. Save results to a CSV file
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = (
    project_folder /
    "bayesian_network_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n========== FINAL RESULTS ==========")
print(results_df.round(4).to_string(index=False))

print(f"\nResults saved to: {results_path}")
print("\nBayesian Network evaluation completed!")
