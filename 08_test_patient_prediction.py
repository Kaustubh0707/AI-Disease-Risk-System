
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

project_folder = Path(__file__).parent
models_folder = project_folder / "models"


# --------------------------------------------------
# 2. Get patient information
# --------------------------------------------------

print("\n========================================")
print(" AI-BASED DISEASE RISK PREDICTION")
print("========================================")
print("Enter the patient's health information.\n")

age = int(input("Age: "))
gender = input("Gender (Male/Female): ").strip().capitalize()
bmi = float(input("BMI: "))
systolic_bp = float(input("Systolic Blood Pressure: "))
diastolic_bp = float(input("Diastolic Blood Pressure: "))
cholesterol = float(input("Cholesterol: "))
glucose = float(input("Glucose Level: "))
smoking = int(input("Smoking (0 = No, 1 = Yes): "))
alcohol = int(input("Alcohol Intake (0 = No, 1 = Yes): "))
physical_activity = int(
    input("Physical Activity (0 = No, 1 = Yes): ")
)
family_history = int(
    input("Family History (0 = No, 1 = Yes): ")
)


# --------------------------------------------------
# 3. Prepare patient data
# --------------------------------------------------

patient = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Blood_Pressure_Systolic": systolic_bp,
    "Blood_Pressure_Diastolic": diastolic_bp,
    "Cholesterol": cholesterol,
    "Glucose_Level": glucose,
    "Smoking": smoking,
    "Alcohol_Intake": alcohol,
    "Physical_Activity": physical_activity,
    "Family_History": family_history
}])


# --------------------------------------------------
# 4. Load the saved models
# --------------------------------------------------

diseases = [
    "Heart_Disease",
    "Diabetes",
    "Stroke"
]

classifiers = [
    "Logistic_Regression",
    "Decision_Tree",
    "Random_Forest"
]


# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

print("\n========================================")
print(" PREDICTION RESULTS")
print("========================================")

for disease in diseases:

    print(f"\n--- {disease.replace('_', ' ')} ---")

    for classifier in classifiers:

        model_path = (
            models_folder /
            f"{disease}_{classifier}.joblib"
        )

        if not model_path.exists():
            print(f"{classifier}: Model file not found.")
            continue

        model = joblib.load(model_path)

        prediction = model.predict(patient)[0]

        # Get probability of the positive class
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(patient)[0]
            classes = list(model.classes_)

            if 1 in classes:
                positive_index = classes.index(1)
                risk_probability = float(
                    probabilities[positive_index]
                )
            else:
                risk_probability = 0.0

            risk_percentage = risk_probability * 100

        else:
            risk_percentage = None

        result = (
            "Positive class predicted"
            if prediction == 1
            else "Negative class predicted"
        )

        print(f"\nModel: {classifier}")
        print(f"Prediction: {result}")

        if risk_percentage is not None:
            print(
                f"Estimated model probability: "
                f"{risk_percentage:.2f}%"
            )


# --------------------------------------------------
# 6. Bayesian Network predictions
# --------------------------------------------------

print("\n========================================")
print(" BAYESIAN NETWORK ESTIMATES")
print("========================================")

from pgmpy.inference import VariableElimination


def categorize(value, threshold):
    return "Higher" if value > threshold else "Lower"


bayesian_evidence = {
    "Heart_Disease": {
        "Age_Group": categorize(age, 60),
        "BP_Group": categorize(systolic_bp, 150),
        "Cholesterol_Group": categorize(cholesterol, 240)
    },

    "Diabetes": {
        "BMI_Group": categorize(bmi, 30),
        "Glucose_Group": categorize(glucose, 140)
    },

    "Stroke": {
        "Age_Group": categorize(age, 60),
        "BP_Group": categorize(systolic_bp, 150),
        "Smoking_Group": "Yes" if smoking == 1 else "No"
    }
}

for disease in diseases:

    model_path = (
        models_folder /
        f"{disease}_Bayesian_Network.joblib"
    )

    if not model_path.exists():
        print(f"{disease}: Bayesian model file not found.")
        continue

    network = joblib.load(model_path)
    inference = VariableElimination(network)

    probability = inference.query(
        variables=[disease],
        evidence=bayesian_evidence[disease],
        show_progress=False
    )

    states = probability.state_names[disease]
    positive_index = states.index("Yes")

    risk = float(probability.values[positive_index]) * 100

    print(
        f"{disease.replace('_', ' ')}: "
        f"{risk:.2f}% estimated model probability"
    )


# --------------------------------------------------
# 7. Important disclaimer
# --------------------------------------------------

print("\n========================================")
print(" IMPORTANT")
print("========================================")
print(
    "These are experimental predictions from an academic dataset."
)
print(
    "They are not medically validated and must not be used "
    "for diagnosis or treatment decisions."
)
