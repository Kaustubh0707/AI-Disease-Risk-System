
import os
from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    jsonify,
)

from pgmpy.inference import VariableElimination


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(__name__)

DEBUG_MODE = os.environ.get("FLASK_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
}


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

HOME_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Disease Risk Prediction</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            color: #1e293b;
        }
        header {
            background: #172554;
            color: white;
            padding: 25px 8%;
        }
        header h1 { margin: 0 0 10px; }
        header p { margin: 0; color: #dbeafe; }
        nav { margin-top: 20px; }
        nav a {
            color: white;
            margin-right: 20px;
            text-decoration: none;
        }
        main {
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .hero, .card {
            background: white;
            padding: 30px;
            border-radius: 14px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px #0000000a;
        }
        .hero h2 { font-size: 30px; margin-top: 0; }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 20px;
        }
        .card h3 { color: #1d4ed8; }
        .button {
            display: inline-block;
            background: #1d4ed8;
            color: white;
            padding: 12px 18px;
            border-radius: 7px;
            text-decoration: none;
            margin: 8px 8px 0 0;
        }
        .secondary { background: #0f766e; }
        .notice {
            background: #fff7ed;
            padding: 18px;
            border-left: 5px solid #f97316;
            border-radius: 6px;
            line-height: 1.6;
        }
        footer {
            text-align: center;
            padding: 25px;
            color: #64748b;
        }
    </style>
</head>
<body>
    <header>
        <h1>AI Disease Risk Prediction</h1>
        <p>AI-Based Disease Risk Prediction and Decision Support System</p>
        <nav>
            <a href="/">Home</a>
            <a href="/predict">Predict Risk</a>
            <a href="/dashboard">Model Dashboard</a>
        </nav>
    </header>
    <main>
        <section class="hero">
            <h2>Welcome to the Disease Risk Prediction System</h2>
            <p>
                Explore machine learning and Bayesian Network models
                using structured health-related data. Compare model
                performance and view experimental predictions.
            </p>
            <a class="button" href="/predict">Try Prediction</a>
            <a class="button secondary" href="/dashboard">
                View Model Dashboard
            </a>
        </section>
        <section class="cards">
            <div class="card">
                <h3>Heart Disease</h3>
                <p>Explore model predictions for the heart disease target.</p>
            </div>
            <div class="card">
                <h3>Diabetes</h3>
                <p>Explore model predictions for the diabetes target.</p>
            </div>
            <div class="card">
                <h3>Stroke</h3>
                <p>Explore model predictions for the stroke target.</p>
            </div>
        </section>
        <section class="card">
            <h2>Algorithms Used</h2>
            <ul>
                <li>Logistic Regression</li>
                <li>Decision Tree</li>
                <li>Random Forest</li>
                <li>Bayesian Network</li>
            </ul>
        </section>
        <section class="notice">
            <strong>Important disclaimer:</strong>
            This is an academic project. Its dataset and model outputs
            have not been clinically validated. Predictions are not
            medical diagnoses and must not guide real medical decisions.
        </section>
    </main>
    <footer>
        AI-Based Disease Risk Prediction and Decision Support System
    </footer>
</body>
</html>
"""


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def categorize(value, threshold):
    """Convert a numeric value into a Bayesian Network category."""
    return "Higher" if value > threshold else "Lower"


def load_classifier(disease, algorithm):
    """Load a trained machine learning model."""
    model_path = MODELS_DIR / f"{disease}_{algorithm}.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}"
        )

    return joblib.load(model_path)


def predict_with_classifier(model, patient):
    """Return a prediction and its positive-class probability."""
    prediction = int(model.predict(patient)[0])
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(probabilities[classes.index(1)])

    return {
        "prediction": prediction,
        "label": "Positive" if prediction == 1 else "Negative",
        "probability": probability,
        "percentage": (
            probability * 100
            if probability is not None
            else None
        ),
    }


def predict_with_bayesian_network(disease, evidence):
    """Infer the probability of a positive disease state."""
    model_path = MODELS_DIR / f"{disease}_Bayesian_Network.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path.name}"
        )

    network = joblib.load(model_path)
    inference = VariableElimination(network)

    result = inference.query(
        variables=[disease],
        evidence=evidence,
        show_progress=False,
    )

    states = list(result.state_names[disease])

    if "Yes" in states:
        positive_index = states.index("Yes")
    elif 1 in states:
        positive_index = states.index(1)
    else:
        raise ValueError(
            f"Could not identify the positive state for {disease}."
        )

    probability = float(result.values[positive_index])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "label": "Positive" if prediction == 1 else "Negative",
        "probability": probability,
        "percentage": probability * 100,
    }


def build_bayesian_evidence(disease, patient_values):
    """Create evidence matching the trained Bayesian Network."""
    age = patient_values["Age"]
    systolic_bp = patient_values["Blood_Pressure_Systolic"]
    cholesterol = patient_values["Cholesterol"]
    bmi = patient_values["BMI"]
    glucose = patient_values["Glucose_Level"]
    smoking = patient_values["Smoking"]

    if disease == "Heart_Disease":
        return {
            "Age_Group": categorize(age, 60),
            "BP_Group": categorize(systolic_bp, 150),
            "Cholesterol_Group": categorize(cholesterol, 240),
        }

    if disease == "Diabetes":
        return {
            "BMI_Group": categorize(bmi, 30),
            "Glucose_Group": categorize(glucose, 140),
        }

    if disease == "Stroke":
        return {
            "Age_Group": categorize(age, 60),
            "BP_Group": categorize(systolic_bp, 150),
            "Smoking_Group": "Yes" if smoking == 1 else "No",
        }

    raise ValueError(f"Unknown disease: {disease}")


def read_patient_form(form):
    """Read and validate patient form values."""
    try:
        age = int(form["age"])
        gender = form["gender"]
        bmi = float(form["bmi"])
        systolic_bp = float(form["systolic_bp"])
        diastolic_bp = float(form["diastolic_bp"])
        cholesterol = float(form["cholesterol"])
        glucose = float(form["glucose"])
        smoking = int(form["smoking"])
        alcohol = int(form["alcohol"])
        physical_activity = int(form["physical_activity"])
        family_history = int(form["family_history"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Please complete all fields with valid numeric values."
        ) from error

    numeric_ranges = {
        "Age": (age, 0, 120),
        "BMI": (bmi, 10, 100),
        "Systolic blood pressure": (systolic_bp, 40, 300),
        "Diastolic blood pressure": (diastolic_bp, 20, 200),
        "Cholesterol": (cholesterol, 50, 700),
        "Glucose level": (glucose, 20, 800),
    }

    for name, (value, minimum, maximum) in numeric_ranges.items():
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}."
            )

    if gender not in {"Male", "Female"}:
        raise ValueError("Please select a valid gender.")

    binary_values = {
        "Smoking": smoking,
        "Alcohol Intake": alcohol,
        "Physical Activity": physical_activity,
        "Family History": family_history,
    }

    for name, value in binary_values.items():
        if value not in {0, 1}:
            raise ValueError(f"{name} must be 0 or 1.")

    return {
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
        "Family_History": family_history,
    }


def generate_all_predictions(patient_values):
    """Run all four algorithms for each disease target."""
    patient = pd.DataFrame([patient_values])

    diseases = [
        ("Heart_Disease", "Heart Disease"),
        ("Diabetes", "Diabetes"),
        ("Stroke", "Stroke"),
    ]

    algorithms = [
        ("Logistic_Regression", "Logistic Regression"),
        ("Decision_Tree", "Decision Tree"),
        ("Random_Forest", "Random Forest"),
    ]

    results = []

    for disease_key, disease_name in diseases:
        model_results = []

        for algorithm_key, algorithm_name in algorithms:
            model = load_classifier(disease_key, algorithm_key)
            prediction_result = predict_with_classifier(model, patient)

            model_results.append({
                "name": algorithm_name,
                **prediction_result,
            })

        evidence = build_bayesian_evidence(
            disease_key,
            patient_values,
        )

        bayesian_result = predict_with_bayesian_network(
            disease_key,
            evidence,
        )

        model_results.append({
            "name": "Bayesian Network",
            **bayesian_result,
        })

        results.append({
            "name": disease_name,
            "models": model_results,
        })

    return results


# --------------------------------------------------
# HOME ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# --------------------------------------------------
# PREDICTION FORM
# --------------------------------------------------

@app.route("/predict", methods=["GET"])
def prediction_form():
    return render_template("predict.html")


# --------------------------------------------------
# PROCESS PATIENT PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_values = read_patient_form(request.form)
        results = generate_all_predictions(patient_values)

        return render_template(
            "results.html",
            results=results,
            patient=patient_values,
        )

    except ValueError as error:
        return (
            render_template_string(
                """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Invalid Input</title>
                </head>
                <body style="font-family:Arial,sans-serif;padding:30px;">
                    <h2>Unable to process prediction</h2>
                    <p>{{ message }}</p>
                    <a href="/predict">Return to prediction form</a>
                </body>
                </html>
                """,
                message=str(error),
            ),
            400,
        )

    except FileNotFoundError as error:
        app.logger.exception("A required model file was not found.")

        return (
            render_template_string(
                """
                <h2>Model file missing</h2>
                <p>{{ message }}</p>
                <a href="/predict">Return to prediction form</a>
                """,
                message=str(error),
            ),
            500,
        )

    except Exception:
        app.logger.exception("Prediction failed.")

        return (
            """
            <h2>Prediction failed</h2>
            <p>An unexpected error occurred. Check the VS Code terminal
            for details.</p>
            <a href="/predict">Return to prediction form</a>
            """,
            500,
        )


# --------------------------------------------------
# MODEL COMPARISON DASHBOARD
# --------------------------------------------------

@app.route("/dashboard", methods=["GET"])
def dashboard():
    results_file = BASE_DIR / "all_models_comparison.csv"
    cv_file = BASE_DIR / "cross_validation_results.csv"

    if not results_file.is_file():
        return "Model comparison results file was not found.", 404

    results_df = pd.read_csv(results_file)
    results_df = results_df.astype(object).where(
        pd.notna(results_df),
        None,
    )
    results = results_df.to_dict(orient="records")

    cv_results = []

    if cv_file.is_file():
        cv_df = pd.read_csv(cv_file)
        cv_df = cv_df.astype(object).where(
            pd.notna(cv_df),
            None,
        )
        cv_results = cv_df.to_dict(orient="records")

    return render_template(
        "dashboard.html",
        results=results,
        cv_results=cv_results,
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    model_count = len(list(MODELS_DIR.glob("*.joblib")))

    return jsonify({
        "status": "running",
        "model_files_found": model_count,
    })


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=DEBUG_MODE,
        port=5000,
    )
