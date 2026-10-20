
# HealthAI — AI-Based Disease Risk Prediction and Decision Support System

## 1. Project Overview

HealthAI is a machine learning-based web application developed to demonstrate disease classification using patient health data.

The system evaluates three disease targets:

- Heart Disease
- Diabetes
- Stroke

It compares four AI algorithms and presents their predictions and model probabilities through an interactive web interface.

**Purpose:** Academic learning and demonstration of Artificial Intelligence and Machine Learning concepts.

## 2. Objectives

- Develop a web-based disease prediction demonstration.
- Apply and compare different classification algorithms.
- Explore probabilistic reasoning using Bayesian Networks.
- Evaluate models using standard performance metrics.
- Present model outputs through an interactive dashboard.

## 3. Technologies Used

| Technology   | Purpose                           |
| ------------ | --------------------------------- |
| Python       | Core programming language         |
| Flask        | Web application framework         |
| Pandas       | Dataset processing                |
| Scikit-learn | Machine learning algorithms       |
| pgmpy        | Bayesian Network inference        |
| HTML and CSS | Web page structure and design     |
| JavaScript   | Interactive functionality         |
| Chart.js     | Dashboard visualizations          |
| Joblib       | Saving and loading trained models |

## 4. Algorithms Implemented

### Logistic Regression

A classification algorithm that estimates the probability of a binary outcome.

### Decision Tree

A tree-based algorithm that makes predictions using a sequence of feature-based decisions.

### Random Forest

An ensemble algorithm that combines multiple decision trees to produce predictions.

### Bayesian Network

A probabilistic graphical model that represents relationships between variables and estimates outcomes using probabilistic inference.

## 5. Dataset

The project uses a dataset containing approximately 2,000 patient records.

Features include:

- Age and gender
- Body Mass Index (BMI)
- Systolic and diastolic blood pressure
- Cholesterol and glucose levels
- Smoking and alcohol intake
- Physical activity
- Family history

Target variables:

- Heart Disease
- Diabetes
- Stroke

Dataset file: `data/healthcare_disease_prediction_2000.csv`

## 6. Main Features

- Patient information form
- Predictions for three disease targets
- Comparison of four classification algorithms
- Model probability visualization
- Interactive model-performance dashboard
- Accuracy, precision, recall, F1-score and ROC-AUC comparison
- Cross-validation results
- Saved models for repeated inference

## 7. Project Structure

```text
AI-Disease-Risk-System/
├── app.py
├── dashboard.py
├── requirements.txt
├── README.md
├── data/
│   └── healthcare_disease_prediction_2000.csv
├── models/
│   ├── Heart_Disease_Logistic_Regression.joblib
│   ├── Heart_Disease_Decision_Tree.joblib
│   ├── Heart_Disease_Random_Forest.joblib
│   ├── Heart_Disease_Bayesian_Network.joblib
│   └── ... other saved models
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── results.html
│   └── dashboard.html
├── static/
│   └── css/
│       └── style.css
├── 01_inspect_dataset.py
├── 02_train_models.py
├── 03_check_dataset.py
├── 04_validate_models.py
├── 05_inspect_tree_rules.py
├── 06_bayesian_network.py
├── 07_compare_all_models.py
├── 08_test_patient_prediction.py
├── model_results.csv
├── cross_validation_results.csv
├── bayesian_network_results.csv
└── all_models_comparison.csv
```

## 8. Installation and Execution

### Prerequisites

- Python 3.12
- pip
- Visual Studio Code (recommended)

### Step 1: Open the project folder

```cmd
cd /d D:\Desktop\AI-Disease-Risk-System
```

### Step 2: Activate the virtual environment

```cmd
.venv\Scripts\activate
```

### Step 3: Install dependencies if needed

```cmd
python -m pip install -r requirements.txt
```

### Step 4: Start the web application

```cmd
python app.py
```

### Step 5: Open the application

Visit these pages in your browser:

- Home: http://127.0.0.1:5000/
- Prediction: http://127.0.0.1:5000/predict
- Dashboard: http://127.0.0.1:5000/dashboard

Keep the terminal running while using the application.

## 9. Model Evaluation

The project records and compares:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, when available
- Five-fold cross-validation metrics

The generated evaluation files are:

- `model_results.csv`
- `cross_validation_results.csv`
- `bayesian_network_results.csv`
- `all_models_comparison.csv`

Model performance should be interpreted in the context of the dataset. High scores do not necessarily indicate real-world generalization.

## 10. Limitations

- The models depend on the quality and representativeness of the available dataset.
- Synthetic or simplified patterns in the data can produce unusually high evaluation scores.
- Model probabilities are not calibrated or clinically validated personal risk estimates.
- The system has not been validated using real-world clinical studies.
- The application is an academic prototype, not a production medical system.

## 11. Future Enhancements

- Validate the models using reliable external datasets.
- Improve model interpretability with feature importance and SHAP.
- Add more rigorous testing and model monitoring.
- Improve security, privacy, and input validation.
- Deploy the application using a production-ready server after appropriate validation.

## 12. Disclaimer

HealthAI is an educational project intended for demonstrating machine learning and probabilistic reasoning. It is not a medical device and must not be used for diagnosis, treatment, or healthcare decisions. Always consult a qualified healthcare professional for medical advice.
