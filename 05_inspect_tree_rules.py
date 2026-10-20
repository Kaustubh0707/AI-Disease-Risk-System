
# Import required libraries
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Set the dataset path
project_folder = Path(__file__).parent
data_path = project_folder / "data" / "healthcare_disease_prediction_2000.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Define disease targets
targets = ["Heart_Disease", "Diabetes", "Stroke"]

# Prepare input features
X = df.drop(columns=["Patient_ID"] + targets)

# Identify numerical and categorical features
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

# Prepare categorical features
preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_columns),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_columns)
])

print("\n========== DECISION TREE INSPECTION ==========")

# Inspect each disease separately
for target in targets:

    print(f"\n\n========== {target} ==========")

    y = df[target]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create the decision tree pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeClassifier(random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("Training accuracy:",
          round(accuracy_score(y_train, train_predictions), 4))
    print("Testing accuracy:",
          round(accuracy_score(y_test, test_predictions), 4))

    # Display tree complexity
    tree = model.named_steps["tree"]

    print("Tree depth:", tree.get_depth())
    print("Number of leaves:", tree.get_n_leaves())

    # Display the most important features
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = pd.Series(
        tree.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\nTop important features:")
    print(importance.head(10).round(4).to_string())

    # Display a shortened version of the decision rules
    print("\nDecision rules (maximum depth 4):")
    print(export_text(
        tree,
        feature_names=list(feature_names),
        max_depth=4
    ))

print("\nInspection completed!")
