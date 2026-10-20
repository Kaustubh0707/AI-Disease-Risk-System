
from pathlib import Path

import pandas as pd
from flask import Flask, render_template

BASE_DIR = Path(__file__).parent
app = Flask(__name__)

RESULTS_FILE = BASE_DIR / "all_models_comparison.csv"
CV_FILE = BASE_DIR / "cross_validation_results.csv"


@app.route("/dashboard")
def dashboard():
    if not RESULTS_FILE.exists():
        return "Error: all_models_comparison.csv was not found.", 404

    results_df = pd.read_csv(RESULTS_FILE)

    # Replace missing metric values with None for safe display.
    results_df = results_df.astype(object).where(
        pd.notna(results_df), None
    )

    results = results_df.to_dict(orient="records")

    cv_results = []
    if CV_FILE.exists():
        cv_df = pd.read_csv(CV_FILE)
        cv_df = cv_df.astype(object).where(pd.notna(cv_df), None)
        cv_results = cv_df.to_dict(orient="records")

    return render_template(
        "dashboard.html",
        results=results,
        cv_results=cv_results
    )


if __name__ == "__main__":
    app.run(port=5001, debug=True)
