"""
Smart Lender - Flask Application
Epic 5, Story 2 & 3: Build Flask app, integrate model, run/test it.

Run: python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained artifacts (created by train_model.py)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
columns = joblib.load("columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        input_dict = {
            "Gender": form["gender"],
            "Married": form["married"],
            "Dependents": int(form["dependents"]),
            "Education": form["education"],
            "Self_Employed": form["self_employed"],
            "ApplicantIncome": float(form["applicant_income"]),
            "CoapplicantIncome": float(form["coapplicant_income"]),
            "LoanAmount": float(form["loan_amount"]),
            "Loan_Amount_Term": float(form["loan_term"]),
            "Credit_History": float(form["credit_history"]),
            "Property_Area": form["property_area"],
        }

        df_input = pd.DataFrame([input_dict])

        # Apply same label encoders used in training
        for col, le in label_encoders.items():
            if col in df_input.columns:
                df_input[col] = le.transform(df_input[col])

        # Ensure column order matches training
        df_input = df_input[columns]

        # Scale
        scaled_input = scaler.transform(df_input)

        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        # Decode label (Loan_Status was label-encoded too, but not stored
        # since it was the target column, so we map manually: N=0, Y=1)
        result = "Approved" if prediction == 1 else "Rejected"

        return render_template(
            "index.html",
            prediction_text=f"Loan Status: {result} (Confidence: {probability*100:.1f}%)",
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
