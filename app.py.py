from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("loan_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        income = int(request.form["income"])
        credit_score = int(request.form["credit_score"])
        loan_amount = int(request.form["loan_amount"])
        employment = request.form["employment"]
        loan_purpose = request.form["loan_purpose"]
        loan_term = int(request.form["loan_term"])
        debt_ratio = float(request.form["debt_ratio"])
        prev_default = request.form["previous_default"]

        # Encode categorical values
        employment_encoded = label_encoders["Employment_Type"].transform([employment])[0]
        loan_purpose_encoded = label_encoders["Loan_Purpose"].transform([loan_purpose])[0]
        prev_default_encoded = label_encoders["Previous_Loan_Default"].transform([prev_default])[0]

        # Prepare input array
        user_data = np.array([[income, credit_score, loan_amount, employment_encoded,
                               loan_term, debt_ratio, prev_default_encoded, loan_purpose_encoded]])

        # Predict
        prediction = model.predict(user_data)

        # Result message
        if prediction[0] == 1:
            result = "✅ Congratulations! Your Loan is Approved."
        else:
            reason = "Low Credit Score" if credit_score < 650 else \
                     "High Debt Ratio" if debt_ratio > 50 else \
                     "Previous Loan Default"
            result = f"❌ Loan Rejected! Reason: {reason}"

        return render_template("index.html", prediction_text=result,
                               income=income, credit_score=credit_score, loan_amount=loan_amount,
                               employment=employment, loan_purpose=loan_purpose,
                               loan_term=loan_term, debt_ratio=debt_ratio,
                               previous_default=prev_default)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
