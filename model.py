import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("realistic_loan_dataset.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Employment_Type", "Loan_Purpose", "Previous_Loan_Default"]
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"].map({"Approved": 1, "Rejected": 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model & encoders
joblib.dump(model, "loan_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model trained and saved successfully!")
