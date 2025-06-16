import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📊 Customer Churn Predictor")

st.markdown("Enter customer details below:")

# Collect user input
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Map to dummy variables
contract_map = {
    "Month-to-month": [1, 0],
    "One year": [0, 1],
    "Two year": [0, 0]
}
internet_map = {
    "DSL": [1, 0],
    "Fiber optic": [0, 1],
    "No": [0, 0]
}
payment_map = {
    "Electronic check": [1, 0, 0],
    "Mailed check": [0, 1, 0],
    "Bank transfer (automatic)": [0, 0, 1],
    "Credit card (automatic)": [0, 0, 0]
}

# Construct final input vector
features = [tenure, monthly_charges] +            contract_map[contract] +            internet_map[internet_service] +            payment_map[payment_method]

# Scale numeric input
features_scaled = scaler.transform([features])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(features_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")
