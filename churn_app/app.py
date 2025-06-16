import streamlit as st
import joblib
import pandas as pd

# Load model components
model = joblib.load("/mount/src/my-portfolio/churn_app/churn_model.pkl")
scaler = joblib.load("/mount/src/my-portfolio/churn_app/scaler.pkl")
feature_columns = joblib.load("/mount/src/my-portfolio/churn_app/feature_columns.pkl")

st.title("📊 Customer Churn Predictor")

st.markdown("Enter customer details below:")

# Input fields
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Map input to one-hot encoded format
user_input = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0
}

# Fill missing features with 0
for col in feature_columns:
    if col not in user_input:
        user_input[col] = 0

X_input = pd.DataFrame([user_input])[feature_columns]
X_scaled = scaler.transform(X_input)

if st.button("Predict Churn"):
    prediction = model.predict(X_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")