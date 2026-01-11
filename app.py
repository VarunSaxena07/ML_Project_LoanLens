import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 LoanLens")
st.write("Simple Loan Approval Prediction App")

# --- User Inputs ---
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

# Predict button
if st.button("Predict Loan Approval"):
    # Put inputs into array (ORDER MATTERS!)
    input_data = np.array([[applicant_income,
                            coapplicant_income,
                            loan_amount,
                            loan_term,
                            credit_history]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
