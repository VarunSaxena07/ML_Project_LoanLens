import streamlit as st
import joblib
import numpy as np

# Load model and scaler (your existing files)
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 LoanLens")
st.write("LoanLens helps assess loan eligibility based on applicant financial data")


# ---------- INPUTS ----------
credit_score = st.number_input("Credit Score", 300, 900)
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
dti_ratio = st.number_input("DTI Ratio")
existing_loans = st.number_input("Existing Loans")
savings = st.number_input("Savings")
collateral_value = st.number_input("Collateral Value")
employment_salaried = st.selectbox("Salaried?", [0, 1])

# ---------- PREDICT ----------
if st.button("Predict Loan Approval"):
    input_data = np.array([[  
    credit_score,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    dti_ratio,
    existing_loans,
    savings,
    collateral_value,
    employment_salaried
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
