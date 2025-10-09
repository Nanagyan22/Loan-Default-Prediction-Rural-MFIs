# app.py – Real-Time Loan Default Scoring (Single Entry Form)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os

# ----------------------------
# Load Random Forest model and preprocessor
# ----------------------------
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    rf_pipeline_path = os.path.join(base_path, "artifacts/rf_pipeline.joblib")
    preprocessor_path = os.path.join(base_path, "artifacts/preprocessor.joblib")

    rf_pipeline = joblib.load(rf_pipeline_path)
    preprocessor = joblib.load(preprocessor_path)
    return rf_pipeline, preprocessor

rf_pipeline, preprocessor = load_model()
threshold = 0.062  # Based on threshold tuning

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Loan Default Prediction - Rural MFIs",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
<div style='text-align: center'>
    <h2 style='color: navy;'>🎯 Real-Time Credit Scoring System for MFIs</h2>
    <h4>University of Ghana | Francis Afful Gyan | ID: 22253332</h4>
    <hr style='border-top: 2px solid #bbb;'/>
</div>
""", unsafe_allow_html=True)

st.subheader("📥 Enter Borrower Transactional Details Below")

# ----------------------------
# Borrower Input Form
# ----------------------------
with st.form("borrower_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=35)
        interest_rate = st.slider("Interest Rate (%)", 10.0, 45.0, 27.5)
        loan_amount = st.number_input("Loan Amount (GHS)", 100.0, 50000.0, value=5000.0, step=100.0)
        monthly_income = st.number_input("Monthly Income (GHS)", 100.0, 20000.0, value=1500.0, step=50.0)
        savings_balance = st.number_input("Savings Account Balance (GHS)", 0.0, 50000.0, value=800.0, step=50.0)

    with col2:
        loan_age_days = st.slider("Loan Age (days)", 100, 2000, 500)
        debt_to_income = st.slider("Debt to Income Ratio", 0.05, 2.0, 0.45)
        loan_to_income = st.slider("Loan to Income Ratio", 0.05, 5.0, 0.6)
        income_variability = st.number_input("Business Income Variability", 0.0, 5000.0, value=300.0)
        avg_days_past_due = st.slider("Avg Days Past Due (previous loans)", 0, 90, 3)

    submitted = st.form_submit_button("🔍 Predict Default")

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:
    with st.spinner("🔎 Analyzing Credit Risk..."):

        # Construct input DataFrame
        input_data = pd.DataFrame([{
            "Age": age,
            "Loan_Age_Days": loan_age_days,
            "Loan_Amount": np.log1p(loan_amount),
            "Monthly_Income": np.log1p(monthly_income),
            "Interest_Rate": interest_rate,
            "Debt_to_Income_Ratio": debt_to_income,
            "Savings_Account_Balance": np.log1p(savings_balance),
            "Loan_to_Income_Ratio": loan_to_income,
            "Business_Income_Variability": np.log1p(income_variability),
            "Avg_Days_Past_Due_Previous": np.log1p(avg_days_past_due)
        }])

        # Make sure all features expected by preprocessor exist
        for col in rf_pipeline.named_steps['preprocessor'].get_feature_names_out():
            if col not in input_data.columns:
                input_data[col] = 0

        # Ensure correct column order
        try:
            input_transformed = rf_pipeline.named_steps['preprocessor'].transform(input_data)
        except Exception as e:
            st.error(f"Error in preprocessing input: {e}")
            st.stop()

        # Predict
        try:
            probability = rf_pipeline.named_steps['clf'].predict_proba(input_transformed)[0, 1]
            prediction = 1 if probability >= threshold else 0
            label = "❗ High Risk (Likely Default)" if prediction else "✅ Low Risk (Good Candidate)"

            # Show Results
            st.success("✅ Prediction Completed")
            st.markdown("### 🔍 Prediction Summary")
            st.metric("Prediction", label)
            st.metric("Default Probability", f"{probability * 100:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
<div style="text-align: center; color: gray; font-size: small;">
    Applied Machine Learning Project &middot; University of Ghana &middot; July 2025
</div>
""", unsafe_allow_html=True)
