import streamlit as st
import numpy as np
import joblib

# Load trained model using joblib
model = joblib.load('model/loan_status_predictor.pkl')

# Encoding dictionaries
encoding = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '4': 4},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 2, 'Urban': 1}
}

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")

st.write("Please fill out the details below:")

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '4'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Rural', 'Urban', 'Semiurban'])

# Predict button
if st.button("Predict Loan Status"):
    try:
        # Apply encoding
        input_data = [
            encoding['Gender'][gender],
            encoding['Married'][married],
            encoding['Dependents'][dependents],
            encoding['Education'][education],
            encoding['Self_Employed'][self_employed],
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            encoding['Property_Area'][property_area]
        ]

        input_array = np.array([input_data])
        prediction = model.predict(input_array)

        if prediction[0] == 1:
            st.success("🎉 Loan Approved!")
        else:
            st.error("❌ Loan Rejected")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
