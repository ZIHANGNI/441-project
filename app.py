import streamlit as st
import joblib
import pandas as pd

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Telecom Churn Predictor')

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input('Tenure (months)', min_value=0)
        contract = st.selectbox('Contract Type',
                                ['Month-to-month', 'One year', 'Two year'])
        internet = st.selectbox('Internet Service',
                                ['DSL', 'Fiber optic', 'No'])

    with col2:
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0)
        total_charges = st.number_input('Total Charges ($)', min_value=0.0)
        payment_method = st.selectbox('Payment Method',
                                      ['Electronic check', 'Mailed check',
                                       'Bank transfer (automatic)',
                                       'Credit card (automatic)'])

    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0,
        'Partner': 0,
        'Dependents': 0,
        'PhoneService': 1,
        'PaperlessBilling': 0,
        'MultipleLines': 0,
        'OnlineSecurity': 0,
        'OnlineBackup': 0,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'gender_Male': 0
    }

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    input_df = pd.DataFrame([input_data])

    trained_features = model.feature_names_in_
    input_df = input_df.reindex(columns=trained_features, fill_value=0)

    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.metric(label="Churn Probability", value=f"{proba:.1%}")
    st.write("Predicted Status:",
             "High Risk (Churn)" if prediction == 1 else "Low Risk (Retained)")

    st.subheader("Key Influencing Factors")
    st.markdown("""
    Based on historical patterns:
    - Longer contract durations reduce churn risk
    - Higher monthly charges correlate with increased churn
    - Electronic check users show higher churn rates
    """)