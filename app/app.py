import os
import pandas as pd
import joblib
import streamlit as st

# PAGE CONFIGURATION
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")


# MODEL PATH
MODEL_PATH = "models/fraud_model.pkl"


# LOAD MODEL
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


# PAGE TITLE
st.title("💳 E-Commerce Fraud Detection")
st.write(
    "Enter transaction information "
    "to predict whether the transaction "
    "is fraudulent."
)


# CHECK MODEL
if model is None:
    st.error("Model not found.")
    st.info("Run src/train.py first " "to create models/fraud_model.pkl.")
    st.stop()


# INPUT FORM
st.subheader("Transaction Information")


customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
customer_gender = st.selectbox("Customer Gender", ["Male", "Female"])
account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)

transaction_amount = st.number_input(
    "Transaction Amount", min_value=0.0, value=500.0, step=10.0
)
payment_method = st.selectbox(
    "Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking"]
)
merchant_category = st.selectbox(
    "Merchant Category", ["Shopping", "Food", "Travel", "Electronics", "Other"]
)

device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
previous_transactions = st.number_input("Previous Transactions", min_value=0, value=20)
failed_transactions = st.number_input("Failed Transactions", min_value=0, value=1)


# PREDICTION BUTTON
if st.button("🔍 Check Transaction", use_container_width=True):

    # CREATE INPUT DATAFRAME
    input_data = pd.DataFrame(
        {
            "transaction_amount": [transaction_amount],
            "customer_age": [customer_age],
            "account_age_days": [account_age_days],
            "transactions_last_24h": [previous_transactions],
            "failed_transactions_24h": [failed_transactions],
            "customer_gender": [customer_gender],
            "payment_method": [payment_method],
            "device_type": [device_type],
            "merchant_category": [merchant_category],
        }
    )

    # try:
    # PREDICTION
    prediction = model.predict(input_data)[0]

    # PROBABILITY
    probability = model.predict_proba(input_data)[0][1]

    # DISPLAY RESULT
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: " f"{probability * 100:.2f}%")
# except:
# st.error("⚠️ MODEL CRASH..")
