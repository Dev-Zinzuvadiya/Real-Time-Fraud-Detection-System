import os
import pandas as pd
import joblib
import streamlit as st

# PAGE CONFIGURATION
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")


# MODEL PATH
MODEL_PATH = "models/fraud_model.pkl"


# LOAD THE PRE-TRAINED MODEL (WITH CACHING)
@st.cache_resource
def load_fraud_model():
    """
    Load the pre-trained fraud detection model.
    Uses Streamlit's cache to avoid reloading on every interaction.

    Returns:
        model: Trained model or None if file not found
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


fraud_model = load_fraud_model()


# DISPLAY APP HEADER
st.title("💳 E-Commerce Fraud Detection")
st.write("""
**Enter transaction information** below to predict whether the transaction
is **fraudulent** or **legitimate**.
""")


# CHECK IF MODEL IS AVAILABLE
if fraud_model is None:
    st.error("⚠️ Model not found.")
    st.info("Please run `src/train.py` first to create `models/fraud_model.pkl`.")
    st.stop()


# ADD SIDEBAR INFORMATION
with st.sidebar:
    st.markdown("## 📋 How It Works")
    st.markdown("""
    1. **Fill in** the transaction details
    2. Click **Check Transaction**
    3. The model predicts if it's **Fraudulent** or **Legitimate**
    """)

    st.markdown("---")
    st.markdown("## 📊 Model Details")
    st.markdown("""
    - **Algorithm**: Machine Learning Classifier
    - **Input Features**: 9 transaction features
    - **Output**: Fraud probability (0-100%)
    """)

    st.markdown("---")
    st.markdown("## ⚠️ Risk Factors")
    st.markdown("""
    - High transaction amount
    - New account
    - Many failed transactions
    - Unusual payment method
    - Suspicious device type
    """)


# TRANSACTION INPUT FORM
st.subheader("📝 Transaction Information")


# ===== CUSTOMER INFORMATION =====
st.markdown("#### 👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Age of the customer in years",
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=365,
        help="How long the account has been active (in days)",
    )

with col2:
    customer_gender = st.selectbox(
        "Customer Gender", ["Male", "Female"], help="Customer's gender"
    )

    device_type = st.selectbox(
        "Device Type",
        ["Mobile", "Desktop", "Tablet"],
        help="Device used for the transaction",
    )


# TRANSACTION INFORMATION
st.markdown("#### 💰 Transaction Details")
col3, col4 = st.columns(2)

with col3:
    transaction_amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        value=500.0,
        step=10.0,
        help="Amount of the transaction in rupees",
    )

    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Debit Card", "UPI", "Net Banking"],
        help="Method used for payment",
    )


with col4:
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Shopping", "Food", "Travel", "Electronics", "Other"],
        help="Category of the merchant",
    )

    previous_transactions = st.number_input(
        "Transactions Last 24h",
        min_value=0,
        value=20,
        help="Number of transactions in the last 24 hours",
    )


# ADDITIONAL INFORMATION
st.markdown("#### ⚠️ Risk Indicators")

failed_transactions = st.number_input(
    "Failed Transactions (Last 24h)",
    min_value=0,
    value=1,
    help="Number of failed transactions in the last 24 hours",
)


# PREDICTION BUTTON AND RESULTS
st.divider()


# PREDICTION BUTTON
if st.button("🔍 Check Transaction", use_container_width=True, type="primary"):

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

    # MAKE PREDICTION
    with st.spinner("🧠 Analyzing transaction..."):
        prediction = fraud_model.predict(input_data)[0]
        probability = fraud_model.predict_proba(input_data)[0][1]

    # DISPLAY RESULTS
    st.divider()
    st.subheader("🎯 Prediction Result")

    # CALCULATE CONFIDENCE
    fraud_probability = probability * 100
    legitimate_probability = (1 - probability) * 100

    # DISPLAY RESULT WITH APPROPRIATE STYLING
    if prediction == 1:
        st.error("🚨 **FRAUDULENT TRANSACTION DETECTED!**")
        st.markdown(
            f"""
        <div style='background-color: #8A1725; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545;'>
            <strong>⚠️ This transaction appears to be fraudulent.</strong><br>
            Please verify with the customer before proceeding.
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ **LEGITIMATE TRANSACTION**")
        st.markdown(
            f"""
        <div style='background-color: #198A34; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745;'>
            <strong>✅ This transaction appears to be legitimate.</strong><br>
            No suspicious activity detected.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # DISPLAY PROBABILITY METRICS
    st.markdown("#### 📊 Probability Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            label="Fraud Probability",
            value=f"{fraud_probability:.2f}%",
            delta="High Risk" if fraud_probability > 50 else "Low Risk",
        )

    with col_b:
        st.metric(
            label="Legitimate Probability",
            value=f"{legitimate_probability:.2f}%",
            delta="Safe" if legitimate_probability > 50 else "Suspicious",
        )

    # DISPLAY RISK LEVEL
    st.markdown("#### 🚦 Risk Level")

    if fraud_probability > 80:
        st.error("🔴 **CRITICAL RISK** - Do not proceed with this transaction")
    elif fraud_probability > 60:
        st.warning("🟠 **HIGH RISK** - Additional verification recommended")
    elif fraud_probability > 40:
        st.warning("🟡 **MEDIUM RISK** - Monitor this transaction")
    elif fraud_probability > 20:
        st.info("🟢 **LOW RISK** - Transaction appears mostly safe")
    else:
        st.success("✅ **MINIMAL RISK** - Transaction is safe")

else:
    # DISPLAY PLACEHOLDER INSTRUCTIONS
    st.info(
        "💡 **Instructions:** Fill in the transaction details above and click 'Check Transaction' to see the prediction."
    )

    # SHOW EXAMPLE SCENARIOS
    with st.expander("📝 View Example Scenarios"):
        st.markdown("""
        **Legitimate Transaction Example:**
        - Customer Age: 35
        - Account Age: 480 days
        - Amount: 47.50
        - Payment: UPI
        - Merchant: Food
        - Device: Mobile
        - Previous Transactions: 15
        - Failed: 0

        **Suspicious Transaction Example:**
        - Customer Age: 22
        - Account Age: 5 days
        - Amount: ₹50,000
        - Payment: UPI
        - Merchant: Electronics
        - Device: Desktop
        - Previous Transactions: 50
        - Failed: 5
        """)


# FOOTER
st.divider()
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Developed with ❤️ using Streamlit | E-Commerce Fraud Detection System</small>
</div>
""",
    unsafe_allow_html=True,
)
