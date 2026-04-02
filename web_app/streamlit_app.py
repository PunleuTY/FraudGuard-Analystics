import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page config 
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants 
MODEL_PATH = r"D:\year 3\Introduction to Data Science\Project (Fraud Detection)\FraudGuard-Analystics\Model\fraud_model.pkl"
SCALER_PATH = r"D:\year 3\Introduction to Data Science\Project (Fraud Detection)\FraudGuard-Analystics\Model\scaler.pkl" # optional but now don't have

FEATURE_ORDER = [
    "amt",
    "hour",
    "day_of_week",
    "is_weekend",
    "distance",
    "transactions_per_card",
    "avg_amt_per_card",
    "unique_merchants_per_card",
    "age",
    "merchant_fraud_rate",
    "merchant_avg_amt",
    "merchant_txn_count",
    "category_fraud_rate",
    "category_count",
]

DAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

#  Model loading (cached) 
@st.cache_resource
def load_model():
    """Load the trained fraud detection model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    """Load the optional scaler from disk. Returns None if not present."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


# Styling 
def inject_css():
    st.markdown(
        """
        <style>
        /* Card-style containers */
        .result-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
        }
        .fraud-card   { background-color: #ffe5e5; color: #c0392b; border: 2px solid #c0392b; }
        .legit-card   { background-color: #e5f5e5; color: #1e8449; border: 2px solid #1e8449; }
        .prob-label   { font-size: 1rem; font-weight: 400; margin-top: 0.4rem; }

        /* Section headers */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #4a4a4a;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 0.3rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Validation 
def validate_inputs(values: dict) -> list[str]:
    """Return a list of validation error messages (empty if all OK)."""
    errors = []
    non_negative = [
        "amt", "distance", "transactions_per_card", "avg_amt_per_card",
        "unique_merchants_per_card", "merchant_avg_amt", "merchant_txn_count",
        "category_count",
    ]
    for field in non_negative:
        if values.get(field, 0) < 0:
            errors.append(f"**{field}** must be ≥ 0.")

    if not (0.0 <= values.get("merchant_fraud_rate", 0) <= 1.0):
        errors.append("**merchant_fraud_rate** must be between 0.0 and 1.0.")
    if not (0.0 <= values.get("category_fraud_rate", 0) <= 1.0):
        errors.append("**category_fraud_rate** must be between 0.0 and 1.0.")
    if not (0 <= values.get("age", 0) <= 120):
        errors.append("**age** must be between 0 and 120.")

    return errors


# Main app 
def main():
    inject_css()
    model = load_model()
    scaler = load_scaler()

    # Header
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown(
        """
        Enter the details of a transaction below and click **Predict** to assess
        whether the transaction is **fraudulent** or **legitimate**.  
        The model returns both a class label and a fraud probability score.
        """
    )
    st.divider()

    # Input form
    with st.form(key="prediction_form"):

        # Row 1: Transaction basics
        st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        amt = col1.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            format="%.2f",
            help="Total monetary value of the transaction.",
        )
        hour = col2.selectbox(
            "Hour of Day",
            options=list(range(24)),
            index=12,
            format_func=lambda h: f"{h:02d}:00",
            help="Hour when the transaction occurred (0 – 23).",
        )
        day_of_week = col3.selectbox(
            "Day of Week",
            options=list(DAY_LABELS.keys()),
            index=0,
            format_func=lambda d: DAY_LABELS[d],
            help="Day on which the transaction occurred.",
        )
        is_weekend = col4.selectbox(
            "Weekend?",
            options=[0, 1],
            index=0,
            format_func=lambda v: "Yes" if v == 1 else "No",
            help="Whether the transaction occurred on a weekend.",
        )

        # Keep is_weekend consistent with day_of_week (soft suggestion only)
        if day_of_week in (5, 6) and is_weekend == 0:
            st.caption("ℹ️ The selected day is a weekend — consider setting Weekend to Yes.")

        st.divider()

        # Row 2: Card-level features 
        st.markdown('<div class="section-header">Card-Level Features</div>', unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)

        distance = col5.number_input(
            "Distance (km)",
            min_value=0.0,
            value=10.0,
            step=0.5,
            format="%.2f",
            help="Distance between the cardholder's home and the merchant.",
        )
        transactions_per_card = col6.number_input(
            "Transactions per Card",
            min_value=0,
            value=5,
            step=1,
            help="Total number of past transactions on this card.",
        )
        avg_amt_per_card = col7.number_input(
            "Avg Amount per Card ($)",
            min_value=0.0,
            value=50.0,
            step=1.0,
            format="%.2f",
            help="Average transaction amount for this card.",
        )
        unique_merchants_per_card = col8.number_input(
            "Unique Merchants per Card",
            min_value=0,
            value=3,
            step=1,
            help="Number of distinct merchants this card has transacted with.",
        )

        st.divider()

        # Row 3: Cardholder features
        st.markdown('<div class="section-header">Cardholder Details</div>', unsafe_allow_html=True)
        col9, col10 = st.columns([1, 3])

        age = col9.number_input(
            "Cardholder Age",
            min_value=0,
            max_value=120,
            value=35,
            step=1,
            help="Age of the cardholder in years.",
        )

        st.divider()

        # Row 4: Merchant features 
        st.markdown('<div class="section-header">Merchant Statistics</div>', unsafe_allow_html=True)
        col11, col12, col13 = st.columns(3)

        merchant_fraud_rate = col11.number_input(
            "Merchant Fraud Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.4f",
            help="Historical fraud rate at this merchant (0 – 1).",
        )
        merchant_avg_amt = col12.number_input(
            "Merchant Avg Amount ($)",
            min_value=0.0,
            value=60.0,
            step=1.0,
            format="%.2f",
            help="Average transaction amount at this merchant.",
        )
        merchant_txn_count = col13.number_input(
            "Merchant Transaction Count",
            min_value=0,
            value=100,
            step=1,
            help="Total number of transactions processed by this merchant.",
        )

        st.divider()

        # Row 5: Category features
        st.markdown('<div class="section-header">Category Statistics</div>', unsafe_allow_html=True)
        col14, col15 = st.columns(2)

        category_fraud_rate = col14.number_input(
            "Category Fraud Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01,
            format="%.4f",
            help="Historical fraud rate for this transaction category (0 – 1).",
        )
        category_count = col15.number_input(
            "Category Transaction Count",
            min_value=0,
            value=200,
            step=1,
            help="Total number of transactions in this category.",
        )

        st.divider()

        # Submit button
        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
            type="primary",
        )

    # Prediction logic
    if submitted:
        user_inputs = {
            "amt": amt,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "distance": distance,
            "transactions_per_card": transactions_per_card,
            "avg_amt_per_card": avg_amt_per_card,
            "unique_merchants_per_card": unique_merchants_per_card,
            "age": age,
            "merchant_fraud_rate": merchant_fraud_rate,
            "merchant_avg_amt": merchant_avg_amt,
            "merchant_txn_count": merchant_txn_count,
            "category_fraud_rate": category_fraud_rate,
            "category_count": category_count,
        }

        # Validate
        errors = validate_inputs(user_inputs)
        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        # Build DataFrame in training feature order
        input_df = pd.DataFrame([user_inputs])[FEATURE_ORDER]

        # Apply scaler if it was used during training
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            input_df = pd.DataFrame(input_scaled, columns=FEATURE_ORDER)

        # Run inference
        try:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            fraud_prob = float(proba[1])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        # Results
        st.subheader("Prediction Result")

        res_col, prob_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(
                    '<div class="result-card fraud-card">'
                    "Fraudulent Transaction"
                    f'<div class="prob-label">Fraud Probability: <strong>{fraud_prob:.2%}</strong></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card legit-card">'
                    "Legitimate Transaction"
                    f'<div class="prob-label">Fraud Probability: <strong>{fraud_prob:.2%}</strong></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )

        with prob_col:
            st.metric(
                label="Fraud Probability Score",
                value=f"{fraud_prob:.4f}",
                delta=f"{'HIGH RISK' if fraud_prob >= 0.5 else 'LOW RISK'}",
                delta_color="inverse",
            )
            # Visual probability bar
            st.progress(
                fraud_prob,
                text=f"Risk level: {fraud_prob:.1%}",
            )

        # Input summary
        with st.expander("Input Summary", expanded=False):
            summary_df = pd.DataFrame(
                list(user_inputs.items()), columns=["Feature", "Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
