"""Streamlit app: predict whether a customer will churn.

Usage:  streamlit run app.py   (run `python src/train.py` first)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

APP_URL = "https://yuxyyy.streamlit.app"
CREATOR = "yuxyyy"

st.set_page_config(page_title="Customer Churn Predictor | yuxyyy.streamlit", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.caption(f"Enter customer details to estimate the probability that they will leave. · by **{CREATOR}** · [{CREATOR}.streamlit.app]({APP_URL})")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
st.sidebar.markdown(f"### 🌐 {CREATOR}.streamlit")
st.sidebar.markdown(f"[{APP_URL}]({APP_URL})")
st.sidebar.divider()
threshold = 0.5
if META_PATH.exists():
    meta = json.loads(META_PATH.read_text())
    threshold = meta.get("threshold", 0.5)
    st.sidebar.markdown(f"**Model:** {meta['best_model']}")
    st.sidebar.markdown(f"**ROC-AUC:** {meta['metrics']['roc_auc']:.3f}")
    st.sidebar.markdown(f"**Recall (churn):** {meta['metrics']['recall']:.3f}")
    st.sidebar.markdown(f"**Decision threshold:** {threshold:.2f}")
    st.sidebar.caption("Trained on the real Telco Customer Churn dataset (7,043 customers).")

YN = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", YN)
    dependents = st.selectbox("Dependents", YN)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", YN)
    lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
with col2:
    security = st.selectbox("Online security", ADDON)
    backup = st.selectbox("Online backup", ADDON)
    protection = st.selectbox("Device protection", ADDON)
    support = st.selectbox("Tech support", ADDON)
    tv = st.selectbox("Streaming TV", ADDON)
    movies = st.selectbox("Streaming movies", ADDON)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", YN)

payment = st.selectbox("Payment method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.number_input("Monthly charges", 10.0, 150.0, 70.0, step=0.5)
total = st.number_input("Total charges", 0.0, 10000.0, float(round(monthly * tenure, 2)), step=10.0)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])
    p = float(model.predict_proba(row)[0, 1])
    st.progress(min(max(p, 0.0), 1.0))
    st.metric("Churn probability", f"{p:.1%}")
    # The model flags a customer as "likely to churn" at the tuned threshold.
    if p >= threshold + 0.2:
        st.error("High risk: consider a retention offer or a longer contract.")
    elif p >= threshold:
        st.warning("Medium risk: flagged as likely to churn, worth a retention check.")
    else:
        st.success("Low risk: customer is likely to stay.")

st.divider()
st.caption(f"Built by {CREATOR} · live at [{CREATOR}.streamlit.app]({APP_URL})")
