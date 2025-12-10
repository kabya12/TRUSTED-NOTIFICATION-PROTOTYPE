# streamlit_app.py — Trusted Notifications demo UI
import streamlit as st
import requests
import os

# By default use local backend. For Cloud deployment, set BACKEND_URL via Streamlit secrets or env.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Trusted Notifications Demo", layout="centered")
st.title("Trusted Notifications — Demo UI")

st.markdown("""
This demo calls the local FastAPI backend to get the **chosen channel** for a notification,
and simulates sending it. Make sure your backend is running at `http://127.0.0.1:8000` (uvicorn).
""")

with st.form("notif_form"):
    event_type = st.selectbox("Event type", [
        "Login OTP", "Payment Confirmation", "Beneficiary Added Alert",
        "KYC Reminder", "Credit Card Bill Reminder", "Fraud Alert"
    ])
    message = st.text_area("Message", value="Your OTP is 123456")
    phone = st.text_input("Phone (optional)", value="")
    email = st.text_input("Email (optional)", value="")
    app_installed = st.checkbox("App installed?", value=False)
    submitted = st.form_submit_button("Send Notification")

if submitted:
    payload = {
        "event_type": event_type,
        "message": message,
        "contact": {
            "phone": phone if phone.strip() else None,
            "email": email if email.strip() else None,
            "app_installed": bool(app_installed)
        }
    }

    try:
        url = f"{BACKEND_URL.rstrip('/')}/send-notification"
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        st.success("Request sent!")
        st.json(resp.json())
    except requests.exceptions.RequestException as e:
        st.error("Request failed. See details below.")
        st.write("Tried to call backend at:", url)
        st.write(e)
