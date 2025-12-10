🔔 Trusted Notifications (TN), Intelligent Multi-Channel Delivery Prototype

A real-time ML + rule-enhanced system for choosing the most reliable customer communication channel.

🌟 Overview

Trusted Notifications is a Machine Learning–powered notification routing engine designed to help banks send alerts through the most effective and reachable channel for each customer.

This prototype combines:

✔ FastAPI Backend with ML inference

✔ Streamlit Frontend Dashboard

✔ EDA & Operational Insights

✔ Customer Lookup + Auto-Fulfilled Details

✔ Channel Prediction + Fallback Logic

✔ Replayable Notification History Logs

The purpose is to increase delivery success, minimize retries, and enhance customer experience through smart channel selection.

✨ Core Features
✅ 1. Multi-Channel Decision Engine (ML + Rules)

Predicts the most reliable channel using:

Event Type

Message Length

App Installed Status

Retry Score

Model: Decision Tree (trained on synthetic-but-realistic behavioural logic)
Channels Supported:
SMS • Email • WhatsApp • Push Notification

Built-in Fallback Rules ensure message delivery is NEVER blocked:

No phone → fallback to Email / Push

No email → fallback to SMS / Push

No app → fallback to SMS / Email

✅ 2. Interactive Streamlit Dashboard (HDFC-themed UI)

A user-friendly control panel for analysts, operations staff, or CRM teams.

Includes:

Upload CSV / Excel datasets

Customer ID search & auto-populate contact details

EDA dashboards

Channel distribution

Retry vs Delivered

Top event types

Data preview tables

Simulate sending notifications (with backend call)

Replay previously sent notifications

Download dataset or test logs

✅ 3. API-Driven Backend (FastAPI)

The backend exposes secure, production-style endpoints:

GET /health

Checks service availability.

POST /send-notification

Accepts:

{
  "event_type": "Login OTP",
  "message": "Your OTP is 123456",
  "contact": {
    "phone": "9876543210",
    "email": null,
    "app_installed": true
  }
}


Returns:

{
  "event_id": "...",
  "chosen_channel": "SMS",
  "provider_message_id": "...",
  "status": "sent"
}


Includes:

ML inference

Channel validation/fallbacks

Simulated provider send (extendable to Twilio / WhatsApp API)

✅ 4. Test Event Logging & Replay Engine

Every test execution is saved to:

logs/test_events.csv


Includes:

Timestamp

Customer ID (optional)

Event Type

Contact info

Chosen Channel

Demo OTP used

Replay source

Replay Feature:
Select any past test → re-run instantly → compare results → log again.

Useful for:

Regression testing

Demonstrations

Behaviour validation after model updates

📊 Data Insights (EDA Module)

The dashboard performs end-to-end operational analysis:

🔹 Channel Distribution

Which channels are mostly used?
Which ones need attention?

🔹 Retry vs Delivered

Evaluates effectiveness of notification attempts.

🔹 Top Event Types

Find commonly triggered events (e.g., Login OTP, KYC Reminder).

🔹 Customer-Level View

Lookup historical patterns for a specific customer.

📁 Project Structure
Trusted_Notifications/
│
├── backend/
│   ├── app.py                     # FastAPI backend (ML + rules)
│   ├── model/
│   │   └── channel_model.joblib   # ML model file
│   └── requirements.txt
│
├── frontend/
│   └── streamlit_dashboard.py     # UI for analytics, testing, replay
│
├── datasets/                      # Sample input datasets
├── logs/                          # Auto-created replay/test logs
│   └── test_events.csv
│
├── README.md
├── requirements.txt
└── .gitignore

⚙️ How to Run Locally

1️⃣ Clone the repository
git clone https://github.com/kabya12/Trusted_Notifications.git
cd Trusted_Notifications

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Start Backend (FastAPI)

From inside backend/:

uvicorn app:app --reload --host 0.0.0.0 --port 8000


Check:

http://127.0.0.1:8000/health

▶️ Start Dashboard (Streamlit UI)

From project root or frontend folder:

streamlit run frontend/streamlit_dashboard.py


App opens at:

http://localhost:8501

🧪 Demo Instructions (Use This for Presentations)

Upload dataset or click Use sample dataset

Explore:

Channel distribution

Retry vs Delivered

Top event types

Search a Customer ID

Auto-filled contact details appear

Choose Event Type → Message → Phone/Email/App flags

Click: Run prediction & simulate send

View:

ML-selected channel

Provider message ID

Real-time status

Scroll to bottom → preview log entries

Replay events to retest or compare model behaviours

🤖 Machine Learning Summary
Model: Decision Tree Classifier
Training Data: 5,000 synthetic rows
Features Used:

Event type

Message length

App installed

Retry score

Performance (synthetic dataset):

Accuracy ≈ 80%

Interpretable, rule-aligned model

⚠️ Limitations

Model uses synthetic labels only

Real delivery outcomes must be used in Phase-2

Behavioural clustering or gradient boosting recommended for production

🔐 Fail-Safe Fallback Logic
Scenario	Action
SMS predicted but no phone	Switch to Email → Push
Push predicted but app not installed	Switch to SMS → Email
Email predicted but no email	Switch to SMS → Push
No contact data at all	API returns 400 with clear message

This ensures 100% notification success.

🛡️ Feasibility, SDLC, and Risks
🔧 Technical Feasibility

Python + FastAPI + Streamlit → lightweight & production-ready

Model is small and fast

Easy integration into existing HDFC notification systems

🌀 SDLC Followed

Requirement Analysis

Feasibility Study

Architecture & Design

Model creation & testing

Backend API development

Streamlit dashboard build

Logging + Replay + EDA

Deployment (Cloud)

Demo preparation

⚠ Risks & Mitigations
Risk	Mitigation
Synthetic training may not generalize	Retrain on live delivery logs
Missing customer contact fields	Input validation + 400 response
Model failure	Fallback rules + replay testing
Data misuse	No PII stored; logs sanitized
💼 Business Value

Ensures reliable delivery of critical alerts (OTP, Fraud, KYC)

Reduces cost by lowering retries

Improves customer engagement

Transparent, explainable logic for audit teams

Dashboard enables operations + CRM insights

🚀 Future Enhancements

Real SMS, WhatsApp, Email API integration

ML upgrade using real engagement data

Personalised channel preference modelling

A/B testing module

Customer segmentation

Admin authentication for dashboard

Containerized deployment (Docker + Kubernetes)

👨‍💻 Author

Kabyashree Boruah
Trusted Notifications Prototype. HDFC Assessment Project

📜 License

This project is for educational and demonstration purposes only.
Not intended for commercial use without permission.