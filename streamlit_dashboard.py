# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import os
from io import BytesIO
from datetime import datetime
import random
import csv

# ---------------- Page config & theme colors ----------------
st.set_page_config(page_title="Trusted Notifications Dashboard", layout="wide")

PRIMARY_BLUE = "#0B5DA7"
ACCENT_ORANGE = "#FF6A00"
NAVY = "#073763"
BG = "#F4F8FB"

# Inject lightweight header + CSS to give an HDFC-like blue/orange look
st.markdown(
    f"""
    <style>
    /* Load a simple header bar */
    .topbar {{
        background: linear-gradient(90deg, {PRIMARY_BLUE} 0%, {NAVY} 100%);
        padding:14px 22px;
        color: white;
        border-radius: 8px;
        margin-bottom: 18px;
    }}
    .brand {{
        display:flex;
        align-items:center;
        gap:12px;
        font-weight:700;
        font-size:20px;
    }}
    .brand .logo {{
        width:38px;
        height:38px;
        border-radius:6px;
        background: white;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        color: {PRIMARY_BLUE};
        font-weight:800;
        font-size:18px;
    }}
    /* Make buttons and ctas use primary color */
    .stButton>button, .stDownloadButton>button {{
        background-color: {PRIMARY_BLUE} !important;
        border: none !important;
        color: #fff !important;
    }}
    /* subtle card elevation for main container */
    .main .block-container {{
        background: {BG};
    }}
    /* small CTA style for markdown links */
    .cta {{
        display:inline-block;
        background:{ACCENT_ORANGE};
        color:white;
        padding:8px 12px;
        border-radius:6px;
        text-decoration:none;
        font-weight:600;
    }}
    /* Sidebar tint */
    [data-testid="stSidebar"] .css-1d391kg {{
        background: linear-gradient(180deg, {PRIMARY_BLUE} 0%, #0A3F86 100%);
        color: white;
    }}
    [data-testid="stSidebar"] .css-1d391kg label, [data-testid="stSidebar"] .css-1d391kg .stMarkdown {{
        color: white;
    }}
    </style>

    <div class="topbar">
      <div class="brand">
        <div class="logo">TN</div>
        <div>Trusted Notifications Prototype</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helper utilities ----------------
def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        else:
            # try csv fallback
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def load_sample_data():
    candidates = [
        "datasets/Trusted_Notifications_Sample_Events_Updated (1).xlsx",
        "datasets/Trusted_Notifications_Sample_Events_Updated.xlsx",
        "datasets/Event_Type_Stats (1).xlsx",
        "data/sample_events.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.lower().endswith((".xls", ".xlsx")):
                    return pd.read_excel(p)
                else:
                    return pd.read_csv(p)
            except Exception:
                pass
    # fallback synthetic sample
    df = pd.DataFrame({
        "Customer_ID": range(1,101),
        "Event_Type": np.random.choice(["Login OTP","Payment Confirmation","Fraud Alert","KYC Reminder","Credit Card Bill Reminder","Beneficiary Added Alert"], 100),
        "Intended_Channel": np.random.choice(["WhatsApp","SMS","Email","Push Notification"], 100),
        "Delivery_Retry_Score": np.random.randint(0,4,100),
        "Delivered_YN": np.random.choice(["Y","N"], 100, p=[0.8,0.2]),
        "Retry_YN": np.random.choice(["Y","N"], 100, p=[0.35,0.65]),
        "Customer_Action": np.random.choice(["Clicked Link","Ignored","Contacted Support"], 100)
    })
    return df

# ---------------- Logging utilities ----------------
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TEST_EVENTS_CSV = os.path.join(LOGS_DIR, "test_events.csv")

def append_test_event(row: dict):
    header = ["timestamp","customer_id","event_type","message","phone","email","app_installed","chosen_channel","provider_message_id","status","otp_demo","replayed_from"]
    exists = os.path.exists(TEST_EVENTS_CSV)
    with open(TEST_EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def read_test_events():
    if os.path.exists(TEST_EVENTS_CSV):
        try:
            return pd.read_csv(TEST_EVENTS_CSV)
        except Exception:
            return None
    return None

# ---------------- Sidebar controls ----------------
st.sidebar.title("Data & Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel (optional)", type=["csv","xlsx","xls"])
if st.sidebar.button("Use sample dataset"):
    df = load_sample_data()
else:
    df = None

if uploaded_file:
    df2 = read_file(uploaded_file)
    if df2 is not None:
        df = df2

if df is None:
    st.sidebar.info("No dataset loaded. Click 'Use sample dataset' or upload your file.")
else:
    st.sidebar.success("Dataset loaded: %d rows" % len(df))

st.sidebar.markdown("---")
search_id = st.sidebar.text_input("Customer ID (quick search)", "")
st.sidebar.markdown("---")
# Filter controls
st.sidebar.subheader("Filters")
event_types = st.sidebar.multiselect("Event type", options=list(df["Event_Type"].unique()) if df is not None and "Event_Type" in df.columns else [], default=None)
channels = st.sidebar.multiselect("Intended channel", options=list(df["Intended_Channel"].unique()) if df is not None and "Intended_Channel" in df.columns else [], default=None)
delivered = st.sidebar.selectbox("Delivered?", options=["All","Y","N"])
st.sidebar.markdown("---")
st.sidebar.caption("Trusted Notifications — Dashboard prototype")

# ---------------- Main layout ----------------
col1, col2 = st.columns([1,3])
with col1:
    st.markdown("### Summary")
    if df is None:
        st.info("No data to summarise")
    else:
        st.metric("Rows", len(df))
        if "Intended_Channel" in df.columns:
            counts = df["Intended_Channel"].value_counts()
            for ch, c in counts.items():
                st.write(f"- **{ch}**: {int(c)}")
        if "Delivery_Retry_Score" in df.columns:
            st.write(f"Avg retry score: **{df['Delivery_Retry_Score'].mean():.2f}**")

with col2:
    st.markdown("# Early Risk Signals Prototype")
    st.write("Explore the data, test the model, and visualize channel distributions and retry patterns.")
    st.markdown("---")

# Apply filters to dataframe for display & charts
def apply_filters(df):
    if df is None:
        return df
    df2 = df.copy()
    if event_types:
        df2 = df2[df2["Event_Type"].isin(event_types)]
    if channels:
        df2 = df2[df2["Intended_Channel"].isin(channels)]
    if delivered != "All" and "Delivered_YN" in df2.columns:
        df2 = df2[df2["Delivered_YN"] == delivered]
    if search_id:
        try:
            df2 = df2[df2["Customer_ID"] == int(search_id)]
        except Exception:
            pass
    return df2

df_filtered = apply_filters(df)

# ---------------- Charts ----------------
st.markdown("### Channel distribution")
if df is None:
    st.info("Load data to see charts.")
else:
    if "Intended_Channel" in df_filtered.columns:
        chart_data = df_filtered["Intended_Channel"].value_counts().reset_index()
        chart_data.columns = ["channel","count"]
        bar = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("channel:N", sort="-y", title="Channel"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["channel","count"]
        ).properties(width=700, height=300)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.write("No 'Intended_Channel' column found in dataset.")

colA, colB = st.columns(2)
with colA:
    st.markdown("#### Retry vs Delivered")
    if df is not None and "Retry_YN" in df_filtered.columns and "Delivered_YN" in df_filtered.columns:
        pivot = df_filtered.groupby(["Retry_YN","Delivered_YN"]).size().reset_index(name="count")
        chart = alt.Chart(pivot).mark_bar().encode(
            x="Retry_YN:N",
            y="count:Q",
            color="Delivered_YN:N",
            tooltip=["Retry_YN","Delivered_YN","count"]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Need 'Retry_YN' and 'Delivered_YN' columns for this chart.")

with colB:
    st.markdown("#### Event types (top 10)")
    if df is not None and "Event_Type" in df_filtered.columns:
        top = df_filtered["Event_Type"].value_counts().head(10).reset_index()
        top.columns = ["event","count"]
        st.table(top)
    else:
        st.write("No Event_Type column.")

st.markdown("---")

# ---------------- Table display ----------------
st.markdown("### Data preview")
if df_filtered is None:
    st.info("No dataset loaded.")
else:
    st.dataframe(df_filtered.head(200))

# ---------------- Model Test / Send Notification panel ----------------
st.markdown("---")
st.markdown("## Test channel decision & simulate send")

with st.form("test_form"):
    # New: Customer ID input (optional)
    customer_id = st.text_input("Customer ID (optional)", "")

    # Attempt to auto-fill fields if customer_id exists in dataset
    default_phone = ""
    default_email = ""
    default_app_installed = False
    customer_row = None
    customer_history = None
    if customer_id and df is not None:
        try:
            cid = int(customer_id)
            matches = df[df["Customer_ID"] == cid]
            if not matches.empty:
                customer_history = matches.copy()
                # take the most recent row for that customer (or the first)
                customer_row = matches.iloc[-1]
                # try to extract contact fields if available (common names)
                for col in ["phone","Phone","phone_number","Phone_Number","mobile","Mobile"]:
                    if col in customer_row.index and pd.notna(customer_row.get(col)):
                        default_phone = str(customer_row.get(col))
                        break
                for col in ["email","Email","e-mail","E-mail"]:
                    if col in customer_row.index and pd.notna(customer_row.get(col)):
                        default_email = str(customer_row.get(col))
                        break
                for col in ["app_installed","App_Installed","AppInstalled"]:
                    if col in customer_row.index:
                        try:
                            default_app_installed = bool(customer_row.get(col))
                        except Exception:
                            default_app_installed = False
        except Exception:
            customer_row = None
            customer_history = None

    test_event = st.selectbox("Event type", options=list(df["Event_Type"].unique()) if df is not None and "Event_Type" in df.columns else ["Login OTP","Payment Confirmation"])
    test_msg = st.text_area("Message", value="Your OTP is 123456")
    # use defaults (auto-filled) but keep fields editable
    test_phone = st.text_input("Phone (optional)", value=default_phone)
    test_email = st.text_input("Email (optional)", value=default_email)
    test_app = st.checkbox("App installed?", value=default_app_installed)
    # demo-only OTP reveal option
    otp_reveal = st.checkbox("Reveal demo OTP (demo-only)", value=False)
    # backend url inside the form so it is included when submitted
    backend_url = st.text_input("Backend URL (leave default for local)", value="http://127.0.0.1:8000")
    submit = st.form_submit_button("Run prediction & simulate send")

if submit:
    # show customer history if available
    if customer_history is not None:
        st.markdown("### Customer history (from uploaded dataset)")
        st.dataframe(customer_history.tail(10))

    # generate demo OTP if user asked for it (6 digit)
    otp_value = ""
    if otp_reveal:
        otp_value = f"{random.randint(0,999999):06d}"
        st.warning(f"DEMO OTP (do not use in prod): {otp_value}")

    payload = {
        "event_type": test_event,
        "message": test_msg,
        "contact": {"phone": test_phone or None, "email": test_email or None, "app_installed": bool(test_app)}
    }

    chosen_channel = None
    provider_message_id = None
    status = None

    try:
        r = requests.post(f"{backend_url.rstrip('/')}/send-notification", json=payload, timeout=8)
        r.raise_for_status()
        resp = r.json()
        st.success("Backend response:")
        st.json(resp)
        chosen_channel = resp.get("chosen_channel")
        provider_message_id = resp.get("provider_message_id")
        status = resp.get("status")
    except Exception as e:
        st.error("Backend request failed; showing local fallback result.")
        st.write(e)
        # simple fallback decision (rules)
        if test_app:
            chosen_channel = "Push Notification"
        elif test_phone:
            chosen_channel = "SMS"
        elif test_email:
            chosen_channel = "Email"
        else:
            chosen_channel = "Email"
        status = "simulated"
        provider_message_id = f"sim-{random.randint(100000,999999)}"
        st.json({"chosen_channel": chosen_channel, "status": status, "provider_message_id": provider_message_id})

    # append the test event to logs/test_events.csv
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": int(customer_id) if (customer_id and customer_id.isdigit()) else "",
        "event_type": test_event,
        "message": test_msg,
        "phone": test_phone or "",
        "email": test_email or "",
        "app_installed": bool(test_app),
        "chosen_channel": chosen_channel or "",
        "provider_message_id": provider_message_id or "",
        "status": status or "",
        "otp_demo": otp_value or "",
        "replayed_from": ""
    }
    try:
        append_test_event(row)
        st.success("Saved test event to logs/test_events.csv")
    except Exception as e:
        st.error("Failed to save test event:")
        st.write(e)

# ---------------- Bottom logs / export & REPLAY ----------------
st.markdown("---")
st.markdown("### Exports, logs & replay actions")
colx, coly = st.columns([2,1])
with colx:
    if st.button("Download filtered CSV"):
        if df_filtered is not None:
            towrite = BytesIO()
            df_filtered.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button("Download CSV", data=towrite, file_name="filtered_data.csv", mime="text/csv")
        else:
            st.warning("No data to download.")
    # allow downloading logs of past test events
    test_events_df = read_test_events()
    if test_events_df is not None:
        st.markdown("#### Test events log")
        st.dataframe(test_events_df.tail(50))
        csv_buffer = BytesIO()
        test_events_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button("Download test events CSV", data=csv_buffer, file_name="test_events.csv", mime="text/csv")

        # ----------- REPLAY PANEL -----------
        st.markdown("#### Replay a saved test event")
        # prepare options list: show index + timestamp + customer + event_type
        test_events_df = test_events_df.fillna("")
        test_events_df["label"] = test_events_df.apply(
            lambda r: f"{r.name} • {r['timestamp']} • cust:{r['customer_id']} • {r['event_type']}", axis=1
        )
        options = test_events_df["label"].tolist()[::-1]  # newest first
        selected_label = st.selectbox("Select event to replay", options=options) if options else None

        if selected_label:
            # find corresponding row
            idx = int(selected_label.split(" • ")[0])
            selected_row = test_events_df.loc[idx].to_dict()
            st.write("Selected event:")
            st.json(selected_row)

            if st.button("Replay selected event"):
                # Build payload from selected row
                replay_payload = {
                    "event_type": selected_row.get("event_type", ""),
                    "message": selected_row.get("message", ""),
                    "contact": {
                        "phone": selected_row.get("phone") or None,
                        "email": selected_row.get("email") or None,
                        "app_installed": bool(selected_row.get("app_installed"))
                    }
                }
                # call backend
                try:
                    backend_url_replay = st.text_input("Backend URL for replay", value="http://127.0.0.1:8000")
                    r2 = requests.post(f"{backend_url_replay.rstrip('/')}/send-notification", json=replay_payload, timeout=8)
                    r2.raise_for_status()
                    resp2 = r2.json()
                    st.success("Replay backend response:")
                    st.json(resp2)
                    # Append a new log row indicating replay
                    new_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "customer_id": int(selected_row.get("customer_id")) if str(selected_row.get("customer_id")).isdigit() else "",
                        "event_type": selected_row.get("event_type", ""),
                        "message": selected_row.get("message", ""),
                        "phone": selected_row.get("phone", ""),
                        "email": selected_row.get("email", ""),
                        "app_installed": bool(selected_row.get("app_installed")),
                        "chosen_channel": resp2.get("chosen_channel", ""),
                        "provider_message_id": resp2.get("provider_message_id", ""),
                        "status": resp2.get("status", ""),
                        "otp_demo": selected_row.get("otp_demo", ""),
                        "replayed_from": selected_row.get("timestamp", "")
                    }
                    append_test_event(new_row)
                    st.success("Replay saved to test events log.")
                except Exception as e:
                    st.error("Replay failed; showing local fallback result.")
                    st.write(e)
                    # fallback determined from the saved record
                    if selected_row.get("app_installed"):
                        chosen = "Push Notification"
                    elif selected_row.get("phone"):
                        chosen = "SMS"
                    elif selected_row.get("email"):
                        chosen = "Email"
                    else:
                        chosen = "Email"
                    fake_provider = f"replay-sim-{random.randint(100000,999999)}"
                    st.json({"chosen_channel": chosen, "status": "simulated", "provider_message_id": fake_provider})
                    # log fallback replay
                    fallback_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "customer_id": int(selected_row.get("customer_id")) if str(selected_row.get("customer_id")).isdigit() else "",
                        "event_type": selected_row.get("event_type", ""),
                        "message": selected_row.get("message", ""),
                        "phone": selected_row.get("phone", ""),
                        "email": selected_row.get("email", ""),
                        "app_installed": bool(selected_row.get("app_installed")),
                        "chosen_channel": chosen,
                        "provider_message_id": fake_provider,
                        "status": "simulated",
                        "otp_demo": selected_row.get("otp_demo", ""),
                        "replayed_from": selected_row.get("timestamp", "")
                    }
                    append_test_event(fallback_row)

with coly:
    if st.button("Clear dataset"):
        st.experimental_rerun()

st.caption("Dashboard prototype : Trusted Notifications. Place your datasets under datasets/ or upload via the sidebar.")
