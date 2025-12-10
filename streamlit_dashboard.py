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
import time
from requests.exceptions import RequestException, ConnectionError

# ---------------- Page config & theme colors ----------------
st.set_page_config(page_title="Trusted Notifications Dashboard", layout="wide")

PRIMARY_BLUE = "#0B5DA7"
ACCENT_ORANGE = "#FF6A00"
NAVY = "#073763"
BG = "#F4F8FB"

# ---------------- Branded header / CSS ----------------
st.markdown(
    f"""
    <style>
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
    .stButton>button, .stDownloadButton>button {{
        background-color: {PRIMARY_BLUE} !important;
        border: none !important;
        color: #fff !important;
    }}
    .main .block-container {{
        background: {BG};
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
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
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

def append_test_event(row: dict, retries: int = 3, delay: float = 0.2) -> bool:
    """
    Append a test event row to CSV. Return True if successful, False otherwise.
    Adds retries and attempts to flush/fsync for better persistence on cloud.
    """
    header = ["timestamp","customer_id","event_type","message","phone","email","app_installed","chosen_channel","provider_message_id","status","otp_demo","replayed_from"]
    for attempt in range(retries):
        try:
            exists = os.path.exists(TEST_EVENTS_CSV)
            with open(TEST_EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # some environments don't allow fsync; ignore
                    pass
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                return False
    return False

def read_test_events() -> pd.DataFrame:
    """
    Read logs robustly. Uses the python engine and on_bad_lines='skip' to handle malformed lines.
    Returns an empty DataFrame with expected columns if reading fails.
    """
    cols = ["timestamp","customer_id","event_type","message","phone","email","app_installed","chosen_channel","provider_message_id","status","otp_demo","replayed_from"]
    if os.path.exists(TEST_EVENTS_CSV):
        try:
            # use python engine and skip bad lines to avoid tokenizing errors when messages contain stray commas
            df = pd.read_csv(TEST_EVENTS_CSV, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
            # ensure expected columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            return df[cols]
        except Exception:
            # fallback empty
            return pd.DataFrame(columns=cols)
    else:
        return pd.DataFrame(columns=cols)

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
st.sidebar.subheader("Filters")
event_types = st.sidebar.multiselect("Event type", options=list(df["Event_Type"].unique()) if df is not None and "Event_Type" in df.columns else [], default=None)
channels = st.sidebar.multiselect("Intended channel", options=list(df["Intended_Channel"].unique()) if df is not None and "Intended_Channel" in df.columns else [], default=None)
delivered = st.sidebar.selectbox("Delivered?", options=["All","Y","N"])
st.sidebar.markdown("---")
st.sidebar.caption("Trusted Notifications — Dashboard prototype")

# ---------------- Main layout / summary ----------------
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

# ---------------- Filters helper ----------------
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

# ---------------- Data preview ----------------
st.markdown("### Data preview")
if df_filtered is None:
    st.info("No dataset loaded.")
else:
    st.dataframe(df_filtered.head(200), use_container_width=True)

# ---------------- Test / simulated send panel ----------------
st.markdown("---")
st.markdown("## Test channel decision & simulate send")
with st.form("test_form"):
    customer_id = st.text_input("Customer ID (optional)", "")

    # autofill contact info if present in dataset
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
                customer_row = matches.iloc[-1]
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
    test_phone = st.text_input("Phone (optional)", value=default_phone)
    test_email = st.text_input("Email (optional)", value=default_email)
    test_app = st.checkbox("App installed?", value=default_app_installed)
    otp_reveal = st.checkbox("Reveal demo OTP (demo-only)", value=False)

    # Optional: use Streamlit secrets for a real backend URL
    default_backend = ""
    try:
        default_backend = st.secrets["BACKEND_URL"]
    except Exception:
        default_backend = ""

    backend_url = st.text_input("Backend URL (leave blank to use simulated fallback)", value=default_backend or "")
    submit = st.form_submit_button("Run prediction & simulate send")

if submit:
    if customer_history is not None:
        st.markdown("### Customer history (from uploaded dataset)")
        st.dataframe(customer_history.tail(10), use_container_width=True)

    # generate demo OTP and include into payload/logs
    otp_value = ""
    if otp_reveal:
        otp_value = f"{random.randint(0,999999):06d}"
        st.warning(f"DEMO OTP (do not use in prod): {otp_value}")

    payload = {
        "event_type": test_event,
        "message": test_msg,
        "contact": {"phone": test_phone or None, "email": test_email or None, "app_installed": bool(test_app)},
        "demo_otp": otp_value or None
    }

    chosen_channel = None
    provider_message_id = None
    status = None

    # Decide if using real backend or simulated fallback (cloud-safe)
    use_simulated = False
    if not backend_url:
        use_simulated = True
        st.info("No backend URL configured — using local simulated fallback.")
    else:
        if backend_url.startswith("http://127.") or backend_url.startswith("http://localhost"):
            if "RUNNING_LOCALLY" not in os.environ:
                use_simulated = True
                st.warning("Backend URL points to localhost. On cloud the app cannot reach localhost — using simulated fallback.")

    if not use_simulated:
        try:
            r = requests.post(f"{backend_url.rstrip('/')}/send-notification", json=payload, timeout=8)
            r.raise_for_status()
            resp = r.json()
            st.success("Backend response:")
            st.json(resp)
            chosen_channel = resp.get("chosen_channel")
            provider_message_id = resp.get("provider_message_id")
            status = resp.get("status")
        except (ConnectionError, RequestException) as e:
            st.error("Backend request failed; using simulated fallback.")
            st.write(str(e))
            use_simulated = True

    if use_simulated:
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
        st.json({"chosen_channel": chosen_channel, "status": status, "provider_message_id": provider_message_id, "note":"simulated fallback used"})

    # log the test event
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
    ok = append_test_event(row)
    if ok:
        st.success("Saved test event to logs/test_events.csv")
        # show quick snapshot download (useful on ephemeral cloud storage)
        logs_df = read_test_events()
        if not logs_df.empty:
            buf = BytesIO()
            logs_df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download latest test_events.csv (snapshot)", data=buf, file_name="test_events.csv", mime="text/csv")
    else:
        st.error("Failed to save test event to logs/test_events.csv")

# ---------------- Unified Logs + Replay module ----------------
st.markdown("---")
st.markdown("## 🔍 Unified Logs & Replay")

# Show logs folder contents (helpful to debug)
if os.path.exists(LOGS_DIR):
    files = os.listdir(LOGS_DIR)
    if files:
        st.write("**Files in logs/**")
        for f in files:
            full = os.path.join(LOGS_DIR, f)
            try:
                size = os.path.getsize(full)
            except Exception:
                size = 0
            st.write(f"- `{f}` — {size} bytes")
    else:
        st.info("Logs folder exists but is empty.")
else:
    st.info("Logs folder not found (it will be created after first test event).")

# Refresh control
if st.button("Refresh logs"):
    # try modern rerun / fallback
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# Read logs (robust)
logs_df = read_test_events()

if logs_df.empty:
    st.info("No test events found yet — run a test notification to generate logs.")
else:
    st.markdown("### 📄 test_events.csv — latest rows")
    # color-code the 'status' column
    def status_color(val):
        v = str(val).lower()
        if v in ("sent","ok"):
            return "background-color: #d4f7d4"  # green-ish
        if "sim" in v or v == "simulated":
            return "background-color: #fff4cc"  # yellow-ish
        if v in ("failed","error","failed_to_send"):
            return "background-color: #ffd6d6"  # red-ish
        return ""

    display_df = logs_df.copy()
    display_df = display_df.reset_index(drop=False).rename(columns={"index":"log_index"})
    # show a styled table in dataframe area (note: Streamlit displays Styler as static HTML in some versions)
    try:
        styled = display_df.tail(200).style.applymap(status_color, subset=["status"])
        st.dataframe(styled, use_container_width=True)
    except Exception:
        # fallback plain
        st.dataframe(display_df.tail(200), use_container_width=True)

    # Download snapshot
    csv_buf = BytesIO()
    logs_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button("Download test_events.csv", data=csv_buf, file_name="test_events.csv", mime="text/csv")

    # Replay controls: operate on the same logs table
    st.markdown("### ▶️ Replay an event from the logs")
    # Build select options: show newest first
    display_df["label"] = display_df.apply(lambda r: f"{r['log_index']} • {r['timestamp']} • cust:{r['customer_id'] or '-'} • {r['event_type']}", axis=1)
    options = display_df["label"].tolist()[::-1]
    selected_label = st.selectbox("Select event to replay", options=options) if options else None

    if selected_label:
        idx = int(selected_label.split(" • ")[0])
        selected_row = display_df[display_df["log_index"] == idx].iloc[0].to_dict()
        st.markdown("Selected event details")
        st.json({k: selected_row.get(k) for k in display_df.columns if k != "label"})

        # backend replay URL input (default from secrets if set)
        default_backend_replay = ""
        try:
            default_backend_replay = st.secrets["BACKEND_URL"]
        except Exception:
            default_backend_replay = ""

        backend_url_replay = st.text_input("Backend URL for replay (leave blank to simulate)", value=default_backend_replay or "")
        if st.button("Replay selected event"):
            replay_payload = {
                "event_type": selected_row.get("event_type", ""),
                "message": selected_row.get("message", ""),
                "contact": {
                    "phone": selected_row.get("phone") or None,
                    "email": selected_row.get("email") or None,
                    "app_installed": True if str(selected_row.get("app_installed")).lower() in ("true","1","yes") else False
                },
                "demo_otp": selected_row.get("otp_demo") or None
            }

            use_simulated_replay = False
            if not backend_url_replay:
                use_simulated_replay = True
            elif backend_url_replay.startswith("http://127.") or backend_url_replay.startswith("http://localhost"):
                if "RUNNING_LOCALLY" not in os.environ:
                    use_simulated_replay = True
                    st.warning("Replay backend points to localhost; using simulated replay on cloud.")

            resp2 = None
            try:
                if not use_simulated_replay:
                    r2 = requests.post(f"{backend_url_replay.rstrip('/')}/send-notification", json=replay_payload, timeout=8)
                    r2.raise_for_status()
                    resp2 = r2.json()
                else:
                    # simulated response
                    if replay_payload["contact"]["app_installed"]:
                        chosen = "Push Notification"
                    elif replay_payload["contact"]["phone"]:
                        chosen = "SMS"
                    elif replay_payload["contact"]["email"]:
                        chosen = "Email"
                    else:
                        chosen = "Email"
                    fake_provider = f"replay-sim-{random.randint(100000,999999)}"
                    resp2 = {"chosen_channel": chosen, "status": "simulated", "provider_message_id": fake_provider}
                    st.json(resp2)

                if resp2 is not None:
                    st.success("Replay response:")
                    st.json(resp2)
                    new_row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "customer_id": int(selected_row.get("customer_id")) if str(selected_row.get("customer_id")).isdigit() else "",
                        "event_type": selected_row.get("event_type", ""),
                        "message": selected_row.get("message", ""),
                        "phone": selected_row.get("phone", ""),
                        "email": selected_row.get("email", ""),
                        "app_installed": selected_row.get("app_installed"),
                        "chosen_channel": resp2.get("chosen_channel", ""),
                        "provider_message_id": resp2.get("provider_message_id", ""),
                        "status": resp2.get("status", ""),
                        "otp_demo": selected_row.get("otp_demo", ""),
                        "replayed_from": selected_row.get("timestamp", "")
                    }
                    ok2 = append_test_event(new_row)
                    if ok2:
                        st.success("Replay saved to test events log.")
                    else:
                        st.error("Failed to save replay to logs.")
            except Exception as e:
                st.error("Replay failed; showing simulated fallback.")
                st.write(str(e))
                # fallback simulated
                if replay_payload["contact"]["app_installed"]:
                    chosen = "Push Notification"
                elif replay_payload["contact"]["phone"]:
                    chosen = "SMS"
                elif replay_payload["contact"]["email"]:
                    chosen = "Email"
                else:
                    chosen = "Email"
                fake_provider = f"replay-sim-{random.randint(100000,999999)}"
                st.json({"chosen_channel": chosen, "status": "simulated", "provider_message_id": fake_provider})
                fallback_row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "customer_id": int(selected_row.get("customer_id")) if str(selected_row.get("customer_id")).isdigit() else "",
                    "event_type": selected_row.get("event_type", ""),
                    "message": selected_row.get("message", ""),
                    "phone": selected_row.get("phone", ""),
                    "email": selected_row.get("email", ""),
                    "app_installed": selected_row.get("app_installed"),
                    "chosen_channel": chosen,
                    "provider_message_id": fake_provider,
                    "status": "simulated",
                    "otp_demo": selected_row.get("otp_demo", ""),
                    "replayed_from": selected_row.get("timestamp", "")
                }
                append_test_event(fallback_row)

# ---------------- Footer guidance ----------------
st.markdown("---")
st.info("Note: Streamlit Cloud filesystem is ephemeral. Download logs after demos or configure remote storage (S3 / DB) for persistence.")
st.caption("Dashboard prototype : Trusted Notifications. Place your datasets under datasets/ or upload via the sidebar.")
