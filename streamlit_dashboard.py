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

# Lightweight branded header + small CSS
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
    .cta {{
        display:inline-block;
        background:{ACCENT_ORANGE};
        color:white;
        padding:8px 12px;
        border-radius:6px;
        text-decoration:none;
        font-weight:600;
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

# ---------------- Logging utilities (robust) ----------------
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TEST_EVENTS_CSV = os.path.join(LOGS_DIR, "test_events.csv")
_LOG_HEADER = ["timestamp","customer_id","event_type","message","phone","email","app_installed","chosen_channel","provider_message_id","status","otp_demo","replayed_from"]


def append_test_event(row: dict, retries: int = 3, delay: float = 0.2) -> bool:
    """
    Append a test event row to CSV safely:
      - Adds quoting for fields so future reads are clean
      - Retries a few times and fsyncs where possible
    Returns True if write succeeded, False otherwise.
    """
    for attempt in range(retries):
        try:
            exists = os.path.exists(TEST_EVENTS_CSV)
            # open in append mode, ensure newline="" for csv module
            with open(TEST_EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_LOG_HEADER, quoting=csv.QUOTE_ALL)
                if not exists:
                    writer.writeheader()
                # ensure all keys exist
                safe_row = {k: ("" if row.get(k) is None else row.get(k)) for k in _LOG_HEADER}
                writer.writerow(safe_row)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # some hosts don't allow fsync; ignore but continue
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
    Robustly read test_events.csv.
    Repair lines that have extra commas by joining middle fields into 'message' (Option A).
    This handles malformed lines where message contained unescaped commas.
    Returns a DataFrame with expected columns.
    """
    cols = _LOG_HEADER.copy()
    if not os.path.exists(TEST_EVENTS_CSV):
        return pd.DataFrame(columns=cols)

    repaired_rows = []
    try:
        with open(TEST_EVENTS_CSV, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            # Attempt to detect header; if header row matches expected header, skip it
            raw = list(reader)
            if not raw:
                return pd.DataFrame(columns=cols)
            # If first row equals header (case-insensitive), skip it
            first = [c.strip().strip('"').strip("'") for c in raw[0]]
            header_like = all(any(c.lower() == h.lower() for c in first) for h in cols[:len(first)])
            start_idx = 1 if header_like else 0

            for row in raw[start_idx:]:
                # Normalize by trimming leading/trailing whitespace from each cell
                row = [c for c in row]
                if len(row) == len(cols):
                    repaired_rows.append(row)
                elif len(row) > len(cols):
                    # too many fields -> assume extra commas are inside the 'message' field (index 3)
                    # keep first 3 fields, join middle chunk into message, then attach last 8 fields
                    # last fixed fields count:
                    last_count = len(cols) - 4  # number of fields after message (phone..replayed_from) = 8
                    if last_count <= 0:
                        # fallback: join everything after index 3
                        msg = ",".join(row[3:])
                        new_row = row[:3] + [msg] + [""] * (len(cols) - 4) + [""]
                        repaired_rows.append(new_row[:len(cols)])
                    else:
                        # join the middle fields that belong to message
                        prefix = row[:3]
                        suffix = row[-last_count:] if last_count <= len(row) else row[3 + 1:]
                        # message spans row[3:len(row)-last_count]
                        msg_fields = row[3:len(row) - last_count]
                        msg = ",".join(msg_fields)
                        new_row = prefix + [msg] + suffix
                        # pad if still short
                        if len(new_row) < len(cols):
                            new_row += [""] * (len(cols) - len(new_row))
                        repaired_rows.append(new_row[:len(cols)])
                else:
                    # too few fields -> pad with blanks
                    new_row = row + [""] * (len(cols) - len(row))
                    repaired_rows.append(new_row[:len(cols)])
        df = pd.DataFrame(repaired_rows, columns=cols)
        # Try to convert types for known columns
        if "timestamp" in df.columns:
            # keep as string generally; but strip
            df["timestamp"] = df["timestamp"].astype(str)
        return df
    except Exception:
        # fallback to returning an empty DF with expected columns
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
# Filter controls
st.sidebar.subheader("Filters")
event_types = st.sidebar.multiselect(
    "Event type",
    options=list(df["Event_Type"].unique()) if df is not None and "Event_Type" in df.columns else [],
    default=None
)
channels = st.sidebar.multiselect(
    "Intended channel",
    options=list(df["Intended_Channel"].unique()) if df is not None and "Intended_Channel" in df.columns else [],
    default=None
)
delivered = st.sidebar.selectbox("Delivered?", options=["All", "Y", "N"])
st.sidebar.markdown("---")
st.sidebar.caption("Trusted Notifications — Dashboard prototype")

# ---------------- Main layout ----------------
col1, col2 = st.columns([1, 3])
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
        chart_data.columns = ["channel", "count"]
        bar = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("channel:N", sort="-y", title="Channel"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["channel", "count"]
        ).properties(width=700, height=300)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.write("No 'Intended_Channel' column found in dataset.")

colA, colB = st.columns(2)
with colA:
    st.markdown("#### Retry vs Delivered")
    if df is not None and "Retry_YN" in df_filtered.columns and "Delivered_YN" in df_filtered.columns:
        pivot = df_filtered.groupby(["Retry_YN", "Delivered_YN"]).size().reset_index(name="count")
        chart = alt.Chart(pivot).mark_bar().encode(
            x="Retry_YN:N",
            y="count:Q",
            color="Delivered_YN:N",
            tooltip=["Retry_YN", "Delivered_YN", "count"]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Need 'Retry_YN' and 'Delivered_YN' columns for this chart.")

with colB:
    st.markdown("#### Event types (top 10)")
    if df is not None and "Event_Type" in df_filtered.columns:
        top = df_filtered["Event_Type"].value_counts().head(10).reset_index()
        top.columns = ["event", "count"]
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

# ---------------- Model Test / Send Notification panel (cloud-safe) ----------------
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
                customer_row = matches.iloc[-1]
                for col in ["phone", "Phone", "phone_number", "Phone_Number", "mobile", "Mobile"]:
                    if col in customer_row.index and pd.notna(customer_row.get(col)):
                        default_phone = str(customer_row.get(col))
                        break
                for col in ["email", "Email", "e-mail", "E-mail"]:
                    if col in customer_row.index and pd.notna(customer_row.get(col)):
                        default_email = str(customer_row.get(col))
                        break
                for col in ["app_installed", "App_Installed", "AppInstalled"]:
                    if col in customer_row.index:
                        try:
                            default_app_installed = bool(customer_row.get(col))
                        except Exception:
                            default_app_installed = False
        except Exception:
            customer_row = None
            customer_history = None

    test_event = st.selectbox("Event type", options=list(df["Event_Type"].unique()) if df is not None and "Event_Type" in df.columns else ["Login OTP", "Payment Confirmation"])
    test_msg = st.text_area("Message", value="Your OTP is 123456")
    test_phone = st.text_input("Phone (optional)", value=default_phone)
    test_email = st.text_input("Email (optional)", value=default_email)
    test_app = st.checkbox("App installed?", value=default_app_installed)
    otp_reveal = st.checkbox("Reveal demo OTP (demo-only)", value=False)
    # Use Streamlit secret BACKEND_URL if present (recommended)
    default_backend = ""
    try:
        default_backend = st.secrets["BACKEND_URL"]
    except Exception:
        default_backend = ""
    backend_url = st.text_input("Backend URL (leave blank to use simulated fallback)", value=default_backend or "")
    submit = st.form_submit_button("Run prediction & simulate send")

if submit:
    # show customer history if available
    if customer_history is not None:
        st.markdown("### Customer history (from uploaded dataset)")
        st.dataframe(customer_history.tail(10), use_container_width=True)

    # generate demo OTP if user asked for it (6 digit)
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

    # Decide whether to call a real backend or use simulated fallback
    use_simulated = False
    if not backend_url:
        use_simulated = True
        st.info("No backend URL configured — using local simulated fallback.")
    else:
        # prevent cloud trying to call localhost
        if backend_url.startswith("http://127.") or backend_url.startswith("http://localhost"):
            if "RUNNING_LOCALLY" not in os.environ:
                use_simulated = True
                st.warning("Backend URL points to localhost. On Streamlit Cloud the app cannot reach localhost — using simulated fallback.")

    # Try real backend if allowed
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
            st.error("Backend request failed; showing local fallback result.")
            st.write(str(e))
            use_simulated = True

    # Simulated fallback
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
        st.json({"chosen_channel": chosen_channel, "status": status, "provider_message_id": provider_message_id, "note": "simulated fallback used"})

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
    ok = append_test_event(row)
    if ok:
        st.success("Saved test event to logs/test_events.csv")
        # Show quick download button (helps avoid cloud-ephemeral loss)
        logs_df = read_test_events()
        if not logs_df.empty:
            csv_buf = BytesIO()
            logs_df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button("Download latest test_events.csv (recommended on cloud)", data=csv_buf, file_name="test_events.csv", mime="text/csv")
    else:
        st.error("Failed to save test event to logs/test_events.csv")

# ---------------- Bottom logs / export & REPLAY ----------------
st.markdown("---")
st.markdown("### Exports, logs & replay actions")
colx, coly = st.columns([2, 1])
with colx:
    if st.button("Download filtered CSV"):
        if df_filtered is not None and not df_filtered.empty:
            towrite = BytesIO()
            df_filtered.to_csv(towrite, index=False)
            towrite.seek(0)
            st.download_button("Download CSV", data=towrite, file_name="filtered_data.csv", mime="text/csv")
        else:
            st.warning("No data to download.")

    # Re-read logs to ensure latest content
    test_events_df = read_test_events()
    if not test_events_df.empty:
        st.markdown("#### Test events log (tail)")
        st.dataframe(test_events_df.tail(50), use_container_width=True)

        csv_buffer = BytesIO()
        test_events_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button("Download test events CSV", data=csv_buffer, file_name="test_events.csv", mime="text/csv")

        # Replay panel
        st.markdown("#### Replay a saved test event")
        te_df = test_events_df.fillna("")
        te_df["label"] = te_df.apply(lambda r: f"{r.name} • {r['timestamp']} • cust:{r['customer_id']} • {r['event_type']}", axis=1)
        options = te_df["label"].tolist()[::-1]
        selected_label = st.selectbox("Select event to replay", options=options) if options else None

        if selected_label:
            idx = int(selected_label.split(" • ")[0])
            selected_row = te_df.loc[idx].to_dict()
            st.write("Selected event:")
            st.json(selected_row)

            if st.button("Replay selected event"):
                replay_payload = {
                    "event_type": selected_row.get("event_type", ""),
                    "message": selected_row.get("message", ""),
                    "contact": {
                        "phone": selected_row.get("phone") or None,
                        "email": selected_row.get("email") or None,
                        "app_installed": bool(selected_row.get("app_installed"))
                    },
                    "demo_otp": selected_row.get("otp_demo") or None
                }

                default_backend_replay = ""
                try:
                    default_backend_replay = st.secrets["BACKEND_URL"]
                except Exception:
                    default_backend_replay = ""

                backend_url_replay = st.text_input("Backend URL for replay (leave blank to simulate)", value=default_backend_replay or "")
                use_simulated_replay = False
                if not backend_url_replay:
                    use_simulated_replay = True
                elif backend_url_replay.startswith("http://127.") or backend_url_replay.startswith("http://localhost"):
                    if "RUNNING_LOCALLY" not in os.environ:
                        use_simulated_replay = True
                        st.warning("Backend replay URL points to localhost; using simulated replay on cloud.")

                resp2 = None
                try:
                    if not use_simulated_replay:
                        r2 = requests.post(f"{backend_url_replay.rstrip('/')}/send-notification", json=replay_payload, timeout=8)
                        r2.raise_for_status()
                        resp2 = r2.json()
                    else:
                        # simulated response
                        if selected_row.get("app_installed"):
                            chosen = "Push Notification"
                        elif selected_row.get("phone"):
                            chosen = "SMS"
                        elif selected_row.get("email"):
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
                            "app_installed": bool(selected_row.get("app_installed")),
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
                    st.error("Replay failed; showing local fallback result.")
                    st.write(str(e))
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
        # clear loaded dataset only (does not touch logs)
        df = None
        # safe rerun / stop
        try:
            st.experimental_rerun()
        except Exception:
            st.stop()

# ---------------- LOGS VIEWER (Always visible panel) ----------------
st.markdown("---")
st.markdown("## 🔍 Test Event Logs (Internal Viewer)")

try:
    # Show logs directory contents
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
        st.warning("Logs folder not found (it will be created after first test event).")

    # Manual refresh control (helps on Cloud)
    if st.button("Refresh logs viewer"):
        try:
            st.experimental_rerun()
        except Exception:
            st.stop()

    # Show test_events.csv contents if it exists
    if os.path.exists(TEST_EVENTS_CSV):
        st.markdown("### 📄 test_events.csv — Latest 200 rows")
        logs_df = read_test_events()
        if logs_df.empty:
            st.info("Log file exists but contains no readable records yet.")
        else:
            # Add a simple status color mapping and render via Styler if possible
            def status_color(val):
                v = str(val).lower()
                if v in ("sent", "ok"):
                    return "background-color: #d4f7d4"
                if "sim" in v or v == "simulated":
                    return "background-color: #fff4cc"
                if v in ("failed", "error", "failed_to_send"):
                    return "background-color: #ffd6d6"
                return ""

            try:
                styled = logs_df.tail(200).style.applymap(status_color, subset=["status"])
                st.write(styled)  # Streamlit will try to render style; falls back to plain table if unsupported
            except Exception:
                # fallback to plain dataframe display
                st.dataframe(logs_df.tail(200), use_container_width=True)

            # helper: download latest log snapshot
            csv_buf = BytesIO()
            logs_df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button("Download test_events.csv (snapshot)", data=csv_buf, file_name="test_events.csv", mime="text/csv")
    else:
        st.info("test_events.csv not found yet — run a test notification to generate logs.")

    # Note about persistence on cloud
    st.info("Note: On Streamlit Cloud the filesystem is ephemeral. Download logs after demos or configure remote storage (S3/DB) for persistence.")

except Exception as e:
    st.error(f"Error loading logs: {e}")

st.caption("Dashboard prototype : Trusted Notifications. Place your datasets under datasets/ or upload via the sidebar.")
