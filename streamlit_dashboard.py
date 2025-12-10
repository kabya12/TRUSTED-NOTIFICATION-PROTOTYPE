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
import traceback
from pathlib import Path

# ---------------- Page config & theme colors ----------------
st.set_page_config(page_title="Trusted Notifications Dashboard", layout="wide")

PRIMARY_BLUE = "#0B5DA7"
ACCENT_ORANGE = "#FF6A00"
NAVY = "#073763"
BG = "#F4F8FB"

# ---------------- Base paths (script dir for stability) ----------------
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEST_EVENTS_CSV = LOGS_DIR / "test_events.csv"

LOG_HEADER = [
    "timestamp", "customer_id", "event_type", "message", "phone", "email",
    "app_installed", "chosen_channel", "provider_message_id", "status", "otp_demo", "replayed_from"
]

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

# ---------------- Helpers ----------------
def do_rerun():
    """Try to rerun Streamlit safely."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            try:
                return pd.read_excel(uploaded_file, engine="openpyxl")
            except ImportError:
                st.error("Reading Excel files requires `openpyxl`. Install it and add to requirements.")
                return None
            except Exception as e:
                st.error(f"Failed to parse Excel file: {e}")
                return None
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

LOG_HEADER = [
    "timestamp", "customer_id", "event_type", "message", "phone", "email",
    "app_installed", "chosen_channel", "provider_message_id", "status", "otp_demo", "replayed_from"
]

def _write_row_to_file(row: dict) -> None:
    """
    Writes a row to test_events.csv.
    Creates file + header automatically.
    Raises exception on failure.
    """
    TEST_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    write_header = not TEST_EVENTS_CSV.exists() or TEST_EVENTS_CSV.stat().st_size == 0

    with open(TEST_EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        clean_row = {k: ("" if row.get(k) is None else str(row.get(k))) for k in LOG_HEADER}
        writer.writerow(clean_row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except:
            pass


def append_test_event(row: dict):
    """
    Writes a row safely.
    If write fails, row is stored in session_state pending buffer.
    Returns (success: bool, message: str)
    """
    try:
        _write_row_to_file(row)
        return True, "OK"

    except Exception as e:
        err = f"{type(e).__name__}: {e}"

        # store in pending buffer
        pend = st.session_state.get("pending_logs", [])
        pend.append({"row": row, "error": err})
        st.session_state["pending_logs"] = pend

        # also write error to logs/errors.log if possible
        try:
            with open(TEST_EVENTS_CSV.parent / "errors.log", "a", encoding="utf-8") as ef:
                ef.write(f"{datetime.utcnow().isoformat()} — {err}\n")
        except:
            pass

        return False, err


def read_test_events_csv() -> pd.DataFrame:
    """Reads CSV safely using csv.DictReader."""
    if not TEST_EVENTS_CSV.exists():
        return pd.DataFrame(columns=LOG_HEADER)

    rows = []
    try:
        with open(TEST_EVENTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: r.get(k, "") for k in LOG_HEADER})
    except:
        return pd.DataFrame(columns=LOG_HEADER)

    return pd.DataFrame(rows, columns=LOG_HEADER)

def flush_pending_logs() -> (int, list): # type: ignore
    """
    Try to flush pending logs from session_state to disk. Returns (count_flushed, failures_list).
    """
    pending = st.session_state.get("pending_logs", [])
    failures = []
    flushed = 0
    if not pending:
        return 0, failures
    # attempt sequential write
    remaining = []
    for item in pending:
        row = item.get("row", {})
        try:
            _write_row_to_file(row)
            flushed += 1
        except Exception as e:
            item["error"] = str(e)
            item["last_try"] = datetime.utcnow().isoformat()
            remaining.append(item)
    st.session_state["pending_logs"] = remaining
    return flushed, remaining

# init pending_logs if not present
if "pending_logs" not in st.session_state:
    st.session_state["pending_logs"] = []

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

# ---------------- Charts / preview ----------------
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
st.markdown("### Data preview")
if df_filtered is None:
    st.info("No dataset loaded.")
else:
    st.dataframe(df_filtered.head(200), use_container_width=True)

# ---------------- Test form ----------------
st.markdown("---")
st.markdown("## Test channel decision & simulate send")
with st.form("test_form"):
    customer_id = st.text_input("Customer ID (optional)", "")

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

    use_simulated = False
    if not backend_url:
        use_simulated = True
        st.info("No backend URL configured — using simulated fallback.")
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
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
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

    # build log row
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

    success, msg = append_test_event(row)
    if success:
        st.success("Saved test event to logs/test_events.csv")
        # quick download snapshot (helpful on ephemeral cloud)
        logs_df = read_test_events_csv()
        if not logs_df.empty:
            buf = BytesIO()
            logs_df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download latest test_events.csv (snapshot)", data=buf, file_name="test_events.csv", mime="text/csv")
    else:
        st.error("Failed to save test event to logs/test_events.csv")
        st.write(f"Error: {msg}")
        pending_count = len(st.session_state.get("pending_logs", []))
        st.warning(f"Stored event in local pending buffer ({pending_count} pending). Use 'Flush pending logs' or download them.")

# ---------------- Logs viewer (simple) ----------------
st.markdown("---")
st.markdown("## 🔍 Test Event Logs (Saved Predictions Only)")

# directory listing
try:
    files = list(LOGS_DIR.iterdir())
    if files:
        st.write("**Files in logs/**")
        for p in sorted(files, key=lambda x: x.name):
            try:
                size = p.stat().st_size
                mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            except Exception:
                size = 0
                mtime = ""
            st.write(f"- `{p.name}` — {size} bytes — last modified: {mtime}")
    else:
        st.info("Logs folder exists but is empty.")
except Exception:
    st.info("Logs folder not available.")

# pending logs controls
pending = st.session_state.get("pending_logs", [])
if pending:
    st.markdown("### ⚠️ Pending logs (not yet written to disk)")
    st.write(f"{len(pending)} rows pending write.")
    if st.button("Flush pending logs"):
        flushed, remaining = flush_pending_logs()
        if flushed:
            st.success(f"Flushed {flushed} pending rows to disk.")
        if remaining:
            st.error(f"{len(remaining)} rows still pending.")
        do_rerun()
    # allow download of pending as CSV
    pending_rows = [p["row"] for p in pending]
    if pending_rows:
        df_pending = pd.DataFrame(pending_rows, columns=LOG_HEADER)
        buf = BytesIO()
        df_pending.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download pending logs as CSV", data=buf, file_name="pending_test_events.csv", mime="text/csv")

# load saved logs
logs_df = read_test_events_csv()
if logs_df.empty:
    st.info("No test events logged yet — run a prediction to generate logs.")
else:
    st.markdown("### 📄 Saved Predictions (Latest rows)")

    def status_color(val):
        v = str(val).lower()
        if v in ("sent", "ok"):
            return "background-color: #d4f7d4"
        if "sim" in v or v == "simulated":
            return "background-color: #fff4cc"
        if v in ("failed", "error", "failed_to_send"):
            return "background-color: #ffd6d6"
        return ""

    view = logs_df.reset_index(drop=False).rename(columns={"index": "log_index"})
    try:
        styled = view.tail(200).style.applymap(status_color, subset=["status"])
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(view.tail(200), use_container_width=True)

    # download button
    csv_buf = BytesIO()
    logs_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button("Download test_events.csv (all logs)", data=csv_buf, file_name="test_events.csv", mime="text/csv")

st.markdown("---")
st.info("Note: On Streamlit Cloud the filesystem is ephemeral. Download logs after demos or configure remote storage (S3/DB) for persistence.")
st.caption("Dashboard prototype : Trusted Notifications. Place your datasets under datasets/ or upload via the sidebar.")
