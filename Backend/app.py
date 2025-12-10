# app.py — Trusted Notifications Backend (improved)
import os
import uuid
import time
import logging
import importlib
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trusted_notifications")

# -------------------------------
# COMPATIBILITY SHIM for sklearn pickle
# -------------------------------
try:
    _ct = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList:
            def __init__(self, cols):
                self.cols = cols
            def __repr__(self):
                return f"_RemainderColsList({self.cols!r})"
        setattr(_ct, "_RemainderColsList", _RemainderColsList)
        logger.info("Installed sklearn compatibility shim: _RemainderColsList")
except Exception:
    logger.exception("Failed to attach sklearn shim (continuing, may still work)")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "channel_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"ERROR: Model file not found at {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded: %s", type(model))
except Exception:
    logger.exception("Failed loading model at %s", MODEL_PATH)
    raise

# -------------------------------
# FASTAPI APP INSTANCE + CORS
# -------------------------------
app = FastAPI(title="Trusted Notifications API", version="1.0")

# For demo allow all origins. In production restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# DATA MODELS
# -------------------------------
class Contact(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    device_id: Optional[str] = None
    app_installed: bool = False


class NotificationIn(BaseModel):
    event_type: str
    message: str
    contact: Contact
    event_id: Optional[str] = None


class SendResult(BaseModel):
    event_id: str
    chosen_channel: str
    provider_message_id: str
    status: str


# -------------------------------
# CHANNEL DECISION FUNCTION (safe)
# -------------------------------
def decide_channel(event_type: str, message: str, app_installed: bool):
    # create a DataFrame with the exact feature names your model expects
    sample_df = pd.DataFrame([{
        "event_type": event_type,
        "msg_len": len(message),
        "app_installed": int(app_installed),
        "delivery_retry_score": 0
    }])

    logger.info("decide_channel sample:\n%s", sample_df.to_dict(orient="records"))

    try:
        # many sklearn pipelines expect a DataFrame (or specific columns)
        pred = model.predict(sample_df)[0]
        logger.info("Model prediction: %s", pred)
        return pred
    except Exception:
        logger.exception("Model prediction failed for sample")
        # raise to be handled by endpoint (or we could return a fallback here)
        raise

# -------------------------------
# SIMULATED SENDING
# -------------------------------
def simulate_send(channel: str, contact: dict, message: str):
    provider_message_id = str(uuid.uuid4())
    # simulate latency
    time.sleep(0.1)
    logger.info("[SIMULATED SEND] %s → %s (%d chars)", channel, contact, len(message))
    return provider_message_id, "sent"

# -------------------------------
# ENDPOINTS
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/send-notification", response_model=SendResult)
def send_notification(payload: NotificationIn):
    try:
        if not (payload.contact.phone or payload.contact.email or payload.contact.app_installed):
            raise HTTPException(status_code=400, detail="No contact method available.")

        event_id = payload.event_id or str(uuid.uuid4())

        # Predict with ML model (wrap in try/except)
        try:
            chosen = decide_channel(
                event_type=payload.event_type,
                message=payload.message,
                app_installed=payload.contact.app_installed
            )
        except Exception as e:
            # Log and provide a safe rule-based fallback
            logger.warning("Prediction failed, using rule-based fallback: %s", e)
            if payload.contact.app_installed:
                chosen = "Push Notification"
            elif payload.contact.phone:
                chosen = "SMS"
            elif payload.contact.email:
                chosen = "Email"
            else:
                chosen = "Email"

        # Fallbacks if model picks an invalid channel
        if chosen == "SMS" and not payload.contact.phone:
            chosen = "Email" if payload.contact.email else "Push Notification"

        if chosen == "Push Notification" and not payload.contact.app_installed:
            chosen = "SMS" if payload.contact.phone else "Email"

        # Simulate send
        provider_id, status = simulate_send(
            chosen, payload.contact.dict(), payload.message
        )

        return SendResult(
            event_id=event_id,
            chosen_channel=chosen,
            provider_message_id=provider_id,
            status=status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("send-notification handler failed")
        return JSONResponse(status_code=500, content={"error": "internal server error", "detail": str(e)})
