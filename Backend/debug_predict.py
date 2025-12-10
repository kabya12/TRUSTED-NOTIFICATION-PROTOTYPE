import traceback, joblib, pandas as pd, os, sys
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "channel_model.joblib")
print("MODEL_PATH ->", MODEL_PATH)
try:
    m = joblib.load(MODEL_PATH)
    print("Loaded model type:", type(m))
    sample = pd.DataFrame([{
        "event_type": "Login OTP",
        "msg_len": len("Your OTP is 123456"),
        "app_installed": int(False),
        "delivery_retry_score": 0
    }])
    print("Sample columns:", list(sample.columns))
    print("Sample dataframe:")
    print(sample)
    print("Calling model.predict(...) now...")
    pred = m.predict(sample)
    print("PREDICTION:", pred)
except Exception:
    print("---- EXCEPTION TRACEBACK ----")
    traceback.print_exc()
    sys.exit(1)
