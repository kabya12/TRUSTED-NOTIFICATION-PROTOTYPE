# import_test.py — run with the same python used for uvicorn
import importlib.util, traceback, sys, os

p = r"C:\Users\91690\Desktop\Trusted_Notifications\backend\app.py"
spec = importlib.util.spec_from_file_location("app_debug", p)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print("IMPORT OK - app.py executed without raising during import")
except Exception:
    print("IMPORT ERROR — full traceback below:")
    traceback.print_exc()
