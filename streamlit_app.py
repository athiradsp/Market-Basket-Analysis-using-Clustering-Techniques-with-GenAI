"""
Streamlit Cloud Entry Point (Executes app.py on every rerun)
"""
import os
import sys

# Ensure root path is added
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Execute main application script app.py
app_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
with open(app_file, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
