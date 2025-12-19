import sys
print("python:", sys.executable)
try:
    import streamlit
    print("streamlit:", streamlit.__version__)
except Exception as e:
    print("streamlit import error:", e)
