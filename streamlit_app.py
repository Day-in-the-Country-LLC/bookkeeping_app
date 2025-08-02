
import app
import streamlit as st
from utils import load_statements

st.title("🧾 Bookkeeping Assistant")

data = load_statements()
if data is not None:
    st.write("Data columns:", data.columns)
    st.write(data.head(10))
    app.main(data)
else:
    st.info("Upload bank statement CSV files to get started.")
