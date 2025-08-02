
import app
import streamlit as st
from utils import load_statements

st.title("🧾 Bookkeeping Assistant")

# Let the user specify whether the uploaded statements are personal or business
account_type = st.selectbox("Account type", ["business", "personal"])

data = load_statements()
if data is not None:
    st.write("Data columns:", data.columns)
    st.write(data.head(10))
    app.main(data, account_type=account_type)
else:
    st.info("Upload bank statement CSV files to get started.")
