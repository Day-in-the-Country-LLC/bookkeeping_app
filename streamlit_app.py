
import app
import streamlit as st
from utils import load_statements

st.title("🧾 Bookkeeping Assistant")

data = load_statements()
if not isinstance(data, type(None)):
    st.write("Data columns:", data.columns)
    st.write(data.head(10))
    app.main(data)