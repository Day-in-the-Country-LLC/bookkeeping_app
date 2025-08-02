import pandas as pd
import streamlit as st


def load_statements():
    """Load one or more CSV bank statements uploaded via Streamlit."""

    uploaded_files = st.file_uploader(
        "Upload Bank Statements (CSV)", accept_multiple_files=True
    )
    if uploaded_files:
        dfs = [pd.read_csv(f) for f in uploaded_files]
        data = pd.concat(dfs, ignore_index=True)
        st.write(f"{len(data)} rows loaded from {len(uploaded_files)} files.")
        normed_data = normalize_bank_data(data)
        st.write(f"Columns in normalized data: {normed_data.columns}")
        return normed_data

    st.info("Upload files to begin.")
    return None


def save_table(df, path="data/output_table.csv"):
    """Persist the categorized transactions to disk."""

    df.to_csv(path, index=False)


def load_existing_table(path="data/output_table.csv"):
    """Return the existing categorized transactions table, if it exists."""

    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(
            columns=["payee", "date", "amount", "note", "category"]
        )


def normalize_bank_data(df):
    """Normalize raw bank statement columns to the expected schema."""

    df.reset_index(inplace=True)
    df.columns = [
        "details",
        "date",
        "payee",
        "amount",
        "type",
        "balance",
        "check_num",
        "na",
    ]
    df.drop(columns=["na", "check_num"], inplace=True)

    # Make sure required columns are present
    for col in ["date", "payee", "amount"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "payee", "amount"])

    return df

