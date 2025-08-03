import pandas as pd
import streamlit as st
import re
from llm import normalize_payees as llm_normalize_payees


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
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["payee", "date", "amount", "note", "category"]
        )

    # Ensure the normalized_payee column exists for downstream grouping
    if "payee" in df.columns:
        df["normalized_payee"] = df["payee"].apply(normalize_payee)
        # Use the LLM to further collapse payee variants across the table
        mapping = llm_normalize_payees(df["normalized_payee"].unique().tolist())
        df["normalized_payee"] = df["normalized_payee"].replace(mapping)
    else:
        df["normalized_payee"] = ""

    return df


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


AGGREGATOR_KEYWORDS = ["PAYPAL", "VENMO", "CASH APP", "ZELLE"]


def normalize_payee(payee: str) -> str:
    """Return a normalized version of a payee string for grouping.

    This strips trailing dates and attempts to extract the underlying vendor
    from aggregator services (e.g., PayPal) so we can reuse categories across
    statements.
    """

    if not isinstance(payee, str):
        return payee

    payee = payee.upper()

    # Remove trailing dates like MM/DD for grouping
    payee_no_date = re.sub(r"\s+\d{2}/\d{2}$", "", payee)

    # Attempt to extract vendor from aggregator services before collapsing spaces
    for keyword in AGGREGATOR_KEYWORDS:
        if payee_no_date.startswith(keyword):
            parts = re.split(r"\s{2,}", payee)
            if len(parts) >= 3:
                candidate = re.sub(r"\s+\d{2}/\d{2}$", "", parts[2])
                return candidate.strip()
            return keyword

    # Normalize Amazon Digital purchases which include random codes and phone numbers
    if payee_no_date.startswith(("AMZN DIGITAL", "AMAZON DIGITAL")):
        return "AMZN DIGITAL"

    # Remove ticket numbers, long digit sequences, and phone numbers
    payee_no_date = re.sub(r"#\d+", "", payee_no_date)
    payee_no_date = re.sub(r"\b\d{3}-\d{7}\b", "", payee_no_date)
    payee_no_date = re.sub(r"\b\d{5,}\b", "", payee_no_date)

    # Collapse multiple spaces and trim
    payee_no_date = re.sub(r"\s{2,}", " ", payee_no_date)
    return payee_no_date.strip()


def confirm_category(suggested: str) -> str:
    """Prompt the user to accept or override a suggested category."""

    override = input(
        f"Suggested category '{suggested}'. Press enter to accept or type new category: "
    ).strip()
    return override or suggested


def propagate_vendor_info(df: pd.DataFrame, normalized_payee: str, note: str, category: str) -> pd.DataFrame:
    """Fill note and category for all transactions matching a vendor.

    Parameters
    ----------
    df:
        Transaction table to update.
    normalized_payee:
        Canonical vendor key produced by ``normalize_payee``.
    note:
        User supplied description of the vendor.
    category:
        Bookkeeping category associated with the vendor.

    Returns
    -------
    DataFrame
        Updated transaction table.
    """

    mask = df["normalized_payee"] == normalized_payee
    if mask.any():
        df.loc[mask, ["note", "category"]] = [note, category]
    return df


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate totals by category for tax preparation."""

    if df.empty:
        return pd.DataFrame(columns=["category", "total_amount"])

    summary = df.groupby("category", dropna=False)["amount"].sum().reset_index()
    summary.rename(columns={"amount": "total_amount"}, inplace=True)
    return summary


def save_summary_table(df: pd.DataFrame, path: str = "data/category_summary.csv") -> None:
    """Persist aggregate category totals to disk for a tax preparer."""

    summary = generate_summary(df)
    summary.to_csv(path, index=False)

