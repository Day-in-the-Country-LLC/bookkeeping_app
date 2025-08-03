import pandas as pd
from utils import (
    load_existing_table,
    save_table,
    normalize_payee,
    confirm_category,
    save_summary_table,
    propagate_vendor_info,
)
from llm import categorize_expense, normalize_payees as llm_normalize_payees


def main(data, account_type: str | None = None):
    """Categorize new transactions and save them to the output table.

    This function walks through the following flow:

    1. Load existing categorized transactions.
    2. Normalize payees using both string heuristics and an LLM so that
       transactions from the same vendor are grouped together.
    3. For each normalized payee appearing in the new data, ask the user to
       describe the expense.
    4. Use that description to suggest a bookkeeping category which the user can
       confirm or override.
    5. Record the note and category for all transactions belonging to that
       vendor.
    6. After all transactions have been categorized, aggregate totals by
       category for downstream reporting.

    Parameters
    ----------
    data: DataFrame
        New bank transactions to categorize.
    account_type: str | None, optional
        If provided, read from and write to tables specific to this account
        type (e.g., ``business`` or ``personal``).
    """

    existing = load_existing_table(account_type=account_type)

    # Ensure consistent column types
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # First pass normalization using regex/string rules
    existing["normalized_payee"] = existing["payee"].apply(normalize_payee)
    data["normalized_payee"] = data["payee"].apply(normalize_payee)

    # Use the LLM to collapse similar payees across existing and new data
    all_payees = pd.concat(
        [existing["normalized_payee"], data["normalized_payee"]]
    ).dropna().unique().tolist()
    mapping = llm_normalize_payees(all_payees)
    existing["normalized_payee"] = existing["normalized_payee"].replace(mapping)
    data["normalized_payee"] = data["normalized_payee"].replace(mapping)

    # Avoid re-processing transactions we have already saved
    existing["transaction_key"] = existing.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )
    data["transaction_key"] = data.apply(
        lambda r: f"{r['payee']}|{r['date']}|{r['amount']}", axis=1
    )
    processed_keys = set(existing["transaction_key"])
    unprocessed_data = data[~data["transaction_key"].isin(processed_keys)]

    print(f"{len(unprocessed_data)} new transactions need categorization.\n")

    # Prompt once per normalized vendor
    for payee, group in unprocessed_data.groupby("normalized_payee"):
        display_payee = group.iloc[0]["payee"]
        print(f"\nProcessing '{display_payee}' ({len(group)} transactions)")
        note = input("Describe the expense: ")

        category = categorize_expense(display_payee, group["amount"].mean(), note)
        category = confirm_category(category)

        # Apply the note/category to all transactions in the group
        group = group.assign(note=note, category=category)
        existing = pd.concat([existing, group], ignore_index=True)
        existing = propagate_vendor_info(existing, payee, note, category)
        save_table(existing, account_type=account_type)
        print(
            f"✅ Saved {len(group)} transaction(s) for '{display_payee}' as '{category}'"
        )

    # Once everything is categorized, update the summary table
    save_summary_table(existing, account_type=account_type)

